"""Generic embedding index for diary / fact RAG (long-tail recall).

A flat, in-memory vector store backed by a plain JSON file. Designed for the
hundreds-to-low-thousands scale of one character's diaries or facts, where a
numpy cosine sweep is faster and simpler than any vector database.

One instance per index (e.g. ``diaries.embeddings.json``,
``facts.embeddings.json``). The index stores only ``{id, model, meta, vector}``
— never the source text — so it stays lean and the live content is always read
fresh from disk by the caller. ``meta`` carries lightweight fields (date,
importance, ...) used for filtering/display.

All network calls degrade gracefully: a failed embedding never crashes startup
or blocks a chat turn — the caller simply gets an empty result.
"""

from __future__ import annotations

import json
import math
import os
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger
from openai import AsyncOpenAI

# Embed in chunks so a large backfill stays under the API's per-request limits.
_EMBED_BATCH = 256


def _char_bigrams(text: str) -> set:
    """Character bigrams of a string — a tokenizer-free lexical signal that works
    for Japanese (no word boundaries needed) and degrades gracefully for mixed
    Japanese/English text.

    Normalised first so English/mixed terms match robustly: NFKC folds full-width
    to half-width (ＲＡＧ→RAG, ２→2), ``lower()`` makes it case-insensitive
    (RAG==rag), and whitespace is stripped so word spacing doesn't matter. All
    no-ops for plain Japanese.
    """
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = "".join(t.split())
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _lexical_overlap(query_bigrams: set, text: str) -> float:
    """Fraction of the query's character bigrams present in ``text`` (0..1).

    Dense embeddings on a homogeneous corpus (one character's same-style diaries)
    barely separate even exact keyword matches; this lexical signal rescues them
    — a chunk literally containing the query's words scores near 1.0.
    """
    if not query_bigrams:
        return 0.0
    return len(query_bigrams & _char_bigrams(text)) / len(query_bigrams)


# Lazily-built janome tokenizer for query keyword extraction (denoising).
_JA_TOKENIZER = None
# Common framing / content-light words to drop from extracted keywords. The
# query "…の記憶を思い出してみて" should reduce to its actual content terms.
_KW_STOP = {
    "する", "ある", "いる", "なる", "できる", "みる", "見る", "思う", "思い出す",
    "覚える", "くる", "来る", "いく", "行く", "言う", "話す", "やる", "くれる",
    "こと", "もの", "ため", "よう", "とき", "ところ", "今", "私", "あなた", "君",
    "それ", "これ", "あれ", "どれ", "記憶", "話", "件", "感じ", "気",
    "今日", "昨日", "明日", "日", "時", "最近", "前", "後", "何", "の",
}


def extract_keywords(text: str) -> List[str]:
    """Content keywords from a query — nouns / verbs / adjectives (base form),
    minus particles, pronouns and a stop list — via janome.

    Used to denoise the retrieval query: conversational framing ("思い出して
    みて", "覚えてる？") is dropped so only the actual subject terms drive
    retrieval. English/mixed terms survive (tagged as nouns). Returns ``[]`` if
    janome is unavailable or nothing meaningful remains, so the caller can fall
    back to the raw query. Never raises.
    """
    text = (text or "").strip()
    if not text:
        return []
    global _JA_TOKENIZER
    try:
        if _JA_TOKENIZER is None:
            from janome.tokenizer import Tokenizer

            _JA_TOKENIZER = Tokenizer()
        out: List[str] = []
        for tok in _JA_TOKENIZER.tokenize(text):
            parts = tok.part_of_speech.split(",")
            pos = parts[0]
            sub = parts[1] if len(parts) > 1 else ""
            if pos not in ("名詞", "動詞", "形容詞"):
                continue
            if sub in ("非自立", "代名詞", "接尾", "数"):
                continue
            base = tok.base_form if tok.base_form and tok.base_form != "*" else tok.surface
            if len(base) <= 1 or base in _KW_STOP:
                continue
            out.append(base)
        # De-duplicate, preserving order.
        seen: set = set()
        return [k for k in out if not (k in seen or seen.add(k))]
    except Exception:
        return []


class VectorIndex:
    """Cosine-similarity index over short or paragraph-length texts.

    The on-disk file is a JSON array of entries::

        {"id": "...", "model": "text-embedding-3-small",
         "meta": {"date": "...", "importance": "low"},
         "vector": [0.012, -0.034, ...]}

    Entries embedded with a different model than the one currently configured
    are dropped on load and re-embedded by :meth:`ensure_indexed`, so changing
    the embedding model (or its dimensions) transparently rebuilds the index.
    """

    def __init__(
        self,
        store_path: str,
        *,
        api_key: str,
        base_url: str = "",
        model: str = "text-embedding-3-small",
    ) -> None:
        self._store_path = store_path
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)

        self._ids: List[str] = []
        self._meta: Dict[str, Dict[str, Any]] = {}
        # Source text per id, kept so hybrid retrieval can compute a lexical
        # (keyword-overlap) signal without re-reading the source files.
        self._texts: Dict[str, str] = {}
        # (N, D) float32 of the raw vectors, plus an L2-normalised copy for
        # fast cosine (cosine == dot of normalised vectors).
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._normed: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the index from disk, dropping entries from a stale model."""
        self._ids = []
        self._meta = {}
        self._texts = {}
        vectors: List[List[float]] = []
        if not os.path.isfile(self._store_path):
            self._rebuild_matrix(vectors)
            return
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning(f"[vector_index] Corrupt index {self._store_path}, rebuilding: {e}")
            self._rebuild_matrix([])
            return

        for entry in raw if isinstance(raw, list) else []:
            uid = entry.get("id")
            vec = entry.get("vector")
            if not uid or not vec or entry.get("model") != self._model:
                continue
            self._ids.append(uid)
            self._meta[uid] = entry.get("meta", {})
            self._texts[uid] = entry.get("text", "")
            vectors.append(vec)
        self._rebuild_matrix(vectors)
        logger.debug(
            f"[vector_index] Loaded {len(self._ids)} vectors from {self._store_path}"
        )

    def _rebuild_matrix(self, vectors: Sequence[Sequence[float]]) -> None:
        if len(vectors):
            self._vectors = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._normed = self._vectors / norms
        else:
            self._vectors = np.zeros((0, 0), dtype=np.float32)
            self._normed = np.zeros((0, 0), dtype=np.float32)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._store_path) or ".", exist_ok=True)
        out = [
            {
                "id": uid,
                "model": self._model,
                "meta": self._meta.get(uid, {}),
                "text": self._texts.get(uid, ""),
                "vector": self._vectors[i].tolist(),
            }
            for i, uid in enumerate(self._ids)
        ]
        tmp = self._store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        os.replace(tmp, self._store_path)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts in batches. Raises on API failure."""
        out: List[List[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            chunk = texts[start : start + _EMBED_BATCH]
            resp = await self._client.embeddings.create(model=self._model, input=chunk)
            # Response preserves input order.
            out.extend(d.embedding for d in resp.data)
        return np.asarray(out, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __contains__(self, uid: str) -> bool:
        return uid in self._meta

    def size(self) -> int:
        return len(self._ids)

    async def ensure_indexed(self, items: List[Dict[str, Any]]) -> None:
        """Embed any missing items and drop vectors whose item no longer exists.

        ``items`` is ``[{"id", "text", "meta"}, ...]`` — the full current set.
        Missing or stale-model ids get embedded (batched); ids present in the
        index but absent from ``items`` are pruned. No-op when already in sync.
        """
        try:
            wanted = {it["id"]: it for it in items if it.get("id") and it.get("text")}
            have = set(self._ids)

            # Prune orphans (diary/fact deleted while its vector lingered).
            removed = have - set(wanted)
            if removed:
                self._drop(removed)

            # Backfill source text onto entries indexed before text was stored
            # (migration for hybrid retrieval), without re-embedding.
            filled = 0
            for uid, it in wanted.items():
                if uid in self._meta and not self._texts.get(uid):
                    self._texts[uid] = it.get("text", "")
                    self._meta[uid] = it.get("meta", self._meta[uid])
                    filled += 1

            missing = [uid for uid in wanted if uid not in self._meta]
            if not missing:
                if removed or filled:
                    self._save()
                return

            texts = [wanted[uid]["text"] for uid in missing]
            vecs = await self._embed(texts)
            for uid, vec in zip(missing, vecs):
                self._append(
                    uid, vec, wanted[uid].get("meta", {}), wanted[uid].get("text", "")
                )
            self._save()
            logger.info(
                f"[vector_index] Indexed {len(missing)} new, pruned {len(removed)}, "
                f"text-filled {filled} ({self._store_path}, total {len(self._ids)})"
            )
        except Exception as e:
            logger.warning(f"[vector_index] ensure_indexed failed ({self._store_path}): {e}")

    async def add(self, uid: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Embed and append a single item (e.g. a freshly written diary)."""
        if not uid or not text or uid in self._meta:
            return
        try:
            vec = (await self._embed([text]))[0]
            self._append(uid, vec, meta or {}, text)
            self._save()
        except Exception as e:
            logger.warning(f"[vector_index] add({uid}) failed: {e}")

    async def add_many(self, items: List[Dict[str, Any]]) -> None:
        """Embed and append several items at once (e.g. all chunks of one new
        diary), in a single batched embedding call. Skips ids already present."""
        new = [
            it for it in items
            if it.get("id") and it.get("text") and it["id"] not in self._meta
        ]
        if not new:
            return
        try:
            vecs = await self._embed([it["text"] for it in new])
            for it, vec in zip(new, vecs):
                self._append(it["id"], vec, it.get("meta", {}), it.get("text", ""))
            self._save()
        except Exception as e:
            logger.warning(f"[vector_index] add_many failed ({self._store_path}): {e}")

    async def retrieve(
        self,
        query: str,
        *,
        exclude_ids: Optional[set] = None,
        similarity_threshold: float = 0.55,
        topn_threshold: float = 0.70,
        max_retrievals: int = 2,
        debug_k: int = 5,
        group_by: Optional[str] = None,
        lexical_weight: float = 0.5,
        keywords: Optional[List[str]] = None,
    ) -> tuple:
        """Return ``(picked, candidates)`` for ``query`` using hybrid scoring.

        Score per entry = ``cosine + lexical_weight · lexical``, where lexical is
        the query↔text character-bigram overlap (0..1). Dense embeddings barely
        separate even exact keyword matches on a homogeneous diary corpus; the
        lexical term rescues them (a chunk literally containing the query words
        is pushed to the top). Set ``lexical_weight=0`` for pure vector.

        When ``group_by`` is set (e.g. ``"parent"``), entries are collapsed by
        that meta field, each group scored by its best member — so a diary
        chunked into sentences is ranked/returned once at the diary level, by its
        strongest-matching sentence. ``id`` is then the group value (parent uid),
        and ``exclude_ids`` is matched against it.

        Selection (topN logic): rank non-excluded entries/groups by hybrid score.
        If the best is below ``similarity_threshold`` pick nothing. Otherwise take
        the top one, then keep taking further entries while each clears the
        stricter ``topn_threshold``, up to ``max_retrievals`` total.

        ``picked`` is ``[{"id", "score", "vector", "lexical", "meta"}, ...]``.
        ``candidates`` is the top ``debug_k`` ``(id, date, hybrid, vector, lexical)``
        BEFORE thresholding — for tuning from the logs. Returns ``([], [])`` on
        embedding failure.
        """
        if not query or not query.strip() or not self._ids:
            return [], []
        exclude_ids = exclude_ids or set()
        try:
            qvec = (await self._embed([query]))[0]
        except Exception as e:
            logger.warning(f"[vector_index] query embed failed ({self._store_path}): {e}")
            return [], []

        qnorm = float(np.linalg.norm(qvec)) or 1.0
        scores = self._normed @ (qvec / qnorm)

        # Lexical signal. With extracted keywords, score by the best-matching
        # keyword — but weight each keyword by how RARE it is in this corpus
        # (IDF), so a precise hit on a distinctive term wins while a hit on a
        # ubiquitous word (which most diaries share) barely counts. Without the
        # IDF weight, MAX-over-keywords saturated to 1.0 for almost any diary
        # that happened to share one common word (esp. on long queries that
        # extract many keywords), which flattened the lexical signal. Falls back
        # to whole-query bigram overlap when no keywords were given.
        use_kw = bool(lexical_weight and keywords)
        # Per-entry bigram sets, computed once and reused for both the IDF
        # document-frequency counts and the per-entry lexical score.
        text_bigrams: Dict[str, set] = {}
        kw_weighted: List[tuple] = []  # [(keyword_bigrams, idf_weight), ...]
        if use_kw:
            for uid in self._ids:
                text_bigrams[uid] = _char_bigrams(self._texts.get(uid, ""))
            kw_weighted = self._keyword_idf_weights(keywords, text_bigrams)
        qbg = _char_bigrams(query) if (lexical_weight and not use_kw) else set()

        # Best HYBRID score per result key. Tuple: (hybrid, vector, lexical, meta).
        best: Dict[str, tuple] = {}
        for i, uid in enumerate(self._ids):
            meta = self._meta.get(uid, {})
            key = meta.get(group_by, uid) if group_by else uid
            if key in exclude_ids:
                continue
            v = float(scores[i])
            if not lexical_weight:
                lx = 0.0
            elif use_kw:
                tb = text_bigrams[uid]
                lx = max(
                    (w * (len(kb & tb) / len(kb)) for kb, w in kw_weighted),
                    default=0.0,
                )
            else:
                lx = _lexical_overlap(qbg, self._texts.get(uid, ""))
            h = v + lexical_weight * lx
            cur = best.get(key)
            if cur is None or h > cur[0]:
                best[key] = (h, v, lx, meta)

        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        candidates = [
            (key, meta.get("date", ""), round(h, 3), round(v, 3), round(lx, 2))
            for key, (h, v, lx, meta) in ranked[:debug_k]
        ]
        if not ranked or ranked[0][1][0] < similarity_threshold:
            return [], candidates

        picked = [ranked[0]]
        for key, tup in ranked[1:]:
            if len(picked) >= max_retrievals or tup[0] < topn_threshold:
                break
            picked.append((key, tup))

        hits = [
            {"id": key, "score": h, "vector": v, "lexical": lx, "meta": meta}
            for key, (h, v, lx, meta) in picked
        ]
        return hits, candidates

    @staticmethod
    def _keyword_idf_weights(
        keywords: List[str], text_bigrams: Dict[str, set]
    ) -> List[tuple]:
        """Return ``[(keyword_bigrams, weight), ...]`` for the query keywords.

        ``weight`` ∈ [0, 1] is the keyword's inverse document frequency,
        normalised so the rarest keyword in this query weighs 1. A keyword that
        appears in (almost) every entry weighs ~0, so a lexical hit on a common
        word can no longer saturate the score — only hits on distinctive terms
        carry weight. A keyword "appears" in an entry when all its character
        bigrams are present (== a full match, the same bar the score's 1.0 uses).

        Returns ``[]`` when nothing discriminates (no valid keyword, or every
        keyword is ubiquitous), so the caller's ``max(..., default=0.0)`` yields
        a zero lexical signal and ranking falls back to pure cosine.
        """
        n = len(text_bigrams) or 1
        entries: List[tuple] = []  # (keyword_bigrams, idf)
        for k in keywords:
            kb = _char_bigrams(k)
            if not kb:
                continue
            # Soft document frequency: sum the SAME bigram-overlap fraction the
            # score uses, rather than counting full matches. A base form like
            # 食べる only partially overlaps the conjugated 食べた in the text, so
            # a hard "all bigrams present" test would score df=0 and mistake a
            # ubiquitous word for a rare one. Summing fractions instead means a
            # term that lightly matches the whole corpus accrues a high df → a
            # low weight, which is exactly what we want.
            df = sum(len(kb & tb) / len(kb) for tb in text_bigrams.values())
            idf = max(math.log(n / (1 + df)), 0.0)
            entries.append((kb, idf))
        if not entries:
            return []
        max_idf = max(idf for _, idf in entries)
        if max_idf <= 0:
            return []  # every keyword is ubiquitous → no discriminating signal
        return [(kb, idf / max_idf) for kb, idf in entries]

    # ------------------------------------------------------------------
    # Internal mutation helpers (keep _ids / _vectors / _normed in lockstep)
    # ------------------------------------------------------------------

    def _append(
        self, uid: str, vec: np.ndarray, meta: Dict[str, Any], text: str = ""
    ) -> None:
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if self._vectors.size == 0:
            self._vectors = vec
        else:
            self._vectors = np.vstack([self._vectors, vec])
        self._ids.append(uid)
        self._meta[uid] = meta
        self._texts[uid] = text
        n = float(np.linalg.norm(vec)) or 1.0
        normed = vec / n
        self._normed = normed if self._normed.size == 0 else np.vstack([self._normed, normed])

    def _drop(self, ids: set) -> None:
        keep = [i for i, uid in enumerate(self._ids) if uid not in ids]
        self._ids = [self._ids[i] for i in keep]
        for uid in ids:
            self._meta.pop(uid, None)
            self._texts.pop(uid, None)
        self._vectors = self._vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        self._normed = self._normed[keep] if keep else np.zeros((0, 0), dtype=np.float32)
