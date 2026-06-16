<h1 align="center">Vibe-Kanojo</h1>

<p align="center">
<a href="./README.md">中文</a> | English | <a href="./README.JP.md">日本語</a>
</p>

<p align="center">
This project is modified and developed purely for personal use. Compatibility and availability of any feature are not guaranteed. For questions about the upstream itself, please head over to the original project instead.
</p>

> Built on top of [t41372/Open-LLM-VTuber](https://github.com/t41372/Open-LLM-VTuber).
> It keeps the upstream's voice chat, Live2D avatar, and ASR/TTS foundation, and adds
> a rewritten memory system plus a Discord bridge — turning a "one-shot voice toy" into
> a companion with **continuous memory you can reach from anywhere**.
>
> The character (prompt, Live2D model, voice) is fully swappable — the one bundled in
> this repo is just a sample. Replace it with whatever persona you want.

---

## ✨ What this project adds

### 🧠 Three-layer persistent memory

So the character actually *remembers* you instead of starting from scratch every chat.
Each layer has its own role:

- **Sliding window** — the verbatim text of the most recent full sessions goes straight
  into context for short-term continuity. A `【セッション開始: 日時】` boundary marker is
  injected between sessions so conversations across different days don't blur together.
- **facts.json** — **structured long-term facts** extracted from conversations (your
  preferences, habits, important people and events), injected into the system prompt in
  chronological order, each tagged with the date it was recorded. Supports automatic
  extraction, dedup, and "surgical" consolidation.
- **diaries/** — a **diary summary** generated at the end of each session, written with
  time-of-day words ("evening", "late night") rather than exact times, kept long-term as
  an index into older memory.

All memory injection uses **prompt caching** (Anthropic 1h cache / OpenAI auto cache),
holding a steady ~99% cache hit on normal turns — long memory doesn't mean high cost.

### ⏰ Time awareness

The character knows "what time it is now" and "when we last talked", and **never
fabricates times**:

- Each user message is tagged with `[YYYY-MM-DD HH:MM:SS Weekday]` (model-only — it never
  appears in replies)
- A strict rule in the system prompt: always check the tag before saying anything
  time-related
- "Now" is anchored to the timestamp of the latest user message

### 💬 Discord access

Reach the character from anywhere, **sharing the same session as the web client** (same
memory, same conversation):

- Text bridge to the OLV WebSocket backend, forwards image attachments
- Skips TTS for text-only chats to save resources

#### Admin slash commands

| Command | Purpose |
|---|---|
| `/restart` | Pull the latest code and restart the services remotely |
| `/logs target:bot\|olv\|both lines:N` | View logs remotely |
| `/status` | Process PID, uptime, current commit |
| `/facts-consolidate` | Trigger consolidation of long-term facts |

### 🔍 Web search

The character can look things up online during casual chat:

- **Claude path**: Anthropic's native `web_search` / `web_fetch` server tools
- **OpenAI path**: client-side implementation — search via Brave / Tavily (both have free
  tiers), fetch extracts the main content itself

When a search happens, an inline marker is shown at the trigger point so you know it
really went online instead of making things up.

---

## 🚀 Quick start

The base deployment (dependencies, ASR/TTS/LLM configuration) is the same as upstream —
see the [Open-LLM-VTuber docs](https://open-llm-vtuber.github.io/docs/quick-start) first.

In short:

```bash
uv sync                     # install dependencies
uv run run_server.py        # start the server (generates model_dict.json etc. on first run)
```

This project's extra config (persistent memory, Discord, web search) is documented in the
relevant blocks of `config_templates/conf.default.yaml`. See `discord_bot/` for how to
enable the Discord bot.

On first run, `model_dict.json`, `mcp_servers.json`, and `restart.bat` are generated
automatically from their `.example` templates. You can edit them directly — they're
gitignored, so your changes won't be synced to the repo.

### One-click launch (Windows)

The repo ships a Windows Terminal launcher template that brings up the OLV server, the
Discord bot, and GPT-SoVITS TTS in three tabs at once:

```bat
copy start_all.example.bat start_all.bat
```

Then edit `CONDA_ENV` (your conda env name) and `TTS_DIR` (your local GPT-SoVITS path) at
the top of `start_all.bat`. It's gitignored, so edit freely without polluting the repo.

`restart.bat` (used by Discord `/restart`) is generated on first run and can be edited
directly (also gitignored); set the conda env name and pull branch at the top.

---

## 📜 Third-party licenses

This project includes Live2D sample models provided by Live2D Inc. These assets are
licensed separately under the
[Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/)
and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html), and are
not covered by this project's MIT license. Commercial use (especially by medium/large
enterprises) may require additional permission from Live2D Inc.

The rest of the code inherits the MIT license of upstream
[t41372/Open-LLM-VTuber](https://github.com/t41372/Open-LLM-VTuber).
