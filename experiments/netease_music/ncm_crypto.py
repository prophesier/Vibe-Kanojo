"""weapi request encryption for music.163.com.

The web player encrypts every ``/api/*`` call into ``/weapi/*`` with a scheme
its bundle calls ``asrsea``:

    secret    = 16 random base62 chars
    params    = b64(AES-CBC(b64(AES-CBC(json, PRESET_KEY, IV)), secret, IV))
    encSecKey = hex(pow(int(reverse(secret)), e, n))    # textbook RSA, no padding

The three constants below were extracted from NetEase's own bundle
(``core_*.js``) on 2026-07-27 rather than copied from a third-party mirror —
every public mirror of this algorithm has since been taken down, and a
constant nobody can re-derive is a constant nobody can fix. The bundle hides
them in an "emoji name -> hex fragment" lookup table (``nm.x.ek.emj``) plus a
fragment order (``md``); note there is also a decoy order named ``nmd`` that
mixes the preset-key fragments into the modulus. ``_extract_constants.py``
re-derives all three and diffs them against what is hardcoded here.

If NetEase ever rotates the key, the symptom is unmistakable: *every* call
starts failing at once (the server cannot recover the AES key, so it never
even sees our payload). Re-run the extractor and update the constants.
"""

from __future__ import annotations

import base64
import json
import secrets
from typing import Any, Dict

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_IV = b"0102030405060708"
_PRESET_KEY = b"0CoJUm6Qyw8W8jud"
_BASE62 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_PUB_EXPONENT = 0x010001
_PUB_MODULUS = int(
    "00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7b725"
    "152b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf695280104e0312"
    "ecbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932575cce10b424"
    "d813cfe4875d3e82047b97ddef52741d546b8e289dc6935b3ece0462db0a22b8e7",
    16,
)
# The modulus is 1024-bit, so encSecKey is always 256 hex chars.
_RSA_HEX_WIDTH = 256


def _aes_cbc_b64(data: bytes, key: bytes) -> bytes:
    """AES-128-CBC with PKCS#7 padding, base64-encoded (what CryptoJS emits)."""
    pad = 16 - len(data) % 16
    padded = data + bytes([pad]) * pad
    encryptor = Cipher(algorithms.AES(key), modes.CBC(_IV)).encryptor()
    return base64.b64encode(encryptor.update(padded) + encryptor.finalize())


def _rsa_no_padding(secret: str) -> str:
    """Textbook RSA over the reversed secret — no padding, so it is
    deterministic and the server can recover the AES key."""
    value = int.from_bytes(secret[::-1].encode(), "big")
    return f"{pow(value, _PUB_EXPONENT, _PUB_MODULUS):0{_RSA_HEX_WIDTH}x}"


def weapi_encrypt(payload: Dict[str, Any]) -> Dict[str, str]:
    """Encrypt a request payload into the ``params`` / ``encSecKey`` form
    posted to ``https://music.163.com/weapi/*``."""
    secret = "".join(secrets.choice(_BASE62) for _ in range(16))
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    params = _aes_cbc_b64(_aes_cbc_b64(text, _PRESET_KEY), secret.encode())
    return {"params": params.decode(), "encSecKey": _rsa_no_padding(secret)}
