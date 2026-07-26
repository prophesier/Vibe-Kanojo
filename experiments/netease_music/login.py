"""One-time NetEase Cloud Music login, by QR code.

    uv run python experiments/netease_music/login.py

Writes a QR image next to this file and opens it; scan it with the NetEase
Cloud Music app on your phone and confirm. The resulting ``MUSIC_U`` cookie is
saved to ``ncm_session.json`` (gitignored, chmod 600) and lasts months —
this is not something you should need to repeat often.

Being logged in buys higher bitrates and access to your own playlists and
anything your account is entitled to; search and many songs work anonymously.

The QR image is rendered by ``segno`` run through ``uvx``, so it stays out of
the project's dependency list — it is needed once per login, not at runtime.
"""

from __future__ import annotations

import asyncio
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ncm_client import NeteaseClient, NeteaseUnavailable  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
QR_FILE = HERE / "login_qr.png"

_POLL_SECONDS = 2
_TIMEOUT_SECONDS = 300
_MESSAGES = {
    801: "スマホでスキャン待ち…",
    802: "スキャン済み。アプリで確認して…",
}


def _write_qr(url: str) -> bool:
    """Render the login URL as a QR image. False if no renderer is available."""
    for args in (
        ["uvx", "--from", "segno", "segno", "-o", str(QR_FILE), "--scale=8", url],
        ["uvx", "--from", "qrcode", "qr", url],
    ):
        try:
            result = subprocess.run(args, capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        if args[1] == "--from" and args[2] == "qrcode":
            QR_FILE.write_bytes(result.stdout)
        if QR_FILE.exists() and QR_FILE.stat().st_size > 0:
            return True
    return False


def _open(path: pathlib.Path) -> None:
    try:
        if sys.platform == "win32":
            subprocess.Popen(["cmd", "/c", "start", "", str(path)], shell=False)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass


async def main() -> int:
    client = NeteaseClient()
    try:
        existing = await client.account()
        if existing:
            print(f"すでにログイン済み: {existing}")
            print("入れ直す場合は ncm_session.json を消してから再実行。")
            return 0

        unikey = await client.qr_key()
        url = NeteaseClient.qr_login_url(unikey)
        if _write_qr(url):
            print(f"QRコード: {QR_FILE}")
            _open(QR_FILE)
        else:
            print("QR画像を作れませんでした。次のURLを自分でQR化してスキャンして:")
            print(f"  {url}")
        print("網易雲音樂アプリでスキャンして確認してください。\n")

        last = None
        for _ in range(_TIMEOUT_SECONDS // _POLL_SECONDS):
            await asyncio.sleep(_POLL_SECONDS)
            result = await client.qr_check(unikey)
            code = result.get("code")
            if code == 803:
                who = await client.account()
                print(f"\nログイン成功: {who or '(nickname unavailable)'}")
                print(f"保存先: {client._session_file}")
                QR_FILE.unlink(missing_ok=True)
                return 0
            if code == 800:
                print("\nQRコードの有効期限切れ。もう一度実行してください。")
                return 1
            if code != last:
                print(_MESSAGES.get(code, f"code={code}"))
                last = code
        print("\nタイムアウトしました。もう一度実行してください。")
        return 1
    except NeteaseUnavailable as e:
        print(f"失敗: {e}")
        return 1
    finally:
        await client.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
