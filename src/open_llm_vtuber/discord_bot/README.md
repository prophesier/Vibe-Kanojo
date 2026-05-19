# Discord Bridge for Open-LLM-VTuber

A minimal Discord bot that forwards chat messages in allowed channels to the
Open-LLM-VTuber (OLV) backend over WebSocket and posts the AI's text replies
back to the channel.

> **Scope:** text bridge only. Voice channel support (TTS push + voice
> receive → ASR) is **not** implemented in this version. See
> [Roadmap: voice channel](#roadmap-voice-channel) at the bottom.

## How it works

```
Discord channel  ──text──►  DiscordVTuberBot
                                 │
                                 ▼
                            OLVBridge  ──ws──►  OLV /proxy-ws  ──►  LLM + TTS
                                 │                       │
                                 ◄───── display_text ────┘
                                 │
                                 ▼
Discord channel  ◄──text──  on_reply()
```

- The bridge connects to `ws://<host>:<port>/proxy-ws` (the same endpoint
  the BiliBili integration uses) and sends each user message as a
  `text-input` frame.
- The OLV backend streams its reply as `audio` messages whose
  `display_text.text` field carries the spoken sentence. The bridge
  collects those, and on `control: conversation-chain-end` it posts the
  joined text back to the originating Discord channel.
- Turns are serialised — only one Discord message is in flight against OLV
  at a time, so replies cannot interleave. Audio bytes are ignored on the
  Discord side.

## 1. Register a Discord bot and get a token

1. Open <https://discord.com/developers/applications> and click **New
   Application**. Give it a name.
2. In the left sidebar pick **Bot**. Optionally set a username and avatar.
3. Under **Privileged Gateway Intents**, enable **Message Content
   Intent**. The bridge needs this to read message text.
4. Click **Reset Token** → **Copy**. This token is the bot's password —
   store it somewhere private; you cannot view it again.
5. In the left sidebar pick **OAuth2 → URL Generator**:
   - Scopes: `bot`
   - Bot permissions: `Send Messages`, `Read Message History`, `View
     Channels` (for voice later: `Connect`, `Speak`, `Use Voice
     Activity`)
6. Open the generated URL in a browser and invite the bot to your server.

## 2. Find the IDs you need

Enable **Developer Mode** in Discord (`Settings → Advanced → Developer
Mode`), then right-click on:

- a server → **Copy Server ID** → use as `guild_ids` entry,
- a channel → **Copy Channel ID** → use as `channel_ids` entry.

Both whitelists are optional — leave them empty to allow every guild /
channel the bot can see.

## 3. Install dependencies

The Discord extras pull in `discord.py`. The bridge itself uses
`websockets` and `loguru`, which OLV already depends on.

```bash
uv sync --extra discord
```

If you don't use `uv`:

```bash
pip install "discord.py>=2.3.2"
```

## 4. Configure `conf.yaml`

Open your `conf.yaml` and add a `discord_config:` section at the bottom
(copy it from `config_templates/conf.default.yaml` if you started from a
template that pre-dated this feature):

```yaml
discord_config:
  # Either set the token here, or leave it empty and export DISCORD_BOT_TOKEN.
  token: ""
  guild_ids: [123456789012345678]
  channel_ids: [987654321098765432]
  respond_to_mentions_only: false
  command_prefix: ""
  olv_ws_url: ""  # empty -> ws://<system_config.host>:<system_config.port>/proxy-ws
```

Recommended: keep the token out of the file and provide it via env var:

```bash
export DISCORD_BOT_TOKEN="..."
```

### Options

| key | type | default | meaning |
| --- | --- | --- | --- |
| `token` | str | `""` | Bot token. Falls back to `$DISCORD_BOT_TOKEN`. |
| `guild_ids` | list[int] | `[]` | Allowed guild IDs. Empty = all. |
| `channel_ids` | list[int] | `[]` | Allowed channel IDs. Empty = all. |
| `respond_to_mentions_only` | bool | `false` | If true, only reply when @-mentioned. |
| `command_prefix` | str | `""` | If set, only forward messages starting with this prefix (stripped before sending). |
| `olv_ws_url` | str | `""` | Override the OLV WS URL. Empty = derive from `system_config`. |

## 5. Run it

In one terminal, start OLV as usual:

```bash
uv run run_server.py
```

In a second terminal, start the bridge:

```bash
uv run python -m src.open_llm_vtuber.discord_bot
```

Or via the script wrapper (same thing):

```bash
uv run python scripts/run_discord_bot.py
```

You should see log lines like:

```
INFO  Connecting to OLV at ws://localhost:12393/proxy-ws
INFO  OLV bridge connected
INFO  Discord bot ready as YourBot#1234 (id=...)
```

Type something in an allowed channel and the bot will reply.

## Troubleshooting

- **"discord_config section missing"** — copy the block from
  `config_templates/conf.default.yaml` into your `conf.yaml`.
- **Bot ignores messages** — make sure (1) the `Message Content Intent`
  is enabled in the Developer Portal, (2) the bot has `View Channel` +
  `Send Messages` permission in the channel, (3) the guild/channel IDs
  are in your whitelist (or the whitelist is empty).
- **"bridge not connected"** in replies — OLV isn't running, isn't
  listening on the configured host/port, or `olv_ws_url` is wrong.
- **"timed out waiting for reply"** — the LLM took longer than 120s; the
  default timeout is in `OLVBridge(..., turn_timeout=...)`.
- **Reply text looks duplicated or empty** — the bridge reads
  `display_text` from OLV's `audio` frames. If the active TTS config
  produces only one big chunk per turn, the text comes in one piece; if
  it produces per-sentence chunks they are joined with spaces.

## Roadmap: voice channel

The minimum viable bridge ends here on purpose. A full voice bridge needs
a noticeably bigger surface:

- **TTS → voice channel.** `discord.py[voice]` requires `libopus` on the
  host. OLV's `audio` frames are base64 WAVs at the TTS engine's native
  sample rate (e.g. 32 kHz / 44.1 kHz); Discord wants 48 kHz stereo
  Opus. We'd need an `AudioSource` that decodes WAV → resamples →
  re-encodes to Opus and pipes into `VoiceClient.play`. The base64-WAV
  payload is already in `bridge._handle_incoming` (`type == "audio"`);
  hooking a sink there is the natural extension point.
- **Voice → ASR.** Discord only exposes received voice through the
  `voice_recv` extension (e.g. `discord-ext-voice-recv`) which yields
  48 kHz stereo PCM per speaker. OLV's ASR path expects float32 mono at
  16 kHz, streamed as `mic-audio-data` chunks with a final
  `mic-audio-end`. That means: per-speaker buffering, VAD-driven
  segmentation (use OLV's `Silero VAD`, not Discord's silence
  detection), resample to 16 kHz mono, normalise to float32, send.
- **Mixing semantics.** Decide who the AI "hears" — everyone? push-to-talk
  via a slash command? a single designated speaker? The current channel
  whitelist isn't enough to express that; you'd want a separate
  `voice_channel_ids` list and probably a `speakers: []` allowlist.

If you want to take this on, the bridge layer already exposes the right
seams: extend `OLVBridge` with `send_audio_chunk(...)` /
`send_audio_end()` methods (just wrap `mic-audio-data` /
`mic-audio-end`), and add an `audio` branch in `_handle_incoming` that
forwards the base64 WAV to a registered voice sink.
