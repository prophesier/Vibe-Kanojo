<h1 align="center">Vibe-Kanojo</h1>

<p align="center">
中文 | <a href="./README.EN.md">English</a> | <a href="./README.JP.md">日本語</a>
</p>

<p align="center">
本项目单纯为个人自用修改开发，不能保证各种功能的兼容及可用性，对本体的任何疑问建议左转原项目询问。
</p>

> 本项目在 [t41372/Open-LLM-VTuber](https://github.com/t41372/Open-LLM-VTuber) 的基础上开发，
> 沿用了它的语音对话、Live2D 形象、ASR/TTS 等底层能力，并在此之上重写了记忆系统、
> 接入了 Discord，让它从"一次性语音玩具"变成一个**有连续记忆、可以多端随时找到**的伴侣。
>
> 角色（提示词、Live2D 模型、声音）都是可替换的——仓库里默认带的只是一个示例角色，
> 你可以换成任何你想要的设定。

---

## ✨ 本项目加了什么

### 🧠 三层持久化记忆

让角色真正"记得"你，而不是每次对话都从零开始。三层各司其职：

- **滑动窗口** —— 最近若干个完整 session 的原文直接进上下文，保证短期连续性。
  每个 session 之间会注入 `【セッション開始: 日時】` 边界标记，避免跨天对话被揉成一团。
- **facts.json** —— 从对话里抽取的**结构化长期事实**（你的偏好、习惯、重要的人和事），
  按时间排序注入系统提示，每条带记录日期。支持自动抽取、去重、以及"外科手术式"的合并。
- **diaries/** —— 每个 session 结束时生成一篇**日记总结**，用时段词（"傍晚""深夜"）而非精确时间，
  长期保留，作为更久远记忆的索引。

所有记忆注入都接入了 **prompt caching**（Anthropic 1h cache / OpenAI 自动 cache），
日常对话稳定 ~99% cache 命中，长记忆不等于高成本。

### ⏰ 时间感知

角色知道"现在几点""上次聊是什么时候"，并且**不会凭空编造时间**：

- 每条用户消息打上 `[YYYY-MM-DD HH:MM:SS 周几]` 时间标记（仅供模型读取，不会出现在回复里）
- 系统提示里有严格规则：任何涉及时间的发言前必须先查标记
- "现在时刻"以最新一条用户消息的时间为基准

### 💬 Discord 交互

人在外面也能找到角色聊天，**与 web 端共享同一个 session**（同一段记忆、同一个对话）：

- 文本桥接到 OLV 的 WebSocket 后端，支持转发图片附件
- 纯文本对话自动跳过 TTS，省资源

#### 管理员斜杠指令

| 指令 | 作用 |
|---|---|
| `/restart` | 远程拉取最新代码并重启服务 |
| `/logs target:bot\|olv\|both lines:N` | 远程查看日志 |
| `/status` | 查看进程 PID、运行时长、当前 commit |
| `/facts-consolidate` | 触发长期事实的合并整理 |

### 🔍 Web 检索

闲聊中角色可以主动联网查信息：

- **Claude 路径**：使用 Anthropic 原生 `web_search` / `web_fetch` server tool
- **OpenAI 路径**：客户端自实现，search 走 Brave / Tavily（均有免费额度），fetch 自己抽正文

检索发生时会在触发位置显示一个内联标记，让你知道它真的查了网，而不是瞎编。

---

## 🚀 快速开始

底层部署（依赖安装、ASR/TTS/LLM 配置）与上游一致，请先参考
[Open-LLM-VTuber 官方文档](https://open-llm-vtuber.github.io/docs/quick-start)。

简要：

```bash
uv sync                     # 安装依赖
uv run run_server.py        # 启动服务（首次会自动生成 model_dict.json 等配置）
```

本项目额外的配置项（持久化记忆、Discord、Web 检索）见 `config_templates/conf.default.yaml`
中对应区块的注释。Discord bot 的启用方式见 `discord_bot/` 下的说明。

首次启动时，`model_dict.json`、`mcp_servers.json`、`restart.bat` 会自动从同名的 `.example`
模板生成，你可以直接编辑它们（已被 git 忽略，改动不会同步到仓库）。

### 一键启动（Windows）

仓库自带一个 Windows Terminal 一键启动脚本模板，可同时拉起 OLV 服务、Discord bot、
GPT-SoVITS TTS 三个标签页：

```bat
copy start_all.example.bat start_all.bat
```

复制后编辑 `start_all.bat` 顶部的 `CONDA_ENV`（你的 conda 环境名）和 `TTS_DIR`
（本地 GPT-SoVITS 路径）即可。`start_all.bat` 已被 git 忽略，随便改不会污染仓库。

`restart.bat`（供 Discord `/restart` 远程重启用）会在首次启动时自动生成，直接编辑即可
（已被 git 忽略）；顶部可设置 conda 环境名与拉取分支。

---

## 📜 第三方许可

本项目包含 Live2D Inc. 提供的 Live2D 示例模型。这些素材依据
[Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/)
和[使用条款](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)单独授权，
不在本项目 MIT 许可的覆盖范围内。商用（尤其是中大型企业）可能需要向 Live2D Inc. 取得额外授权。

其余代码沿用上游 [t41372/Open-LLM-VTuber](https://github.com/t41372/Open-LLM-VTuber) 的 MIT 许可。
