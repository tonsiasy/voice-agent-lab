# voice-agent-lab

基于 **LiveKit Agents** 的实时语音对话 Agent 最小实现：Agent 作为参与者接入
WebRTC 房间，完成「听 → 想 → 说」闭环，全链路自建适配器、无付费语音 API。

配套工程笔记（含实测延迟数据与踩坑记录）：
`tonybrain/knowledge/ai-engineering/2026-09-01-WebRTC-SFU-MCU-语音Agent补盲.md` §九。

## 管道

```
麦克风 → WebRTC → silero VAD（本地断句）
       → faster-whisper small（本地 STT，自定义 stt.STT 适配器，非流式）
       → DeepSeek（openai 兼容接口，改 base_url）
       → edge-tts（免 key TTS，自定义 tts.TTS 适配器，mp3 分片交框架解码）
       → WebRTC → 扬声器
```

Selective Forwarding Unit（SFU，选择性转发单元）由 `livekit-server` 提供；
Agent 以 `dev` 模式自动派驻新房间，不依赖启动顺序。

## 实现要点

- **自定义 STT 适配器**（`FasterWhisperSTT`）：继承 `stt.STT` 声明
  `streaming=False`，`AgentSession` 会用 VAD 自动切段后逐段调用；
  转写在 `asyncio.to_thread` 里跑，避免阻塞事件循环。
- **自定义 TTS 适配器**（`EdgeTTS`）：继承 `tts.TTS` + `tts.ChunkedStream`，
  把 edge-tts 的 mp3 分片推给 `AudioEmitter`，由框架用 `av` 解码重采样。
- **口语化 persona**：输出会被直接转成语音，故约束"不超过两句、无列表/代码/表情"。

## 实测数据（Intel CPU / whisper small int8，2026-09-02）

| 指标 | 实测 |
|---|---|
| 转写耗时 ÷ 音频时长 | **1.97x（比实时慢一倍）** |
| 用户说完 → 回复文本 | 中位 16.0s，最快 8.0s |

**结论**：对话式语音 Agent 要求 STT 显著快于实时（流式场景通常 <0.3x），
本地小模型在无 GPU 的 x86 上达不到——这是**选型问题不是调参问题**，
生产需走流式云端 ASR 或 GPU 部署。中文识别质量同样是瓶颈
（大量误识、偶发繁体输出），**管道通 ≠ 体验可用**。

## 运行

```bash
# 1. 依赖（Intel macOS 需 onnxruntime==1.18.1，新版无 x86 wheel）
uv sync

# 2. 配置
cp .env.example .env && vim .env      # 填 DEEPSEEK_API_KEY

# 3. 模型
uv run python agent.py download-files  # silero VAD
#    whisper 模型：从 hf-mirror 直接下载到 models/faster-whisper-small/
#    （HF 新版 Xet 协议与镜像不兼容，建议 curl 直取 model.bin/config.json/
#      tokenizer.json/vocabulary.txt；注意 404 响应体会被存成正常文件）

# 4. 启动三件套
livekit-server --config livekit-dev.yaml --bind 0.0.0.0 &
uv run python agent.py dev &
cd web && python3 -m http.server 8088 &

# 5. 浏览器打开 http://localhost:8088/ ，选身份加入
```

## 踩坑备忘

- **ICE 通告地址必须正确**：`livekit-server` 启动时探测的 IP 若与机器实际 IP
  不符（例如换过网络），所有连接 0 秒断连（`joinDuration: "0s"`）且无明显报错。
  本机实验用 `node_ip: 127.0.0.1` 钉死（见 `livekit-dev.yaml`）。
- **https 页面连 `ws://localhost` 会被混合内容策略拦截**——用本地 http 页，
  生产用 wss。失败形态很迷惑：摄像头预览照常显示，看着像已入会。
- **无头浏览器跑不了 WebRTC**（`could not establish pc connection`），
  自动化验证需 fake-device 参数或真实浏览器。
- **判断"连上没有"问服务器**，别看页面：
  `POST /twirp/livekit.RoomService/ListParticipants`。

## 目录

```
agent.py            Agent 主体（VAD/STT/LLM/TTS 装配 + 两个自定义适配器）
web/index.html      入会页（身份下拉、SDK 就绪门、错误上屏）
livekit-dev.yaml    本机 SFU 配置（node_ip 钉死回环）
.env.example        环境变量样例
models/             whisper 模型（gitignore，约 467MB）
```
