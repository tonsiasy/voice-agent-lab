"""字错率(CER)计算:编辑距离 ÷ 参考字数。中英混合按字符切。"""
import re, sys

def norm(s: str) -> str:
    """去标点空格,英文转大写(ASR 大小写不敏感)。"""
    s = re.sub(r"[，。、？！,.?!\s—－-]", "", s)
    return s.upper()

def edit_distance(a: str, b: str) -> tuple[int, int, int]:
    m, n = len(a), len(b)
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): d[i][0] = i
    for j in range(n+1): d[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(a[i-1]!=b[j-1]))
    return d[m][n], m, n

REF = ("你好我是王通今天我们来测试一下流式语音识别的性能表现"
       "这段音频包含中文和英文混合的内容比如LiveKitWebRTC和ASR这些技术名词"
       "希望识别的准确率和延迟都能够满足实时对话的要求")

HYPS = {
 "whisper small (非流式)":
   "你好,我是王通。今天我们来测试一下流逝语音识别的性能表现。这段音频包含中文和英文混合的内容,"
   "比如LiveKit, WebRTC,和ASR这些技术名词。希望识别的准确率和延迟都能够满足实时对话的要求。",
 "streaming zipformer":
   "你好我是王通今天我们来测试一下刘氏语音识别的性能表现这段音频包含中文和英文混合的内容"
   "比如 KIT WEBRTC和 ASR这些技术名词希望识别的准确率和延迟都能够满足实时对话的要求",
}

r = norm(REF)
print(f"参考文本 {len(r)} 字\n")
for name, hyp in HYPS.items():
    dist, _, hn = edit_distance(r, norm(hyp))
    print(f"{name:24s} CER = {dist/len(r)*100:5.1f}%  (编辑距离 {dist}, 输出 {hn} 字)")
