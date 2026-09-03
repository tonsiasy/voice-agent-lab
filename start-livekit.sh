#!/bin/bash
# 每次启动动态探测当前局域网 IP,写入 node_ip。
#
# 为什么不用 127.0.0.1:回环地址与浏览器的真实网卡地址(en0)属于不同接口,
# 这条"跨接口"路径在某些网络环境下(VPN/企业级网络扩展)会被拦截,
# 表现为 ICE 候选配对后 STUN 请求发出但零回应(state: failed)。
# 让服务器与浏览器用同一个真实网卡地址,是真正意义上的本机同网段流量,
# 不会触发这类拦截逻辑。
#
# 为什么不硬编码某个 IP(Day 1 的教训):换网络后 IP 会变,硬编码值随时失效。
# 这里每次启动都重新探测,自动适配。
set -euo pipefail
cd "$(dirname "$0")"

IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")
echo "[start-livekit] 探测到当前局域网 IP: $IP"

cat > livekit-dev.yaml <<EOF
port: 7880
rtc:
  tcp_port: 7881
  port_range_start: 7882
  port_range_end: 7892
  use_external_ip: false
  node_ip: $IP
keys:
  devkey: secret
EOF

pkill -f "livekit-server --config livekit-dev.yaml" 2>/dev/null || true
sleep 1
exec livekit-server --config livekit-dev.yaml --bind 0.0.0.0
