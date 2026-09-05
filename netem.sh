#!/bin/bash
# macOS 弱网整形——Linux `tc netem` 的替代。
#
# 原理: dummynet = dnctl(定义"管道"的带宽/延迟/丢包) + pfctl(把匹配的流量塞进管道)。
# 只整形 SFU 媒体端口 UDP 7882-7899(7893+ 留给自检),信令(TCP 7880)不受影响,
# 这样能单独观察媒体面 QoS 机制,不会把 WebSocket 一起搞挂。
#
# 管道编号 = 方向 × pf 钩子:
#   pipe 1 / 3 = 下行 SFU → 客户端 (源端口在媒体范围)   out / in 钩子
#   pipe 2 / 4 = 上行 客户端 → SFU (目的端口在媒体范围) out / in 钩子
# 本机自连的包在 lo0 上会经过 pf 的 out 与 in 两个钩子——哪一个真正命中,
# 用 `diag` 看规则计数器;确认后可只保留一侧避免双重整形。
#
# 用法(需要 sudo):
#   sudo ./netem.sh down loss=5                 下行 5% 丢包
#   sudo ./netem.sh down bw=400                 下行限 400 kbit/s
#   sudo ./netem.sh down delay=200 loss=2       可组合
#   sudo ./netem.sh up bw=300                   上行限 300 kbit/s
#   sudo ./netem.sh clear down                  仅清某个方向(管道置为直通)
#   sudo ./netem.sh off                         全部清除、卸载规则
#   sudo ./netem.sh status | diag
#
# 每次变更把当前条件写进 /tmp/netem.marker,观测脚本会把它标进 CSV 每一行。
set -euo pipefail

ANCHOR="webrtc-netem"
PORTS="7882:7899"
MARKER="/tmp/netem.marker"
STATE="/tmp/netem.pf-was-enabled"
HOOKS="${NETEM_HOOKS:-out in}"   # 可用 NETEM_HOOKS=out 只挂一侧

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

require_root() {
  if [[ $EUID -ne 0 ]]; then echo "需要 sudo: sudo $0 $*" >&2; exit 1; fi
}

rules_text() {
  local hook
  for hook in $HOOKS; do
    local dp up
    case "$hook" in out) dp=1; up=2 ;; in) dp=3; up=4 ;; esac
    echo "dummynet $hook quick proto udp from any port $PORTS to any pipe $dp"
    echo "dummynet $hook quick proto udp from any to any port $PORTS pipe $up"
  done
}

ensure_rules() {
  if [[ ! -f "$STATE" ]]; then
    pfctl -s info 2>/dev/null | grep -q "Status: Enabled" && echo 1 > "$STATE" || echo 0 > "$STATE"
  fi
  # 把 anchor 挂进主规则集(仅内存,不改 /etc/pf.conf)
  if ! pfctl -sr 2>/dev/null | grep -q "anchor \"$ANCHOR\""; then
    (cat /etc/pf.conf; echo "dummynet-anchor \"$ANCHOR\""; echo "anchor \"$ANCHOR\"") | pfctl -q -f -
  fi
  rules_text | pfctl -a "$ANCHOR" -q -f -
  pfctl -s info 2>/dev/null | grep -q "Status: Enabled" || pfctl -q -e 2>/dev/null || pfctl -q -E 2>/dev/null || true
}

# 把 key=value 列表翻译成 dnctl 参数;缺省项显式归零,保证"重新设置"而非"叠加"
build_pipe_args() {
  local bw=0 delay=0 plr=0
  for kv in "$@"; do
    case "$kv" in
      loss=*)  plr=$(awk -v p="${kv#loss=}" 'BEGIN{printf "%.4f", p/100}') ;;
      delay=*) delay="${kv#delay=}" ;;
      bw=*)    bw="${kv#bw=}Kbit/s" ;;
      *) echo "未知参数: $kv (支持 loss= delay= bw=)" >&2; exit 1 ;;
    esac
  done
  echo "bw $bw delay $delay plr $plr"
}

pipes_of() { case "$1" in down) echo "1 3" ;; up) echo "2 4" ;; *) echo "方向须为 up|down" >&2; exit 1 ;; esac; }

set_marker() { echo "$*" > "$MARKER"; chmod 644 "$MARKER"; }

cmd_shape() {
  local dir="$1"; shift
  [[ $# -ge 1 ]] || usage
  ensure_rules
  local args; args=$(build_pipe_args "$@")
  local p
  # shellcheck disable=SC2086
  for p in $(pipes_of "$dir"); do dnctl pipe "$p" config $args; done
  set_marker "$dir $*"
  echo "[netem] $dir ← $* (pipes $(pipes_of "$dir"): $args)"
  cmd_status
}

cmd_clear() {
  local p
  for p in $(pipes_of "$1"); do dnctl pipe "$p" config bw 0 delay 0 plr 0; done
  set_marker "clear $1"
  echo "[netem] $1 直通"
}

cmd_off() {
  pfctl -a "$ANCHOR" -q -F all 2>/dev/null || true
  dnctl -q flush 2>/dev/null || true
  pfctl -q -f /etc/pf.conf 2>/dev/null || true
  if [[ -f "$STATE" ]] && [[ "$(cat "$STATE")" == "0" ]]; then pfctl -q -d 2>/dev/null || true; fi
  rm -f "$STATE"
  set_marker "baseline"
  echo "[netem] 已全部清除,pf 规则还原"
}

cmd_status() {
  echo "--- pipes ---"; dnctl list 2>/dev/null || echo "(无)"
  echo "--- rules ---"; pfctl -a "$ANCHOR" -sr 2>/dev/null || echo "(无)"
  echo "--- marker ---"; cat "$MARKER" 2>/dev/null || echo "(无)"
}

cmd_diag() {
  echo "=== pf 状态 ==="; pfctl -s info 2>&1 | head -3
  echo "=== 主规则集里的 anchor 行 ==="; pfctl -sr 2>&1 | grep -n "anchor" || echo "(无)"
  echo "=== dummynet 规则 ==="; pfctl -s dummynet 2>&1 || true
  echo "=== anchor 规则 + 计数器 (Evaluations/Packets) ==="; pfctl -a "$ANCHOR" -vsr 2>&1 || echo "(无)"
  echo "=== 管道 + 计数器 ==="; dnctl list 2>&1 || echo "(无)"
  echo "=== lo0 是否 skip ==="; pfctl -s Interfaces -v 2>/dev/null | grep -A2 "^lo0" || echo "(无接口信息)"
}

[[ $# -ge 1 ]] || usage
require_root "$@"
case "$1" in
  up|down) cmd_shape "$@" ;;
  clear)   [[ $# -eq 2 ]] || usage; cmd_clear "$2" ;;
  off)     cmd_off ;;
  status)  cmd_status ;;
  diag)    cmd_diag ;;
  *)       usage ;;
esac
