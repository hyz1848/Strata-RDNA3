# Gyro-L 部署指南（当前生产配置）

机器: 2x Radeon RX 7900 XTX (gfx1100), 89 GiB RAM, Linux
引擎: Strata 0.1.54 (自编译, HIP)
模型: Qwen3.8-Flash-Next Gyro-L abliterated, TQ3_0 (trellis, APR/prism.hadamard)
日期: 2026-10-31

---

## 1. 目录与文件

| 用途 | 路径 |
|---|---|
| 引擎源码 | `/opt/Strata-main` |
| 构建目录 (Ninja + HIP) | `/opt/Strata-main/build-hip` |
| 引擎二进制 | `/opt/Strata-main/engine/strata` |
| 视觉二进制 | `/opt/Strata-main/engine/strata-vision` |
| 生产配置 | `/opt/Strata-main/strata-gyro-l-abliterated.json` |
| 运行日志 | `/opt/Strata-main/strata-gyro-l.log` |
| 模型 GGUF (108.3 GB) | `/opt/models/gyro-l-abliterated/Qwen3.8-Flash-Next-Gyro-L-abliterated-TQ3_0.gguf` |
| mmproj | `/opt/models/gyro-l-abliterated/mmproj-F16.gguf` |
| MTP 草稿模型 | `/opt/Strata-data/mtp/rt` (源: `mtp-Qwen3.8-Flash-Next-draft.gguf`) |
| 专家 pack | `/opt/Strata-data/packs/gyro-l-abliterated` |
| IQ3_S 备选 pack | `/opt/Strata-data/packs/huihui-swift-iq3s` |
| IQ3_S 配置 | `/opt/Strata-main/strata-swift-iq3s-huihui.json` |

Python 环境: `/opt/Strata-main/.venv` (TheRock `rocm-sdk-core 10.2.0a20261009`)

---

## 2. 编译

```bash
cd /opt/Strata-main
cmake --build build-hip --target strata -j 8      # HIP 引擎
# 产物: build-hip/strata

# 安装到运行位置（先停服务, 否则 "文本文件忙"）
fuser -k 1234/tcp; pkill -x strata-host; sleep 4
cp build-hip/strata engine/strata
```

首次/干净配置（需要时）:

```bash
cd /opt/Strata-main
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_BACKEND=hip \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j 8
```

注意: `build-hip` 之外的文件（例如 `src/dsv41/ds4_tq.cu`）不一定在 `strata` 目标的
依赖里；改完代码若 `ninja: no work to do`，说明该文件不参与本引擎构建。

---

## 3. 模型准备

### 3.1 下载 GGUF

```bash
# hf-mirror 对大文件最稳; 并发多进程 curl 会互相破坏, 用单进程 -C - 续传
curl -L -C - -o /opt/models/gyro-l-abliterated/Qwen3.8-Flash-Next-Gyro-L-abliterated-TQ3_0.gguf \
  https://hf-mirror.com/<repo>/resolve/main/Qwen3.8-Flash-Next-Gyro-L-abliterated-TQ3_0.gguf
# 校验: 大小必须等于 108323559904 字节; 抽样 12 段 1MB 与服务器比对
```

### 3.2 建专家 pack

```bash
cd /opt/Strata-main
.venv/bin/python tools/iq_pack.py \
  --gguf /opt/models/gyro-l-abliterated/Qwen3.8-Flash-Next-Gyro-L-abliterated-TQ3_0.gguf \
  --out /opt/Strata-data/packs/gyro-l-abliterated \
  --compat-bf16
```

产物里重要的是:

- `native_experts.txt` 首行 `# strata native experts v5`（v5 = Hadamard 折叠专家）
- `hadamard.txt`: `block 128`, `signs 640 ...`, `signs 2560 ...`, `layer N 1 1 1 ...`
- `tokenizer/`

### 3.3 MTP 草稿模型

```bash
# 从 mtp-Qwen3.8-Flash-Next-draft.gguf 生成 /opt/Strata-data/mtp/rt (工具在 tools/)
```

---

## 4. 配置

`/opt/Strata-main/strata-gyro-l-abliterated.json` 关键字段:

```jsonc
{
  "exe": "/opt/Strata-main/engine/strata",
  "port": 1234,
  "host": "0.0.0.0",
  "model_name": "qwen3.8-27b",
  "api_key": "YOUR_API_KEY",
  "parallel": 4,
  "layer_split": "24",
  "tokenizer": "/opt/Strata-data/packs/gyro-l-abliterated/tokenizer",
  "log": "/opt/Strata-main/strata-gyro-l.log",
  "vram_reserve_mib": 2048,
  "user_args": ["--vram-reserve-mib", "3072"]
}
```

引擎参数 (`args`):

```
--pack /opt/Strata-data/packs/gyro-l-abliterated
--native /opt/models/.../Gyro-L-abliterated-TQ3_0.gguf
--ple-gguf /opt/models/.../Gyro-L-abliterated-TQ3_0.gguf
--expert-profile /opt/Strata-main/data/expert-profile.bin
--expert-cache auto
--prefill auto:16384
--spec 6 --spec-min-p 0.5 --mtp /opt/Strata-data/mtp/rt
--max-context 393216 --kv int8 --kv-resident 32768
--vision --trim-stage-weights
--rope-scaling yarn --rope-scale 1.5
--conversation-cache-mib 1024 --conversation-cache-slots 4
--lookup-chain 3
```

关键环境变量 (`env`):

| 变量 | 值 | 作用 |
|---|---|---|
| `STRATA_PF_FUSED` | 1 | 启用 fused prompt 专家内核 |
| `STRATA_PF_FUSED_TQ` | 1 | **让 trellis (TQ) 类型走 fused 内核**（本机新功能） |
| `STRATA_PF_FUSED_APR` | 1 | **fused 路径支持 APR/Hadamard 折叠专家**（本机新功能） |
| `STRATA_PF_GEMM` | 1 | prompt 走 hipBLASLt GEMM（38K 场景 +5%） |
| `STRATA_PF_GU_EXACT` | 1 | gate/up 精确模式 |
| `STRATA_PF_PIECE` | 8192 | prefill 分段 |
| `STRATA_SPEC_CUM` | 0.5 | 累积式 speculative |
| `STRATA_SPLIT_OWN` | 0 | 关闭 split-own（实测更慢） |
| `STRATA_STAGE_TRIM` | 1 | 裁剪 stage 权重 |
| `STRATA_ROUTE_TAIL_SKIP` | 7 | 跳过低排名专家（decode +10-20%） |
| `STRATA_IO_THREADS` | 64 | IO 线程数 |
| `STRATA_SPLIT_RING` | 384 | split ring 大小 |
| `STRATA_PCIE_KEEP` / `STRATA_PCIE_USAGE` | 96 / 1 | PCIe 保活 |
| `STRATA_COMBINE_HITS` / `STRATA_WAIT_FUSE` / `STRATA_WAIT_COMBINE` | 1 | combine 优化 |
| `STRATA_HIPBLASLT_TUNING` | `tools/hip/gfx1100-hipblaslt-100500.txt` | GEMM 调优表 |

APR/旋转相关的调试开关（新增，默认 `xh`）:

- `STRATA_PF_APR_DBG=xh` 两侧都旋转（正常）
- `STRATA_PF_APR_DBG=h` 只转 H（输入不转，仅调试）
- `STRATA_PF_APR_DBG=` 都不转（仅测速，输出错误）

---

## 5. 启动 / 停止

### 启动（生产, 端口 1234）

```bash
cd /opt/Strata-main
setsid nohup env ROCP_TOOL_ATTACH=1 .venv/bin/python serve/server.py \
  --engine strata \
  --config /opt/Strata-main/strata-gyro-l-abliterated.json \
  --port 1234 --host 0.0.0.0 \
  > /tmp/prod-serve.log 2>&1 < /dev/null &
```

`ROCP_TOOL_ATTACH=1` 只用于允许 rocprof attach，不影响性能。

启动约需 85-110 秒（108 GB mmap + pack + MTP + 视觉）。

### 健康检查

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  http://192.168.3.166:1234/v1/models

curl -s -H "Authorization: Bearer YOUR_API_KEY" \
  http://192.168.3.166:1234/v1/models | \
  python3 -c "import json,sys;d=json.load(sys.stdin);print(d['data'][0]['id'],d['data'][0]['status']['value'])"
```

### 停止

```bash
# 注意: 不要用 pkill -f "<含自身命令行子串的 pattern>" —— 会把自己这条命令也杀掉
fuser -k 1234/tcp 2>/dev/null; sleep 3
pkill -x strata-host; sleep 4
pkill -x strata-vision
```

引擎现在处理 SIGTERM: 会按 QUIT 流程退出, **保存 expert profile 与会话状态**。

---

## 6. 调用

OpenAI 兼容接口:

```bash
curl -s http://192.168.3.166:1234/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b",
       "messages":[{"role":"user","content":"你好"}],
       "max_tokens":128,"temperature":0.6}'
```

返回里的 `timings` 字段给出 prompt/decode 的 tok/s 与 draft 命中。

思考模型的 `reasoning_content` 会先输出；用 `reasoning_budget_tokens` 限制思考预算。

---

## 7. 本次修改的代码（相对上游 0.1.54）

| 文件 | 改动 | 备份 |
|---|---|---|
| `src/prefill/fused_tq.cuh` | 新增: trellis 子块解码 (状态机 + int8 码本) | - |
| `src/prefill/moe_fused_iq.cu` | TQ 类型支持、`unit()` 128 值块修正、TQ dispatch、APR 的 FP32 H 写出 + `apr_rot_quant_h_kernel` | `/opt/moe_fused_iq.cu.bak-1144` |
| `src/prefill/prefill.cpp` | `STRATA_PF_FUSED_APR` 门控、输入侧 Hadamard | - |
| `include/strata/prefill/moe_fused_iq.hpp` | `NativeGeom` 增加 `had_sh` / `had_block` | - |
| `src/program/generate.cpp` | SIGTERM/SIGINT → QUIT 优雅退出 | - |
| `apps/host/server/index.mjs` | CORS 白名单加 `192.168.3.166` | - |
| `apps/host/server/native-control.mjs` | host/port/api_key 双 key | - |
| `apps/host/server/inference.mjs` | `STRATA_SERVER_KEY` / authHeaders | - |

回滚到本次修改前的引擎:

```bash
cp /opt/Strata-main/engine/strata.pre-apr-0.1.54 /opt/Strata-main/engine/strata
# 并从配置中删掉 STRATA_PF_FUSED_TQ / STRATA_PF_FUSED_APR
```

---

## 8. 性能基线（本机实测）

| 场景 | 数值 |
|---|---|
| 8-10K prompt prefill | **1650-1920 tok/s** (fused TQ+APR) vs 1672-1687 (MMQ+PF_GEMM) |
| 38K prompt prefill | 2641-2678 (fused) vs **2755** (MMQ+PF_GEMM) |
| decode 单请求 | ~55-60 tok/s |
| decode 4 并发 | ~130 tok/s（用户口径） |
| 上下文 59 → 38561 | decode 54.5 → 52.8 tok/s（attention 占比 <5%） |

decode 已确认瓶颈是 trellis 解码的固有指令成本（不是 attention、不是 PCIe、不是批组）。

---

## 9. 质量评测

脚本（可复现, temperature=0）:

```bash
cd /opt
python3 quality_eval.py 1234 <tag>   # 基础 11 题
python3 eval2.py       1234 <tag>   # 多步推理/代码执行/指令 10 题
python3 eval3.py       1234 <tag>   # 28 题（算术答案程序生成）
```

结果文件: `/opt/bench-<tag>.json`, `/opt/eval2-<tag>.json`, `/opt/eval3-<tag>.json`

49 题累计:

| 模型 | 得分 |
|---|---|
| Gyro-L (TQ3) | 44/49 (89.8%) |
| IQ3_S (huihui) | 41/49 (83.7%) |

---

## 10. 常见问题

**启动后 48 秒卡顿 / swap 增长**
会话缓存过大。当前 `--conversation-cache-mib 1024 --conversation-cache-slots 4` 是修正后的值（曾用 8192 导致 swap 打满）。

**PCIe 掉到 2.5 GT/s**
冷启动恢复（断电重启）。用 `lspci -vv | grep -A2 "LnkSta"` 检查。

**`fused layout's MoE buffers are too small`**
该层没有走 fused 路径（例如 had_gu/had_d 但未开 `STRATA_PF_FUSED_APR`）。T=10552 这类大 chunk 会直接报错。

**输出 NaN / 乱码**
检查是否 `STRATA_PF_APR_DBG` 被设成只旋转一侧；正常应为 `xh`。

**切换回 IQ3_S**

```bash
fuser -k 1234/tcp; sleep 3; pkill -x strata-host; sleep 4; pkill -x strata-vision
cd /opt/Strata-main
setsid nohup .venv/bin/python serve/server.py --engine strata \
  --config /opt/Strata-main/strata-swift-iq3s-huihui.json --port 1234 --host 0.0.0.0 \
  > /tmp/iq3s-serve.log 2>&1 < /dev/null &
```

---

## 11. 管理界面与附加服务

| 服务 | 端口 | 启动方式 |
|---|---|---|
| Strata 管理 UI | 8791 | `systemctl --user start strata-host` (`~/.config/systemd/user/strata-host.service`) |
| OCR (PaddleOCR) | 1240 | `systemctl --user start paddle-ocr`（已开自启） |

管理 UI 通过 `STRATA_SERVER_URL=http://127.0.0.1:1234` 接管引擎。
