# Qwen3-ASR AX650 专用 Native Runner

这是 `qwen_asr_onnx` Python 包内的专用子包，只服务 Qwen3-ASR-0.6B 的 AX650/MC50、C64/P448/CTX2047、BF16 activation 模型。它不是通用 LLM runtime，也不提供聊天、动态 batch、采样策略或 Python 逐层调度。

## 调用关系

```text
qwen_asr_onnx.ax_m4c.AxQwenAsr
    -> CFFI
        -> libax_qwen_asr.so（稳定 C ABI）
            -> 专用 Qwen3-ASR adapter
                -> AXERA ax-llm 的 AX650 model runner
                    -> AXEngine / 设备侧 KV cache
```

导入 `qwen_asr_onnx.ax_m4c` 不会加载动态库或 AXEngine。第一次构造 `AxQwenAsr` 时，Python 才从子包的 `lib/libax_qwen_asr.so` 加载 native 实现。模型路径始终由调用者在运行时传入。

## Python 用法

```python
from pathlib import Path

from qwen_asr_onnx.ax_m4c import AxQwenAsr

model_dir = Path("/path/to/Qwen3-ASR-0.6B-AX650-C64-P448-CTX2047")
pcm16_bytes = Path("audio.pcm").read_bytes()

with AxQwenAsr(model_dir) as asr:
    asr.warmup()
    text = asr.transcribe_pcm16(pcm16_bytes, sample_rate=16000)
    print(text)
```

输入必须是一维、连续、16 kHz、单声道 PCM16；其他采样率会明确报错，不会静默重采样。返回值是模型原始 UTF-8 文本，不删除 `language Chinese<asr_text>`，也不做 `.strip()` 或替换。

`AxQwenAsr` 持有唯一 native handle，支持 context manager，`close()` 幂等，并禁止 copy、deepcopy 和 pickle。当前进程只允许一个 live handle；同一 handle 的推理和关闭由 Python 与 C++ 两层串行化。

薄命令行示例只使用标准库读取 WAV：

```bash
poetry run python examples/ax-m4c/infer.py MODEL_DIR AUDIO.wav
```

## 构建

设备构建依赖 AX650 SDK 3.6.2 的 `msp/out` 目录。CMake 只编译本模型需要的官方 runner、memory/logger 和 tokenizer 最小源文件集合，并固定以下 revision：

- `AXERA-TECH/ax-llm`: `29427ac732044b423cd191d8bd37aff3ec772f21`
- `tokenizer.axera`: `ffa443bdd260e91c3eb7d8c6baeedb02f42bbbd1`

联网构建会把它们下载到 CMake build 目录；离线构建可显式传入已有 checkout，运行时不依赖临时源码目录。

```bash
BSP_MSP_DIR=/absolute/path/to/msp_3.6.2/out \
src/qwen_asr_onnx/ax_m4c/native/build_ax650.sh
```

可选环境变量：

- `AX_QWEN_ASR_BUILD_DIR`：构建目录；
- `AX_QWEN_ASR_BUILD_JOBS`：并行任务数，默认 4；
- `AX_QWEN_ASR_AXLLM_SOURCE_DIR`：固定 revision 的 ax-llm checkout；
- `AX_QWEN_ASR_TOKENIZER_SOURCE_DIR`：固定 revision 的 tokenizer checkout。

脚本把 runtime 产物安装到 `qwen_asr_onnx/ax_m4c/lib/`。本机生成的 `.so`、`.so.*` 已被 Git 忽略，不得提交；ARM64 wheel 后续需要在发行构建中把该目录作为平台 package data 纳入。

如果 AXEngine 库不在系统动态链接器默认路径中，运行前设置板卡实际库目录，例如：

```bash
export LD_LIBRARY_PATH=/soc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## 固定模型校验

`ax_qwen_asr_create()` 会 canonicalize 模型根目录并校验：

- `config.json`、`post_config.json` 的类型、必需字段和固定贪心配置；
- 28 层 decoder、词表 151936、hidden size 1024；
- `conv_frontend.axmodel`、`encoder.axmodel`、28 个 layer、post、tokenizer 和 embedding 都是模型根目录内的普通文件；
- 配置中的相对路径不能是绝对路径、不能包含 `..`，解析后的符号链接不能逃逸模型根目录；
- BF16 embedding 尺寸必须精确为 `151936 * 1024 * 2 = 311164928` 字节；
- conv、encoder、decoder 的 9 个 shape group、post 的 tensor 名称和字节数符合当前导出 profile。

模型加载一次并长期持有 AXEngine context、conv、encoder、28 层 decoder、post、tokenizer、只读 mmap embedding、设备侧 KV cache 和复用 buffer。请求间会清零 KV 内容但不会重建 session。

## 音频、prompt 与解码

native 预处理遵循 Qwen3-ASR 参考算法：16 kHz、400 点 FFT、hop 160、periodic Hann、`center=True`、reflect padding、128 个 Slaney mel bin、power spectrum、`log10`、8.0 动态范围裁剪和 `(x + 4) / 4` 归一化。

固定 400 点 DFT 使用预计算 Bluestein chirp 加 1024 点 radix-2 FFT。`rita.wav` 的 513 帧结果相对现有 librosa 实现最大绝对误差实测约 `1.57e-5`，单元测试容差为 `rtol=2e-4, atol=2e-5`。

三次下采样后的 audio token 会替换 prompt 中的 `AUDIO_PAD` embedding。prompt 使用固定的 system/user/assistant 和 Qwen3-ASR special token 布局。decoder 使用：

- batch size 1；
- 每块 64 token 的 prefill group；
- group 0 单 token decode；
- BF16 causal mask 和 position indices；
- 每层设备侧 KV cache；
- 固定 argmax；
- 最多生成 256 token，且不超过 CTX2047；
- `IM_END` 或 `ENDOFTEXT` 停止。

## C ABI

公开头文件是 `native/include/ax_qwen_asr.h`，ABI version 为 1。除 create/warmup/transcribe/last_error/destroy 外，还提供只读的 `ax_qwen_asr_get_last_metrics()`。

所有 C++ 异常都在 ABI 边界转换成错误码。输出由调用者分配；容量不足时返回 `AX_QWEN_ASR_BUFFER_TOO_SMALL`，并把包含结尾 NUL 的尺寸写入 `required_size`。C++ 不分配需要 Python 释放的内存。

`ax_qwen_asr_last_error()` 返回当前线程的错误快照，调用者不得释放；指针在同一线程下一次 `ax_qwen_asr_*` 调用前有效。`destroy(NULL)` 安全。裸 C 调用者不能让 destroy 与同一 handle 上尚未返回的调用并发执行。

## 测试

无 NPU 测试会在 `build/ax-m4c-host-tests` 编译 test-only core/fake ABI，不读取完整模型或加载 AXEngine：

```bash
poetry run python -m pytest -q tests/ax_m4c -m "not ax650"
```

真实板卡集成测试的路径通过环境变量传入：

```bash
AX_QWEN_ASR_MODEL_DIR=MODEL_DIR \
AX_QWEN_ASR_AUDIO=AUDIO.wav \
poetry run python -m pytest -q -s -m ax650 tests/ax_m4c/test_integration_ax650.py
```

启动完整模型前必须确认 `MemAvailable >= 2.5 GiB`，并确认没有残留 ASR 容器或推理进程。容器测试应限制内存，只挂载只读模型和源码，并一次只运行一个 ASR 容器/进程。

## AX650 实测报告

测试条件：

- 日期：2026-08-21；
- 板卡：AX650C / MC50，AXEngine/SDK `V3.6.2_20250603154858`；
- 基础镜像：`registry.cn-hangzhou.aliyuncs.com/migo-dl/axengine:0.1.3-m4c-aarch64`；
- 容器 Linux 内存上限：3 GiB；
- 模型：Qwen3-ASR-0.6B AX650 C64/P448/CTX2047 BF16；
- 音频：`rita.wav`，82079 samples，5.1299375 秒；
- 预热 1 次，热态 5 轮；
- 代码 revision：`55fa8f3d262b` 加本工作区当前未提交实现。

原始结果：

```text
language Chinese<asr_text>在同一个话题上纠缠不清的人，是不会讨女孩子喜欢的。
```

| 指标 | 实测值 |
|---|---:|
| 冷启动（构造并加载全部模型） | 2.013 s |
| warm-up（1 秒静音、完整链路、生成 1 token） | 0.213 s |
| 热态 wall time（5 轮平均） | 1.408 s |
| RTF（wall / 5.1299375） | 0.274 |
| 倍速（5.1299375 / wall） | 3.645× |
| native total（平均） | 1404.689 ms |
| 音频预处理（平均） | 163.143 ms |
| encoder（平均，含 prompt embedding/fuse） | 84.550 ms |
| decoder TTFT（平均） | 155.417 ms |
| decoder 总时间（平均） | 1156.979 ms |
| Linux VmHWM | 241020 KiB |
| Linux 峰值匿名页（20 ms 采样） | 52200 KiB |
| Linux 峰值文件页（20 ms 采样） | 175456 KiB |
| Linux 峰值采样总 RSS | 221016 KiB |
| CMM/NPU used 峰值 | 1741880 KiB |
| CMM/NPU used 基线与退出后 | 283176 KiB |

完整基准入口：

```bash
poetry run python examples/ax-m4c/benchmark.py MODEL_DIR AUDIO.wav \
    --warmups 1 --rounds 5
```

脚本输出 JSON，逐轮记录 wall、RTF、倍速、preprocess、encoder、TTFT、decoder total、native total、Linux RSS 分类和平台可读时的 CMM used。冷启动必须用没有既有 runner 的新进程；严格冷容器测试时先停止常驻容器。

## 已知边界与后续优化入口

- 当前只接受 16 kHz 单声道 PCM16，不重采样；
- 当前只支持单进程、单 handle、串行请求；
- 固定 256 token 贪心上限，不提供 Python logits/KV/tensor 接口；
- 基础镜像本身没有 CMake、SDK header 或 CFFI，需在构建环境准备，或制作一次性派生运行镜像；
- 平台 wheel 尚未配置自动携带 `.so`，源码构建脚本已把产物位置固定；
- 预处理仍在 CPU，后续可在不改变数值测试的前提下增加 NEON/vectorized mel accumulation；
- 本次实测基于未提交工作区，正式发布应在提交后重跑并替换 revision。

INT4 不引入通用模型注册表：新增一个内部 `QwenAsrInt4` 固定 profile，继续保持 BF16 activation/设备 KV 和相同 C ABI，只替换并校验 INT4 layer/post 资产及对应 tensor profile，然后复用同一正确性/基准矩阵。

CTX1024 同样作为内部固定 profile：把 context、decode mask、KV 容量和最大生成位置集中到编译期 profile，加载时根据 AXModel tensor 精确校验为 1024；prefill、prompt、mel 和 C ABI 不变。只有得到独立 CTX1024 模型资产后才启用该分支，不做运行时猜测。
