# qwen-asr-onnx

Qwen ASR 的 ONNX 推理项目。仓库同时包含一个严格限定模型 profile 的 AX650/MC50 native 后端：

- 现有 ONNX 后端位于 `qwen_asr_onnx.inferencers`；
- AX650 专用后端位于 `qwen_asr_onnx.ax_m4c`，不会反向侵入 ONNX runner；
- AX650 后端的 Python/CFFI 边界只传 PCM16 和最终 UTF-8 文本，逐层调度、KV cache、mask 和 buffer 均留在 C++/设备侧。

AX650 的构建、使用、测试和实测报告见 [AX650 专用后端说明](src/qwen_asr_onnx/ax_m4c/README.md)。
