docker run -it --rm \
-p 50051:50051 \
-e CONFIG_PATH=examples/qwen-asr-onnx/config.yaml \
-e MODEL_PATH=model-bin/Daumee/Qwen3-ASR-0.6B-ONNX-CPU \
-v ${PWD}/model-bin:/app/model-bin \
registry.cn-hangzhou.aliyuncs.com/migo-dl/qwen-asr-onnx:0.2.0a1-amd64

# 213
docker run -it --rm \
--name wcw_qwen-asr-onnx \
-p 50015:50051 \
-e CONFIG_PATH=examples/qwen-asr-onnx/config.yaml \
-e MODEL_PATH=/model-bin/Qwen3-ASR-0.6B-ONNX-CPU \
-v ${PWD}:/model-bin \
registry.cn-hangzhou.aliyuncs.com/migo-dl/qwen-asr-onnx:0.2.0a1-amd64