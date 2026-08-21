#!/usr/bin/env bash
set -euo pipefail

native_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "${native_dir}/.." && pwd)"
build_dir="${AX_QWEN_ASR_BUILD_DIR:-${native_dir}/build-ax650}"

cmake_args=(
    -S "${native_dir}"
    -B "${build_dir}"
    -DCMAKE_BUILD_TYPE=Release
    -DAX_QWEN_ASR_BUILD_DEVICE=ON
    -DAX_QWEN_ASR_BUILD_HOST_TESTS=OFF
    -DCMAKE_INSTALL_PREFIX="${package_dir}"
)

if [[ -n "${BSP_MSP_DIR:-}" ]]; then
    cmake_args+=("-DBSP_MSP_DIR=${BSP_MSP_DIR}")
fi
if [[ -n "${AX_QWEN_ASR_AXLLM_SOURCE_DIR:-}" ]]; then
    cmake_args+=("-DAX_QWEN_ASR_AXLLM_SOURCE_DIR=${AX_QWEN_ASR_AXLLM_SOURCE_DIR}")
fi
if [[ -n "${AX_QWEN_ASR_TOKENIZER_SOURCE_DIR:-}" ]]; then
    cmake_args+=("-DAX_QWEN_ASR_TOKENIZER_SOURCE_DIR=${AX_QWEN_ASR_TOKENIZER_SOURCE_DIR}")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel "${AX_QWEN_ASR_BUILD_JOBS:-4}"
cmake --install "${build_dir}" --component Runtime
