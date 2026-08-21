cmake_minimum_required(VERSION 3.18)
project(ax_qwen_asr LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(AX_QWEN_ASR_BUILD_DEVICE "Build the AX650/MC50 AXEngine backend" OFF)
option(AX_QWEN_ASR_BUILD_HOST_TESTS "Build host-only native test libraries" OFF)
set(AX_QWEN_ASR_AXLLM_SOURCE_DIR "" CACHE PATH "Existing AXERA-TECH/ax-llm checkout")
set(AX_QWEN_ASR_TOKENIZER_SOURCE_DIR "" CACHE PATH "Existing tokenizer.axera checkout")
set(BSP_MSP_DIR "" CACHE PATH "AX650 SDK msp/out directory")

add_library(ax_qwen_asr_core STATIC
    src/audio_features.cpp
    src/model_config.cpp
    src/prompt.cpp
)
set_target_properties(ax_qwen_asr_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
target_include_directories(ax_qwen_asr_core PUBLIC src)
target_compile_options(ax_qwen_asr_core PRIVATE -Wall -Wextra -Wpedantic)

set(AX_QWEN_ASR_RUNNER_SOURCES src/qwen_asr_runner.cpp)
set(AX_QWEN_ASR_DEVICE_LIBRARIES "")

if(AX_QWEN_ASR_BUILD_DEVICE)
    include(FetchContent)

    if(NOT AX_QWEN_ASR_AXLLM_SOURCE_DIR)
        FetchContent_Declare(
            axllm_source
            GIT_REPOSITORY https://github.com/AXERA-TECH/ax-llm.git
            GIT_TAG 29427ac732044b423cd191d8bd37aff3ec772f21
            GIT_SHALLOW FALSE
            GIT_SUBMODULES ""
        )
        FetchContent_GetProperties(axllm_source)
        if(NOT axllm_source_POPULATED)
            FetchContent_Populate(axllm_source)
        endif()
        set(AX_QWEN_ASR_AXLLM_SOURCE_DIR "${axllm_source_SOURCE_DIR}")
    endif()

    if(NOT AX_QWEN_ASR_TOKENIZER_SOURCE_DIR)
        FetchContent_Declare(
            tokenizer_source
            GIT_REPOSITORY https://github.com/ZHEQIUSHUI/tokenizer.axera.git
            GIT_TAG ffa443bdd260e91c3eb7d8c6baeedb02f42bbbd1
            GIT_SHALLOW FALSE
        )
        FetchContent_GetProperties(tokenizer_source)
        if(NOT tokenizer_source_POPULATED)
            FetchContent_Populate(tokenizer_source)
        endif()
        set(AX_QWEN_ASR_TOKENIZER_SOURCE_DIR "${tokenizer_source_SOURCE_DIR}")
    endif()

    if(NOT EXISTS "${AX_QWEN_ASR_AXLLM_SOURCE_DIR}/src/runner/ax_model_runner/ax_model_runner_ax650.cpp")
        message(FATAL_ERROR "AX_QWEN_ASR_AXLLM_SOURCE_DIR is not the pinned ax-llm source tree")
    endif()
    if(NOT EXISTS "${AX_QWEN_ASR_TOKENIZER_SOURCE_DIR}/src/tokenizer/tokenizer_optimized.cpp")
        message(FATAL_ERROR "AX_QWEN_ASR_TOKENIZER_SOURCE_DIR is not tokenizer.axera")
    endif()

    if(NOT BSP_MSP_DIR)
        if(EXISTS "/soc/include/ax_engine_api.h")
            set(BSP_MSP_DIR "/soc")
        else()
            message(FATAL_ERROR "AX650 build requires -DBSP_MSP_DIR=<SDK msp/out>")
        endif()
    endif()
    if(NOT EXISTS "${BSP_MSP_DIR}/include/ax_engine_api.h" OR
       NOT EXISTS "${BSP_MSP_DIR}/lib/libax_engine.so")
        message(FATAL_ERROR "BSP_MSP_DIR must contain include/ax_engine_api.h and lib/libax_engine.so")
    endif()

    add_library(ax_qwen_asr_axrunner STATIC
        "${AX_QWEN_ASR_AXLLM_SOURCE_DIR}/src/runner/ax_model_runner/ax_model_runner_ax650.cpp"
        "${AX_QWEN_ASR_AXLLM_SOURCE_DIR}/src/runner/utils/memory_utils.cpp"
        "${AX_QWEN_ASR_AXLLM_SOURCE_DIR}/src/runner/utils/logger.cpp"
    )
    set_target_properties(ax_qwen_asr_axrunner PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )
    target_include_directories(ax_qwen_asr_axrunner PUBLIC
        "${AX_QWEN_ASR_AXLLM_SOURCE_DIR}/src/runner/ax_model_runner"
        "${AX_QWEN_ASR_AXLLM_SOURCE_DIR}/src/runner/utils"
        "${BSP_MSP_DIR}/include"
    )
    target_compile_options(ax_qwen_asr_axrunner PRIVATE -Wall -Wextra)

    add_library(ax_qwen_asr_tokenizer STATIC
        "${AX_QWEN_ASR_TOKENIZER_SOURCE_DIR}/src/tokenizer/tokenizer_optimized.cpp"
    )
    set_target_properties(ax_qwen_asr_tokenizer PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )
    target_include_directories(ax_qwen_asr_tokenizer PUBLIC
        "${AX_QWEN_ASR_TOKENIZER_SOURCE_DIR}/src"
    )

    list(APPEND AX_QWEN_ASR_DEVICE_LIBRARIES
        ax_qwen_asr_axrunner
        ax_qwen_asr_tokenizer
        "${BSP_MSP_DIR}/lib/libax_engine.so"
        "${BSP_MSP_DIR}/lib/libax_interpreter.so"
        "${BSP_MSP_DIR}/lib/libax_sys.so"
        pthread
        ${CMAKE_DL_LIBS}
    )
endif()

add_library(ax_qwen_asr SHARED
    src/ax_qwen_asr.cpp
    ${AX_QWEN_ASR_RUNNER_SOURCES}
)
target_include_directories(ax_qwen_asr PRIVATE include src)
target_link_libraries(ax_qwen_asr PRIVATE ax_qwen_asr_core ${AX_QWEN_ASR_DEVICE_LIBRARIES})
target_compile_options(ax_qwen_asr PRIVATE -Wall -Wextra -Wpedantic)
if(UNIX AND NOT APPLE)
    target_link_options(ax_qwen_asr PRIVATE
        "-Wl,--exclude-libs,ALL"
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    )
    set_property(TARGET ax_qwen_asr APPEND PROPERTY
        LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()
set_target_properties(ax_qwen_asr PROPERTIES
    OUTPUT_NAME ax_qwen_asr
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION 1.0.0
    SOVERSION 1
)
if(AX_QWEN_ASR_BUILD_DEVICE)
    target_compile_definitions(ax_qwen_asr PRIVATE AX_QWEN_ASR_WITH_AXENGINE=1)
endif()

install(TARGETS ax_qwen_asr LIBRARY DESTINATION lib COMPONENT Runtime)
install(FILES include/ax_qwen_asr.h DESTINATION include COMPONENT Development)

if(AX_QWEN_ASR_BUILD_HOST_TESTS)
    add_library(ax_qwen_asr_test_core SHARED
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../tests/ax_m4c/native_core_api.cpp"
    )
    target_include_directories(ax_qwen_asr_test_core PRIVATE src)
    target_link_libraries(ax_qwen_asr_test_core PRIVATE ax_qwen_asr_core)

    add_library(ax_qwen_asr_test_abi SHARED
        src/ax_qwen_asr.cpp
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../tests/ax_m4c/native_fake_runner.cpp"
    )
    target_include_directories(ax_qwen_asr_test_abi PRIVATE include src)
    target_link_libraries(ax_qwen_asr_test_abi PRIVATE ax_qwen_asr_core)
endif()
