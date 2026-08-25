from __future__ import annotations


def test_prompt_tokens_match_fixed_asr_layout(core_native) -> None:
    ffi, lib = core_native
    required = ffi.new("size_t *")
    assert lib.ax_qwen_asr_test_prompt(65, ffi.NULL, 0, required) == 2
    assert required[0] == 80
    output = ffi.new("uint32_t[]", required[0])
    assert lib.ax_qwen_asr_test_prompt(65, output, required[0], required) == 0
    ids = list(output)
    expected = [151644, 8948, 198, 151645, 198]
    expected += [151644, 872, 198, 151669]
    expected += [151676] * 65
    expected += [151670, 151645, 198, 151644, 77091, 198]
    assert ids == expected


def test_prompt_rejects_more_than_p448(core_native) -> None:
    ffi, lib = core_native
    required = ffi.new("size_t *")
    assert lib.ax_qwen_asr_test_prompt(434, ffi.NULL, 0, required) == 3
    assert "P448" in ffi.string(lib.ax_qwen_asr_test_last_error()).decode()
