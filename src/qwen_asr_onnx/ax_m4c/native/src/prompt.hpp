#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ax_qwen_asr {

constexpr std::uint32_t kAudioStartId = 151669;
constexpr std::uint32_t kAudioEndId = 151670;
constexpr std::uint32_t kAudioPadId = 151676;
constexpr std::uint32_t kImStartId = 151644;
constexpr std::uint32_t kImEndId = 151645;
constexpr std::uint32_t kEndOfTextId = 151643;
constexpr std::uint32_t kNewlineId = 198;
constexpr std::uint32_t kSystemId = 8948;
constexpr std::uint32_t kUserId = 872;
constexpr std::uint32_t kAssistantId = 77091;

std::vector<std::uint32_t> BuildPromptIds(std::size_t audio_token_count);

}  // namespace ax_qwen_asr
