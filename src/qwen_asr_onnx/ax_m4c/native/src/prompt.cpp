#include "prompt.hpp"

#include "model_config.hpp"

#include <stdexcept>

namespace ax_qwen_asr {

std::vector<std::uint32_t> BuildPromptIds(std::size_t audio_token_count) {
    // 固定使用空 context、未指定 language 的 Qwen3-ASR prompt 布局。
    std::vector<std::uint32_t> ids = {
        kImStartId, kSystemId, kNewlineId, kImEndId, kNewlineId,
        kImStartId, kUserId, kNewlineId, kAudioStartId,
    };
    ids.insert(ids.end(), audio_token_count, kAudioPadId);
    ids.insert(ids.end(), {
        kAudioEndId, kImEndId, kNewlineId,
        kImStartId, kAssistantId, kNewlineId,
    });
    if (ids.size() > kMaxPrefillTokens) {
        throw std::invalid_argument(
            "prompt token 数 " + std::to_string(ids.size()) +
            " 超过 P448 固定上限 " + std::to_string(kMaxPrefillTokens));
    }
    return ids;
}

}  // namespace ax_qwen_asr
