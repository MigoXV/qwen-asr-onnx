#include "model_config.hpp"

#include <cctype>
#include <charconv>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <system_error>
#include <variant>

namespace ax_qwen_asr {
namespace {

struct JsonValue {
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;
    std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array, Object> value;
};

class JsonParser {
public:
    explicit JsonParser(std::string text) : text_(std::move(text)) {}

    JsonValue Parse() {
        SkipSpace();
        JsonValue result = ParseValue();
        SkipSpace();
        if (position_ != text_.size()) Fail("JSON 根值之后存在多余内容");
        return result;
    }

private:
    JsonValue ParseValue() {
        SkipSpace();
        if (position_ >= text_.size()) Fail("JSON 值意外结束");
        const char c = text_[position_];
        if (c == '{') return JsonValue{ParseObject()};
        if (c == '[') return JsonValue{ParseArray()};
        if (c == '"') return JsonValue{ParseString()};
        if (c == 't') return ParseLiteral("true", JsonValue{true});
        if (c == 'f') return ParseLiteral("false", JsonValue{false});
        if (c == 'n') return ParseLiteral("null", JsonValue{nullptr});
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return ParseNumber();
        Fail("无法识别的 JSON 值");
    }

    JsonValue ParseLiteral(const char* literal, JsonValue result) {
        const std::string expected(literal);
        if (text_.compare(position_, expected.size(), expected) != 0) {
            Fail("非法 JSON 字面量");
        }
        position_ += expected.size();
        return result;
    }

    JsonValue::Object ParseObject() {
        Expect('{');
        JsonValue::Object object;
        SkipSpace();
        if (Consume('}')) return object;
        while (true) {
            SkipSpace();
            if (position_ >= text_.size() || text_[position_] != '"') {
                Fail("JSON object key 必须是字符串");
            }
            std::string key = ParseString();
            SkipSpace();
            Expect(':');
            if (!object.emplace(key, ParseValue()).second) {
                Fail("JSON object 存在重复字段：" + key);
            }
            SkipSpace();
            if (Consume('}')) break;
            Expect(',');
        }
        return object;
    }

    JsonValue::Array ParseArray() {
        Expect('[');
        JsonValue::Array array;
        SkipSpace();
        if (Consume(']')) return array;
        while (true) {
            array.push_back(ParseValue());
            SkipSpace();
            if (Consume(']')) break;
            Expect(',');
        }
        return array;
    }

    std::string ParseString() {
        Expect('"');
        std::string result;
        while (position_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(text_[position_++]);
            if (c == '"') return result;
            if (c < 0x20) Fail("JSON 字符串包含控制字符");
            if (c != '\\') {
                result.push_back(static_cast<char>(c));
                continue;
            }
            if (position_ >= text_.size()) Fail("JSON 转义意外结束");
            const char escape = text_[position_++];
            switch (escape) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': AppendUnicode(result); break;
                default: Fail("JSON 字符串包含非法转义");
            }
        }
        Fail("JSON 字符串未结束");
    }

    void AppendUnicode(std::string& output) {
        const std::uint32_t codepoint = ParseHex4();
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
            Fail("配置 JSON 不接受 UTF-16 surrogate 转义");
        }
        if (codepoint <= 0x7F) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FF) {
            output.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            output.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }

    std::uint32_t ParseHex4() {
        if (position_ + 4 > text_.size()) Fail("JSON unicode 转义长度不足");
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<std::uint32_t>(c - 'A' + 10);
            else Fail("JSON unicode 转义包含非十六进制字符");
        }
        return value;
    }

    JsonValue ParseNumber() {
        const std::size_t start = position_;
        Consume('-');
        if (Consume('0')) {
            if (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) {
                Fail("JSON number 不允许前导零");
            }
        } else {
            RequireDigit();
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        }
        bool integral = true;
        if (Consume('.')) {
            integral = false;
            RequireDigit();
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            integral = false;
            ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
            RequireDigit();
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        }
        const std::string token = text_.substr(start, position_ - start);
        if (integral) {
            std::int64_t value = 0;
            const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
            if (result.ec != std::errc() || result.ptr != token.data() + token.size()) {
                Fail("JSON integer 超出 int64 范围");
            }
            return JsonValue{value};
        }
        try {
            std::size_t consumed = 0;
            const double value = std::stod(token, &consumed);
            if (consumed != token.size()) Fail("非法 JSON number");
            return JsonValue{value};
        } catch (const std::exception&) {
            Fail("非法 JSON number");
        }
    }

    void RequireDigit() {
        if (position_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[position_]))) {
            Fail("JSON number 缺少数字");
        }
    }

    void SkipSpace() {
        while (position_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_;
    }

    bool Consume(char expected) {
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void Expect(char expected) {
        if (!Consume(expected)) Fail(std::string("JSON 缺少字符 '") + expected + "'");
    }

    [[noreturn]] void Fail(const std::string& message) const {
        throw ModelConfigError(message + "（字节偏移 " + std::to_string(position_) + "）");
    }

    std::string text_;
    std::size_t position_ = 0;
};

std::string ReadTextFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw ModelFileError("无法读取文件：" + path.string());
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (!stream.good() && !stream.eof()) throw ModelFileError("读取文件失败：" + path.string());
    return buffer.str();
}

const JsonValue::Object& AsObject(const JsonValue& value, const std::string& field) {
    const auto* object = std::get_if<JsonValue::Object>(&value.value);
    if (!object) throw ModelConfigError(field + " 必须是 JSON object");
    return *object;
}

const JsonValue& Required(const JsonValue::Object& object, const std::string& field) {
    const auto it = object.find(field);
    if (it == object.end()) throw ModelConfigError("config.json 缺少字段：" + field);
    return it->second;
}

std::string RequiredString(const JsonValue::Object& object, const std::string& field) {
    const auto* value = std::get_if<std::string>(&Required(object, field).value);
    if (!value || value->empty()) throw ModelConfigError(field + " 必须是非空字符串");
    return *value;
}

std::int64_t RequiredInteger(const JsonValue::Object& object, const std::string& field) {
    const auto* value = std::get_if<std::int64_t>(&Required(object, field).value);
    if (!value) throw ModelConfigError(field + " 必须是整数");
    return *value;
}

bool RequiredBool(const JsonValue::Object& object, const std::string& field) {
    const auto* value = std::get_if<bool>(&Required(object, field).value);
    if (!value) throw ModelConfigError(field + " 必须是 bool");
    return *value;
}

void RequireEqual(const std::string& actual, const char* expected, const char* field) {
    if (actual != expected) {
        throw ModelConfigError(std::string(field) + " 固定要求为 " + expected + "，实际为 " + actual);
    }
}

void RequireEqual(std::int64_t actual, std::int64_t expected, const char* field) {
    if (actual != expected) {
        throw ModelConfigError(std::string(field) + " 固定要求为 " + std::to_string(expected) +
                               "，实际为 " + std::to_string(actual));
    }
}

std::filesystem::path SafeRelativePath(const std::string& value, const std::string& field) {
    const std::filesystem::path path(value);
    if (path.empty() || path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        throw ModelConfigError(field + " 必须是模型目录内的相对路径");
    }
    for (const auto& part : path) {
        if (part == "..") throw ModelConfigError(field + " 不允许包含 '..'");
    }
    return path.lexically_normal();
}

void RequireRegularFile(const std::filesystem::path& path) {
    std::error_code error;
    const auto status = std::filesystem::status(path, error);
    if (error || !std::filesystem::exists(status)) {
        throw ModelFileError("缺少模型文件：" + path.string());
    }
    if (!std::filesystem::is_regular_file(status)) {
        throw ModelFileError("模型路径不是普通文件：" + path.string());
    }
}

std::filesystem::path ResolveInsideRoot(
    const std::filesystem::path& root,
    const std::filesystem::path& relative,
    const std::string& field) {
    const auto joined = root / relative;
    RequireRegularFile(joined);
    std::error_code error;
    const auto canonical = std::filesystem::canonical(joined, error);
    if (error) throw ModelFileError("无法规范化模型文件：" + joined.string());
    const auto rel = canonical.lexically_relative(root);
    if (rel.empty() || *rel.begin() == "..") {
        throw ModelConfigError(field + " 解析后逃逸模型根目录：" + canonical.string());
    }
    return canonical;
}

void ValidateTemplate(const std::string& value) {
    std::size_t replacements = 0;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '%') continue;
        if (i + 1 >= value.size() || value[i + 1] != 'd') {
            throw ModelConfigError("template_filename_axmodel 只允许一个 %d 占位符");
        }
        ++replacements;
        ++i;
    }
    if (replacements != 1) {
        throw ModelConfigError("template_filename_axmodel 必须恰好包含一个 %d 占位符");
    }
}

std::string FormatLayer(const std::string& pattern, std::size_t index) {
    const auto marker = pattern.find("%d");
    return pattern.substr(0, marker) + std::to_string(index) + pattern.substr(marker + 2);
}

void ValidateGreedyPostConfig(const std::filesystem::path& path) {
    const auto root = AsObject(JsonParser(ReadTextFile(path)).Parse(), "post_config.json");
    for (const char* field : {
             "enable_temperature",
             "enable_repetition_penalty",
             "enable_top_p_sampling",
             "enable_top_k_sampling"}) {
        if (RequiredBool(root, field)) {
            throw ModelConfigError(std::string("固定贪心解码要求 post_config.json 中 ") + field + " 为 false");
        }
    }
}

}  // namespace

ModelConfig ModelConfig::Load(const std::filesystem::path& model_dir) {
    if (model_dir.empty()) throw ModelConfigError("model_dir 不能为空");

    std::error_code error;
    const auto root = std::filesystem::canonical(model_dir, error);
    if (error || !std::filesystem::is_directory(root)) {
        throw ModelFileError("模型目录不存在或不是目录：" + model_dir.string());
    }

    ModelConfig config;
    config.root = root;
    config.config_file = ResolveInsideRoot(root, "config.json", "config.json");
    const auto object = AsObject(JsonParser(ReadTextFile(config.config_file)).Parse(), "config.json");

    RequireEqual(
        RequiredString(object, "system_prompt"),
        "you are a helpful assistant.",
        "system_prompt");
    RequireEqual(
        RequiredString(object, "model_name"),
        "AXERA-TECH/Qwen3-0.6B",
        "model_name");
    RequireEqual(RequiredString(object, "tokenizer_type"), "Qwen3", "tokenizer_type");
    RequireEqual(RequiredInteger(object, "axmodel_num"), kDecoderLayerCount, "axmodel_num");
    RequireEqual(RequiredInteger(object, "tokens_embed_num"), kVocabularySize, "tokens_embed_num");
    RequireEqual(RequiredInteger(object, "tokens_embed_size"), kHiddenSize, "tokens_embed_size");
    if (!RequiredBool(object, "use_mmap_load_embed")) {
        throw ModelConfigError("use_mmap_load_embed 固定要求为 true");
    }
    if (!RequiredBool(object, "use_mmap_load_layer")) {
        throw ModelConfigError("use_mmap_load_layer 固定要求为 true");
    }

    const auto* devices = std::get_if<JsonValue::Array>(&Required(object, "devices").value);
    if (!devices || devices->size() != 1) throw ModelConfigError("devices 固定要求为 [0]");
    const auto* device = std::get_if<std::int64_t>(&devices->front().value);
    if (!device || *device != 0) throw ModelConfigError("devices 固定要求为 [0]");

    config.layer_template = RequiredString(object, "template_filename_axmodel");
    ValidateTemplate(config.layer_template);
    config.layer_count = kDecoderLayerCount;
    config.vocabulary_size = kVocabularySize;
    config.hidden_size = kHiddenSize;

    config.conv_frontend = ResolveInsideRoot(root, "conv_frontend.axmodel", "conv_frontend.axmodel");
    config.encoder = ResolveInsideRoot(root, "encoder.axmodel", "encoder.axmodel");
    config.post = ResolveInsideRoot(
        root,
        SafeRelativePath(RequiredString(object, "filename_post_axmodel"), "filename_post_axmodel"),
        "filename_post_axmodel");
    config.embedding = ResolveInsideRoot(
        root,
        SafeRelativePath(RequiredString(object, "filename_tokens_embed"), "filename_tokens_embed"),
        "filename_tokens_embed");
    config.tokenizer = ResolveInsideRoot(
        root,
        SafeRelativePath(RequiredString(object, "url_tokenizer_model"), "url_tokenizer_model"),
        "url_tokenizer_model");
    config.post_config = ResolveInsideRoot(
        root,
        SafeRelativePath(RequiredString(object, "post_config_path"), "post_config_path"),
        "post_config_path");

    config.decoder_layers.reserve(kDecoderLayerCount);
    for (std::size_t index = 0; index < kDecoderLayerCount; ++index) {
        config.decoder_layers.push_back(ResolveInsideRoot(
            root,
            SafeRelativePath(FormatLayer(config.layer_template, index), "template_filename_axmodel"),
            "template_filename_axmodel"));
    }

    const std::uintmax_t expected_bytes =
        kVocabularySize * kHiddenSize * kEmbeddingElementBytes;
    const auto embedding_bytes = std::filesystem::file_size(config.embedding, error);
    if (error || embedding_bytes != expected_bytes) {
        throw ModelFileError(
            "embedding 文件尺寸错误：期望 " + std::to_string(expected_bytes) +
            " 字节，实际 " + (error ? std::string("不可读取") : std::to_string(embedding_bytes)));
    }

    ValidateGreedyPostConfig(config.post_config);
    return config;
}

}  // namespace ax_qwen_asr
