#include "llm/OllamaLLMClient.hpp"
#include "nlohmann/json.hpp"

#include <windows.h>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace llm {

static std::string read_all(std::istream& in) {
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static bool ensure_dir(const fs::path& p) {
    try { fs::create_directories(p); return true; }
    catch (...) { return false; }
}

// very small FNV-1a hash for cache keys (deterministic, no deps)
static uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= (uint64_t)c;
        h *= 1099511628211ull;
    }
    return h;
}

static std::string hex_u64(uint64_t x) {
    const char* hex = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = hex[x & 0xF];
        x >>= 4;
    }
    return out;
}

// Run a command and return exit code. (cmdline must be mutable for CreateProcess)
static DWORD run_wait_exitcode(const std::string& cmdline_utf8) {
    STARTUPINFOA si{};
    si.cb = sizeof(si);

    PROCESS_INFORMATION pi{};
    std::string cmdline = cmdline_utf8;

    BOOL ok = CreateProcessA(
        NULL,
        cmdline.data(),
        NULL, NULL,
        FALSE,
        CREATE_NO_WINDOW,
        NULL,
        NULL,
        &si,
        &pi
    );

    if (!ok) return (DWORD)-1;

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    return code;
}

OllamaLLMClient::OllamaLLMClient(const std::string& model, const std::string& cache_dir)
    : model_(model), cache_dir_(cache_dir) {
    ensure_dir(cache_dir_);
}

std::string OllamaLLMClient::cache_key(const std::string& task, const std::string& input) const {
    std::string s = model_ + "\n" + task + "\n" + input;
    return task + "_v1-" + hex_u64(fnv1a64(s));
}

bool OllamaLLMClient::load_cache(const std::string& key, std::string& out) const {
    fs::path p = cache_dir_ / (key + ".json");
    std::ifstream f(p, std::ios::in);
    if (!f) return false;
    out = read_all(f);
    return true;
}

void OllamaLLMClient::save_cache(const std::string& key, const std::string& content) const {
    fs::path p = cache_dir_ / (key + ".json");
    std::ofstream f(p, std::ios::out | std::ios::trunc);
    if (!f) return;
    f << content;
}

std::string OllamaLLMClient::prompt_analyzer_onecall(const std::string& posting_text) const {
    std::ostringstream p;
    p <<
R"(You are extracting job-skill evidence from a job posting.
Return ONLY valid JSON. No markdown. No commentary.

Output schema:
{
  "evidence": [
    {
      "span_type": "requirement|preferred|responsibility|other",
      "span_text": "...",
      "polarity": "positive|negated",
      "strength": "must|should|nice|unknown",
      "skills": [
        {"raw":"...","canonical":"...","confidence":0.0}
      ]
    }
  ]
}

Rules:
- Only include spans that actually express requirements/preferences/responsibilities.
- "skills" must be skills/tools/techniques that appear explicitly in span_text.
- If a skill is explicitly NOT required, set polarity="negated".
- Use strength="must" for required, "should" for preferred, "nice" for bonus/optional.
- Keep span_text short (1-3 sentences or a bullet block).
- Keep outputs small: at most 10 evidence items, at most 5 skills per evidence item.
- confidence in [0,1].
- If nothing found, return {"evidence":[]}.

Job posting:
)";
    p << posting_text;
    return p.str();
}

std::string OllamaLLMClient::prompt_segmenter(const std::string& posting_text) const {
    std::ostringstream p;
    p <<
R"(You are extracting requirement spans from a job posting.
Return ONLY valid JSON. No markdown. No commentary.

Output schema:
{"spans":[{"type":"requirement|preferred|responsibility|other","text":"..."}]}

Rules:
- Spans should be short (1-3 sentences or a bullet block).
- Capture only spans that actually contain requirements/preferences/responsibilities.
- If nothing is found, return {"spans":[]}.

Job posting:
)";
    p << posting_text;
    return p.str();
}

std::string OllamaLLMClient::prompt_extractor(const Span& span) const {
    std::ostringstream p;
    p <<
R"(Extract skills from the given span.
Return ONLY valid JSON. No markdown. No commentary.

Output schema:
{
  "span_type":"requirement|preferred|responsibility|other",
  "span_text":"...",
  "polarity":"positive|negated",
  "strength":"must|should|nice|unknown",
  "skills":[{"raw":"...","canonical":"...","confidence":0.0}]
}

Rules:
- Only extract skills that are explicitly present in the span text.
- "canonical" should be the normalized name if obvious alias (e.g., C++17 -> C++).
- If a skill is stated as NOT required / not needed, set polarity="negated".
- If the span is a must/required, strength="must"; if preferred/bonus, strength="nice" or "should".
- confidence in [0,1].
- If no skills, return skills=[].

Span type: )" << span.type << "\n"
    << "Span text:\n" << span.text;
    return p.str();
}

std::string OllamaLLMClient::run_ollama_json(const std::string& prompt) const {
    ensure_dir(cache_dir_);

    fs::path payload = cache_dir_ / "ollama_payload.tmp.json";
    fs::path resp    = cache_dir_ / "ollama_response.tmp.json";
    fs::path err     = cache_dir_ / "ollama_curl_error.tmp.txt";

    auto esc = [](const std::string& s) {
        std::string o;
        o.reserve(s.size() + 32);
        for (char c : s) {
            switch (c) {
                case '\\': o += "\\\\"; break;
                case '"':  o += "\\\""; break;
                case '\n': o += "\\n";  break;
                case '\r': o += "\\r";  break;
                case '\t': o += "\\t";  break;
                default:   o += c; break;
            }
        }
        return o;
    };

    {
        std::ofstream f(payload, std::ios::out | std::ios::trunc);
        if (!f) return "";

        f << "{"
          << "\"model\":\""  << esc(model_)  << "\","
          << "\"prompt\":\"" << esc(prompt) << "\","
          << "\"stream\":false,"
          << "\"format\":\"json\","
          << "\"options\":{"
              << "\"temperature\":0,"
              << "\"num_predict\":3072"
          << "}"
          << "}";
    }

    // write response to file (avoid pipe / quoting issues)
    std::ostringstream cmd;
    cmd << "cmd.exe /C curl.exe -s "
        << "-o \"" << resp.string() << "\" "
        << "http://127.0.0.1:11434/api/generate "
        << "-H \"Content-Type: application/json\" "
        << "--data-binary \"@" << payload.string() << "\" "
        << "2> \"" << err.string() << "\"";

    DWORD code = run_wait_exitcode(cmd.str());
    if (code != 0) return "";

    std::ifstream rf(resp, std::ios::in);
    if (!rf) return "";
    std::string out = read_all(rf);
    if (out.empty()) return "";

    try {
        auto j = nlohmann::json::parse(out);
        if (j.contains("response") && j["response"].is_string()) {
            return j["response"].get<std::string>();
        }
    } catch (...) {
        return "";
    }

    return "";
}

std::vector<Span> OllamaLLMClient::parse_spans_json(const std::string& s) const {
    std::vector<Span> spans;
    json j;
    try { j = json::parse(s); }
    catch (...) { return spans; }

    if (!j.is_object()) return spans;
    if (!j.contains("spans") || !j["spans"].is_array()) return spans;

    for (const auto& it : j["spans"]) {
        if (!it.is_object()) continue;
        Span sp;
        if (it.contains("type") && it["type"].is_string()) sp.type = it["type"].get<std::string>();
        if (it.contains("text") && it["text"].is_string()) sp.text = it["text"].get<std::string>();
        if (!sp.text.empty()) spans.push_back(std::move(sp));
    }
    return spans;
}

EvidenceSpan OllamaLLMClient::parse_evidence_json(const std::string& s) const {
    EvidenceSpan ev;
    json j;
    try { j = json::parse(s); }
    catch (...) { return ev; }

    if (!j.is_object()) return ev;

    if (j.contains("span_type") && j["span_type"].is_string()) ev.span_type = j["span_type"].get<std::string>();
    if (j.contains("span_text") && j["span_text"].is_string()) ev.span_text = j["span_text"].get<std::string>();
    if (j.contains("polarity") && j["polarity"].is_string()) ev.polarity = j["polarity"].get<std::string>();
    if (j.contains("strength") && j["strength"].is_string()) ev.strength = j["strength"].get<std::string>();

    if (j.contains("skills") && j["skills"].is_array()) {
        for (const auto& s2 : j["skills"]) {
            if (!s2.is_object()) continue;
            SkillHit sh;
            if (s2.contains("raw") && s2["raw"].is_string()) sh.raw = s2["raw"].get<std::string>();
            if (s2.contains("canonical") && s2["canonical"].is_string()) sh.canonical = s2["canonical"].get<std::string>();
            if (s2.contains("confidence") && (s2["confidence"].is_number_float() || s2["confidence"].is_number_integer()))
                sh.confidence = s2["confidence"].get<double>();
            if (!sh.raw.empty() || !sh.canonical.empty()) ev.skills.push_back(std::move(sh));
        }
    }

    return ev;
}

std::vector<EvidenceSpan> OllamaLLMClient::parse_evidence_list_json(const std::string& s) const {
    std::vector<EvidenceSpan> out;
    json j;
    try { j = json::parse(s); }
    catch (...) { return out; }

    if (!j.is_object()) return out;
    if (!j.contains("evidence") || !j["evidence"].is_array()) return out;

    for (const auto& e : j["evidence"]) {
        if (!e.is_object()) continue;

        EvidenceSpan ev;
        if (e.contains("span_type") && e["span_type"].is_string()) ev.span_type = e["span_type"].get<std::string>();
        if (e.contains("span_text") && e["span_text"].is_string()) ev.span_text = e["span_text"].get<std::string>();
        if (e.contains("polarity") && e["polarity"].is_string()) ev.polarity = e["polarity"].get<std::string>();
        if (e.contains("strength") && e["strength"].is_string()) ev.strength = e["strength"].get<std::string>();

        if (e.contains("skills") && e["skills"].is_array()) {
            for (const auto& s2 : e["skills"]) {
                if (!s2.is_object()) continue;
                SkillHit sh;
                if (s2.contains("raw") && s2["raw"].is_string()) sh.raw = s2["raw"].get<std::string>();
                if (s2.contains("canonical") && s2["canonical"].is_string()) sh.canonical = s2["canonical"].get<std::string>();
                if (s2.contains("confidence") && (s2["confidence"].is_number_float() || s2["confidence"].is_number_integer()))
                    sh.confidence = s2["confidence"].get<double>();
                if (!sh.raw.empty() || !sh.canonical.empty()) ev.skills.push_back(std::move(sh));
            }
        }

        if (!ev.span_type.empty() || !ev.span_text.empty() || !ev.skills.empty()) out.push_back(std::move(ev));
    }

    return out;
}

std::vector<EvidenceSpan> OllamaLLMClient::analyze_posting(const std::string& posting_id,
                                                          const std::string& posting_text) {
    const std::string key = cache_key("analyze", posting_id + "\n" + posting_text);

    std::string cached;
    if (load_cache(key, cached)) {
        auto a = cached.find('{');
        auto b = cached.rfind('}');
        std::string s = cached;
        if (a != std::string::npos && b != std::string::npos && b > a) s = cached.substr(a, b - a + 1);
        return parse_evidence_list_json(s);
    }

    std::string prompt = prompt_analyzer_onecall(posting_text);
    std::string out = run_ollama_json(prompt);
    if (out.empty()) return {};

    auto a = out.find('{');
    auto b = out.rfind('}');
    if (a != std::string::npos && b != std::string::npos && b > a) out = out.substr(a, b - a + 1);

    auto parsed = parse_evidence_list_json(out);
    if (!parsed.empty()) save_cache(key, out);
    return parsed;
}

std::vector<Span> OllamaLLMClient::segment(const std::string& posting_text) {
    const std::string key = cache_key("segment", posting_text);

    std::string cached;
    if (load_cache(key, cached)) return parse_spans_json(cached);

    std::string prompt = prompt_segmenter(posting_text);
    std::string out = run_ollama_json(prompt);
    if (out.empty()) return {};

    auto a = out.find('{');
    auto b = out.rfind('}');
    if (a != std::string::npos && b != std::string::npos && b > a) out = out.substr(a, b - a + 1);

    save_cache(key, out);
    return parse_spans_json(out);
}

EvidenceSpan OllamaLLMClient::extract(const Span& span) {
    const std::string key = cache_key("extract", span.type + "\n" + span.text);

    std::string cached;
    if (load_cache(key, cached)) return parse_evidence_json(cached);

    std::string prompt = prompt_extractor(span);
    std::string out = run_ollama_json(prompt);
    if (out.empty()) return EvidenceSpan{};

    auto a = out.find('{');
    auto b = out.rfind('}');
    if (a != std::string::npos && b != std::string::npos && b > a) out = out.substr(a, b - a + 1);

    save_cache(key, out);
    return parse_evidence_json(out);
}

} // namespace llm
