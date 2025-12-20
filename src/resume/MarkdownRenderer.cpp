#include "resume/MarkdownRenderer.hpp"

#include <fstream>
#include <stdexcept>

namespace resume {

static std::string exp_header(const Experience& e) {
    std::string h = e.title;
    if (!e.organization.empty()) {
        if (!h.empty()) h += " — ";
        h += e.organization;
    }
    if (!e.dates.empty()) {
        if (!h.empty()) h += " ";
        h += "(" + e.dates + ")";
    }
    return h;
}

static std::string proj_header(const Project& p) {
    std::string h = p.name;
    if (!p.context.empty()) {
        h += " (" + p.context + ")";
    }
    return h;
}

ConcreteResume build_concrete_resume(const AbstractResume& resume, const std::vector<ScoredBullet>& selected) {
    std::unordered_map<std::string, const Experience*> exp_by_id;
    exp_by_id.reserve(resume.experiences.size() * 2 + 8);
    for (const auto& e : resume.experiences) exp_by_id[e.id] = &e;

    std::unordered_map<std::string, const Project*> proj_by_id;
    proj_by_id.reserve(resume.projects.size() * 2 + 8);
    for (const auto& p : resume.projects) proj_by_id[p.id] = &p;

    // Section -> parent_key -> entry index
    struct EntryKey {
        std::string section;
        std::string parent_id;
    };

    ConcreteResume cr;

    auto ensure_section = [&](const std::string& title) -> size_t {
        for (size_t i = 0; i < cr.sections.size(); ++i) {
            if (cr.sections[i].title == title) return i;
        }
        cr.sections.push_back(ConcreteSection{title, {}});
        return cr.sections.size() - 1;
    };

    std::unordered_map<std::string, size_t> entry_index; // "Section::parent_id" -> idx in that section
    entry_index.reserve(selected.size() * 2 + 8);

    for (const auto& sb : selected) {
        const std::string sec = sb.section;
        const size_t si = ensure_section(sec);

        const std::string ek = sec + "::" + sb.parent_id;

        auto it = entry_index.find(ek);
        if (it == entry_index.end()) {
            ConcreteEntry e;

            if (sec == "Experience") {
                auto eit = exp_by_id.find(sb.parent_id);
                if (eit != exp_by_id.end() && eit->second) e.header = exp_header(*eit->second);
                else e.header = sb.parent_title;
            } else if (sec == "Project") {
                auto pit = proj_by_id.find(sb.parent_id);
                if (pit != proj_by_id.end() && pit->second) e.header = proj_header(*pit->second);
                else e.header = sb.parent_title;
            } else {
                e.header = sb.parent_title;
            }

            cr.sections[si].entries.push_back(std::move(e));
            entry_index[ek] = cr.sections[si].entries.size() - 1;
            it = entry_index.find(ek);
        }

        cr.sections[si].entries[it->second].bullets.push_back(sb.text);
    }

    return cr;
}

std::string render_markdown(const ConcreteResume& cr) {
    std::string out;

    for (const auto& sec : cr.sections) {
        out += "## " + sec.title + "\n\n";
        for (const auto& e : sec.entries) {
            out += "**" + e.header + "**\n";
            for (const auto& b : e.bullets) {
                out += "- " + b + "\n";
            }
            out += "\n";
        }
    }

    return out;
}

void write_markdown(const std::filesystem::path& out_path, const std::string& md) {
    std::filesystem::create_directories(out_path.parent_path());

    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("Failed to open output file: " + out_path.string());

    out << md;
    if (md.empty() || md.back() != '\n') out << "\n";
}

}  // namespace resume
