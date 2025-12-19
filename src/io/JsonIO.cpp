#include "io/JsonIO.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static void require_object(const json& j, const std::string& where) {
    if (!j.is_object()) {
        throw std::runtime_error(where + " must be an object");
    }
}

static void require_array(const json& j, const std::string& where) {
    if (!j.is_array()) {
        throw std::runtime_error(where + " must be an array");
    }
}

static std::string require_string(const json& j, const char* key, const std::string& where) {
    if (!j.contains(key)) {
        throw std::runtime_error(where + " missing required field: " + std::string(key));
    }
    if (!j.at(key).is_string()) {
        throw std::runtime_error(where + "." + std::string(key) + " must be a string");
    }
    return j.at(key).get<std::string>();
}

static std::vector<std::string> require_string_array(const json& j, const char* key, const std::string& where) {
    if (!j.contains(key)) {
        throw std::runtime_error(where + " missing required field: " + std::string(key));
    }
    const json& arr = j.at(key);
    if (!arr.is_array()) {
        throw std::runtime_error(where + "." + std::string(key) + " must be an array");
    }
    std::vector<std::string> out;
    out.reserve(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
        if (!arr.at(i).is_string()) {
            std::ostringstream oss;
            oss << where << "." << key << "[" << i << "] must be a string";
            throw std::runtime_error(oss.str());
        }
        out.push_back(arr.at(i).get<std::string>());
    }
    return out;
}

static Bullet parseBullet(const json& j, const std::string& where) {
    require_object(j, where);

    Bullet b;
    b.id   = require_string(j, "id", where);
    b.text = require_string(j, "text", where);
    b.tags = require_string_array(j, "tags", where);
    return b;
}

static Experience parseExperience(const json& j, const std::string& where) {
    require_object(j, where);

    Experience e;
    e.id           = require_string(j, "id", where);
    e.title        = require_string(j, "title", where);
    e.organization = require_string(j, "organization", where);
    e.dates        = require_string(j, "dates", where);

    if (!j.contains("bullets")) {
        throw std::runtime_error(where + " missing required field: bullets");
    }
    const json& bullets = j.at("bullets");
    require_array(bullets, where + ".bullets");

    for (size_t i = 0; i < bullets.size(); ++i) {
        std::ostringstream oss;
        oss << where << ".bullets[" << i << "]";
        e.bullets.push_back(parseBullet(bullets.at(i), oss.str()));
    }

    return e;
}

static Project parseProject(const json& j, const std::string& where) {
    require_object(j, where);

    Project p;
    p.id      = require_string(j, "id", where);
    p.name    = require_string(j, "name", where);
    p.context = require_string(j, "context", where);

    if (!j.contains("bullets")) {
        throw std::runtime_error(where + " missing required field: bullets");
    }
    const json& bullets = j.at("bullets");
    require_array(bullets, where + ".bullets");

    for (size_t i = 0; i < bullets.size(); ++i) {
        std::ostringstream oss;
        oss << where << ".bullets[" << i << "]";
        p.bullets.push_back(parseBullet(bullets.at(i), oss.str()));
    }

    return p;
}

AbstractResume loadAbstractResume(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open resume file: " + path);
    }

    json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("failed to parse JSON: ") + e.what());
    }

    require_object(j, "root");

    AbstractResume ar;

    if (j.contains("experiences")) {
        const json& exps = j.at("experiences");
        require_array(exps, "root.experiences");
        for (size_t i = 0; i < exps.size(); ++i) {
            std::ostringstream oss;
            oss << "root.experiences[" << i << "]";
            ar.experiences.push_back(parseExperience(exps.at(i), oss.str()));
        }
    }

    if (j.contains("projects")) {
        const json& projs = j.at("projects");
        require_array(projs, "root.projects");
        for (size_t i = 0; i < projs.size(); ++i) {
            std::ostringstream oss;
            oss << "root.projects[" << i << "]";
            ar.projects.push_back(parseProject(projs.at(i), oss.str()));
        }
    }

    return ar;
}
