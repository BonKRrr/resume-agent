#pragma once
#include <string>
#include <vector>
#include <map>

struct Bullet {
    std::string id;                  // unique, stable
    std::string text;                // original bullet text
    std::vector<std::string> tags;   // C++, Linux, backend, testing
};

struct Experience {
    std::string id;
    std::string title;
    std::string organization;
    std::string dates;
    std::vector<Bullet> bullets;
};

struct Project {
    std::string id;
    std::string name;
    std::string context;             // course / personal / work
    std::vector<Bullet> bullets;
};

struct AbstractResume {
    std::vector<Experience> experiences;
    std::vector<Project> projects;
};
