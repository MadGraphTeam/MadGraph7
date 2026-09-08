#pragma once

#include <chrono>
#include <string>

#include <nlohmann/json.hpp>

namespace madspace {

// Accumulates JSON status content from possibly multiple sources and
// periodically rewrites it to a file.
class StatusFile {
public:
    // file_name is where the status is written; writes happen at most every
    // min_interval_sec, unless forced (see write()).
    explicit StatusFile(const std::string& file_name, double min_interval_sec = 10.0);

    // Merges content into the accumulated status (top-level keys are
    // overwritten, "run_times" is merged one level deeper) and rewrites the
    // file if force_write is set, if this is the first write, or if
    // min_interval_sec has passed since the last write.
    void write(const nlohmann::json& content, bool force_write = false);

    // If a status was ever written and its "status" field is not "done",
    // marks it "done" and writes it out one last time.
    ~StatusFile();

private:
    std::string _file_name;
    std::chrono::duration<double> _min_interval;
    std::chrono::time_point<std::chrono::steady_clock> _last_write_time;
    bool _has_written = false;
    nlohmann::json _content = nlohmann::json::object();
};

} // namespace madspace
