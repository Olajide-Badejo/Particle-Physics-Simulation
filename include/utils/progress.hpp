#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace cps {

// Terminal progress reporting.
//
// Two constraints shape this. A long run must show that it is alive and say
// when it will finish, and it must not spend measurable time doing so, since
// the whole point of the exercise is timing. The redraw is therefore rate
// limited to ten updates per second and the bar is only drawn when standard
// error is a terminal. Under continuous integration the same information goes
// out as ordinary lines, occasionally, so the log stays readable.
class ProgressBar {
   public:
    ProgressBar(std::string label, long long total);

    // Safe to call every step; most calls return immediately.
    void update(long long current);

    void finish();

    double elapsed_seconds() const;

   private:
    void draw(long long current, bool final_draw);

    std::string label_;
    long long total_ = 0;
    long long last_drawn_ = -1;
    bool tty_ = false;
    bool finished_ = false;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_draw_;
};

// Formats a duration as h:mm:ss or m:ss, whichever is shorter and still exact.
std::string format_duration(double seconds);

// Human readable count with thousands separators, for particle counts.
std::string format_count(long long value);

}  // namespace cps
