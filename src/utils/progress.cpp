#include "cps/utils/progress.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

#if defined(_WIN32)
#include <io.h>
#define CPS_ISATTY _isatty
#define CPS_FILENO _fileno
#else
#include <unistd.h>
#define CPS_ISATTY isatty
#define CPS_FILENO fileno
#endif

namespace cps {
namespace {

constexpr int BAR_WIDTH = 32;
constexpr double MIN_REDRAW_SECONDS = 0.1;  // at most ten redraws per second

}  // namespace

ProgressBar::ProgressBar(std::string label, long long total)
    : label_(std::move(label)),
      total_(total),
      tty_(CPS_ISATTY(CPS_FILENO(stderr)) != 0),
      start_(std::chrono::steady_clock::now()),
      last_draw_(std::chrono::steady_clock::now() - std::chrono::seconds(1)) {}

ProgressBar::~ProgressBar() {
    if (line_open_) {
        std::fputc('\n', stderr);
        std::fflush(stderr);
    }
}

double ProgressBar::elapsed_seconds() const {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_).count();
}

void ProgressBar::update(long long current) {
    if (finished_) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double since_draw = std::chrono::duration<double>(now - last_draw_).count();
    if (since_draw < MIN_REDRAW_SECONDS && current < total_) {
        return;
    }
    last_draw_ = now;
    draw(current, false);
}

void ProgressBar::finish() {
    if (finished_) {
        return;
    }
    draw(total_, true);
    finished_ = true;
}

void ProgressBar::draw(long long current, bool final_draw) {
    const double elapsed = elapsed_seconds();
    const double fraction =
        total_ > 0 ? std::clamp(static_cast<double>(current) / static_cast<double>(total_), 0.0, 1.0)
                   : 1.0;
    const double rate = elapsed > 0.0 ? static_cast<double>(current) / elapsed : 0.0;
    const double remaining = rate > 0.0 ? static_cast<double>(total_ - current) / rate : 0.0;

    if (tty_) {
        char bar[BAR_WIDTH + 1];
        const int filled = static_cast<int>(std::lround(fraction * BAR_WIDTH));
        for (int i = 0; i < BAR_WIDTH; ++i) {
            bar[i] = i < filled ? '=' : ' ';
        }
        bar[BAR_WIDTH] = '\0';
        std::fprintf(stderr, "\r%-18s [%s] %5.1f%%  %9.1f step/s  elapsed %s  eta %s   ",
                     label_.c_str(), bar, fraction * 100.0, rate, format_duration(elapsed).c_str(),
                     format_duration(remaining).c_str());
        if (final_draw) {
            std::fprintf(stderr, "\n");
        }
        line_open_ = !final_draw;
        std::fflush(stderr);
        return;
    }

    // Not a terminal. Emit a plain line, and only at ten percent boundaries so
    // a continuous integration log stays short.
    const long long decile = total_ > 0 ? (current * 10) / total_ : 10;
    if (final_draw || decile != last_drawn_) {
        last_drawn_ = decile;
        std::fprintf(stderr, "%s: %lld/%lld (%.1f%%) %.1f step/s elapsed %s eta %s\n",
                     label_.c_str(), current, total_, fraction * 100.0, rate,
                     format_duration(elapsed).c_str(), format_duration(remaining).c_str());
        std::fflush(stderr);
    }
}

std::string format_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        seconds = 0.0;
    }
    const long long total = static_cast<long long>(std::llround(seconds));
    const long long hours = total / 3600;
    const long long minutes = (total % 3600) / 60;
    const long long secs = total % 60;
    char buffer[32];
    if (hours > 0) {
        std::snprintf(buffer, sizeof(buffer), "%lld:%02lld:%02lld", hours, minutes, secs);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%lld:%02lld", minutes, secs);
    }
    return buffer;
}

std::string format_count(long long value) {
    const bool negative = value < 0;
    std::string digits = std::to_string(negative ? -value : value);
    std::string out;
    int since_group = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (since_group == 3) {
            out.push_back(',');
            since_group = 0;
        }
        out.push_back(*it);
        ++since_group;
    }
    if (negative) {
        out.push_back('-');
    }
    std::reverse(out.begin(), out.end());
    return out;
}

}  // namespace cps
