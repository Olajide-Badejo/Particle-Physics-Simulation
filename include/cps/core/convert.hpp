#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace cps {

// The input cannot be converted as asked: it is missing, it is neither kind of
// file, or it needs a --config it did not get. The caller exits 2 on this, and 1
// on any other error, such as an output that cannot be written.
class ConvertInputError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Picks the conversion by what the input is, and returns a phrase saying which it
// did. A configuration file is written in canonical schema 2, the same bytes
// --print-config gives. A CPS1 snapshot is written as CPS2, fingerprinted from the
// configuration at config_path, which is required for it and refused otherwise.
// Its particle count must match that configuration, its ids are assigned in file
// order, and its time stays NaN because CPS1 has none, which a resume treats the
// same way as it treats CPS1 itself.
std::string convert_file(const std::filesystem::path& input, const std::filesystem::path& output,
                         const std::string& config_path);

}  // namespace cps
