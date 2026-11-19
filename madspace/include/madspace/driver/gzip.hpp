#pragma once

#include <string>
#include <string_view>

namespace madspace {

/// nvCOMP gzip algorithm, 0 (fastest) to 5 (smallest output); 3 currently acts as 2.
inline constexpr int default_gzip_algorithm = 5;

/**
 * Gzip compressor for event output, obtained from @ref Device::gzip_compressor.
 *
 * Every call appends complete gzip members, so the output of successive calls
 * concatenates into one valid multi-member `.gz` file (RFC 1952), readable by
 * zcat, zlib and Python's gzip module.
 */
class GzipCompressor {
public:
    virtual ~GzipCompressor() = default;
    /// Append `data`, gzip-compressed, to `out`.
    virtual void compress(std::string_view data, std::string& out) = 0;
};

} // namespace madspace
