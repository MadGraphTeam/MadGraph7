#include "device.hpp"

#include <nvcomp/gzip.h>

#include <algorithm>
#include <cstring>
#include <vector>

using namespace madspace;
using namespace madspace::gpu;

namespace {

// 64 KiB is the largest chunk nvCOMP's gzip compressor takes and its fastest.
constexpr std::size_t chunk_bytes = 1 << 16;

void check_nvcomp(nvcompStatus_t status, const char* what) {
    if (status != nvcompSuccess) {
        throw std::runtime_error(
            std::format(
                "nvCOMP {} failed with status {}", what, static_cast<int>(status)
            )
        );
    }
}

std::size_t round_up(std::size_t value, std::size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

__global__ void kernel_chunk_tables(
    std::size_t chunk_count,
    std::size_t data_bytes,
    std::size_t out_stride,
    const char* data,
    char* out,
    const void** chunk_ptrs,
    std::size_t* chunk_sizes,
    void** out_ptrs
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= chunk_count) {
        return;
    }
    std::size_t begin = i * chunk_bytes;
    chunk_ptrs[i] = data + begin;
    chunk_sizes[i] = min(chunk_bytes, data_bytes - begin);
    out_ptrs[i] = out + i * out_stride;
}

// One block per chunk, gathering the compressed chunks into one contiguous stream.
__global__ void kernel_pack_chunks(
    void* const* out_ptrs,
    const std::size_t* offsets,
    const std::size_t* sizes,
    char* packed
) {
    const char* source = static_cast<const char*>(out_ptrs[blockIdx.x]);
    char* target = packed + offsets[blockIdx.x];
    for (std::size_t i = threadIdx.x; i < sizes[blockIdx.x]; i += blockDim.x) {
        target[i] = source[i];
    }
}

template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() {
        if (_ptr != nullptr) {
            ignore_error(cudaFree(_ptr));
        }
    }
    void resize(std::size_t count) {
        if (_ptr != nullptr) {
            check_error(cudaFree(_ptr));
        }
        check_error(cudaMalloc(&_ptr, std::max<std::size_t>(count, 1) * sizeof(T)));
    }
    T* get() const { return _ptr; }

private:
    T* _ptr = nullptr;
};

/// Compresses through a pinned staging buffer: one H2D copy of the text and one
/// D2H copy of the packed gzip members per call.
class NvcompGzipCompressor : public GzipCompressor {
public:
    NvcompGzipCompressor(const GpuDevice& device, int algorithm) : _device(device) {
        if (algorithm < 0 || algorithm > 5) {
            throw std::invalid_argument(
                std::format("nvCOMP gzip algorithm must be 0 to 5, got {}", algorithm)
            );
        }
        _options.algorithm = algorithm;
        nvcompAlignmentRequirements_t alignment;
        check_nvcomp(
            nvcompBatchedGzipCompressGetRequiredAlignments(_options, &alignment),
            "alignment query"
        );
        std::size_t max_chunk_out;
        check_nvcomp(
            nvcompBatchedGzipCompressGetMaxOutputChunkSize(
                chunk_bytes, _options, &max_chunk_out
            ),
            "output size query"
        );
        _out_stride =
            round_up(max_chunk_out, std::max<std::size_t>(alignment.output, 16));
    }

    NvcompGzipCompressor(const NvcompGzipCompressor&) = delete;
    NvcompGzipCompressor& operator=(const NvcompGzipCompressor&) = delete;

    ~NvcompGzipCompressor() override {
        if (_host_data != nullptr) {
            ignore_error(cudaFreeHost(_host_data));
        }
        if (_stream != nullptr) {
            ignore_error(cudaStreamDestroy(_stream));
        }
    }

    void compress(std::string_view data, std::string& out) override {
        if (data.empty()) {
            return;
        }
        _device.activate();
        reserve(data.size());
        std::size_t chunk_count = (data.size() + chunk_bytes - 1) / chunk_bytes;
        std::memcpy(_host_data, data.data(), data.size());
        check_error(cudaMemcpyAsync(
            _data.get(), _host_data, data.size(), cudaMemcpyHostToDevice, _stream
        ));
        unsigned table_blocks = static_cast<unsigned>((chunk_count + 255) / 256);
        kernel_chunk_tables<<<table_blocks, 256, 0, _stream>>>(
            chunk_count,
            data.size(),
            _out_stride,
            _data.get(),
            _chunks_out.get(),
            _chunk_ptrs.get(),
            _chunk_sizes.get(),
            _out_ptrs.get()
        );
        check_error();
        check_nvcomp(
            nvcompBatchedGzipCompressAsync(
                _chunk_ptrs.get(),
                _chunk_sizes.get(),
                chunk_bytes,
                chunk_count,
                _temp.get(),
                _temp_bytes,
                _out_ptrs.get(),
                _out_sizes.get(),
                _options,
                _statuses.get(),
                _stream
            ),
            "compression launch"
        );
        // The extra zero size makes the exclusive scan end on the total.
        check_error(cudaMemsetAsync(
            _out_sizes.get() + chunk_count, 0, sizeof(std::size_t), _stream
        ));
        check_error(
            cub::DeviceScan::ExclusiveSum(
                _scan_temp.get(),
                _scan_temp_bytes,
                _out_sizes.get(),
                _offsets.get(),
                chunk_count + 1,
                _stream
            )
        );
        kernel_pack_chunks<<<static_cast<unsigned>(chunk_count), 256, 0, _stream>>>(
            _out_ptrs.get(), _offsets.get(), _out_sizes.get(), _packed.get()
        );
        check_error();
        std::size_t packed_bytes;
        check_error(cudaMemcpyAsync(
            &packed_bytes,
            _offsets.get() + chunk_count,
            sizeof(std::size_t),
            cudaMemcpyDeviceToHost,
            _stream
        ));
        check_error(cudaMemcpyAsync(
            _host_statuses.data(),
            _statuses.get(),
            chunk_count * sizeof(nvcompStatus_t),
            cudaMemcpyDeviceToHost,
            _stream
        ));
        check_error(cudaStreamSynchronize(_stream));
        for (std::size_t i = 0; i < chunk_count; ++i) {
            check_nvcomp(_host_statuses[i], "chunk compression");
        }
        std::size_t out_offset = out.size();
        out.resize(out_offset + packed_bytes);
        check_error(cudaMemcpy(
            out.data() + out_offset, _packed.get(), packed_bytes, cudaMemcpyDeviceToHost
        ));
    }

private:
    void reserve(std::size_t data_bytes) {
        if (data_bytes <= _capacity) {
            return;
        }
        if (_stream == nullptr) {
            check_error(cudaStreamCreate(&_stream));
        }
        _capacity = round_up(data_bytes, chunk_bytes);
        std::size_t chunk_count = _capacity / chunk_bytes;
        if (_host_data != nullptr) {
            check_error(cudaFreeHost(_host_data));
        }
        check_error(cudaMallocHost(&_host_data, _capacity));
        _host_statuses.resize(chunk_count);
        _data.resize(_capacity);
        _chunks_out.resize(chunk_count * _out_stride);
        _packed.resize(chunk_count * _out_stride);
        _chunk_ptrs.resize(chunk_count);
        _chunk_sizes.resize(chunk_count);
        _out_ptrs.resize(chunk_count);
        _out_sizes.resize(chunk_count + 1);
        _offsets.resize(chunk_count + 1);
        _statuses.resize(chunk_count);
        check_nvcomp(
            nvcompBatchedGzipCompressGetTempSizeAsync(
                chunk_count, chunk_bytes, _options, &_temp_bytes, _capacity
            ),
            "temporary size query"
        );
        _temp.resize(_temp_bytes);
        _scan_temp_bytes = 0;
        check_error(
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                _scan_temp_bytes,
                _out_sizes.get(),
                _offsets.get(),
                chunk_count + 1,
                _stream
            )
        );
        _scan_temp.resize(_scan_temp_bytes);
    }

    const GpuDevice& _device;
    nvcompBatchedGzipCompressOpts_t _options = nvcompBatchedGzipCompressDefaultOpts;
    std::size_t _out_stride;
    std::size_t _capacity = 0;
    cudaStream_t _stream = nullptr;
    char* _host_data = nullptr;
    std::vector<nvcompStatus_t> _host_statuses;
    DeviceArray<char> _data;
    DeviceArray<char> _chunks_out;
    DeviceArray<char> _packed;
    DeviceArray<const void*> _chunk_ptrs;
    DeviceArray<std::size_t> _chunk_sizes;
    DeviceArray<void*> _out_ptrs;
    DeviceArray<std::size_t> _out_sizes;
    DeviceArray<std::size_t> _offsets;
    DeviceArray<nvcompStatus_t> _statuses;
    DeviceArray<char> _temp;
    std::size_t _temp_bytes = 0;
    DeviceArray<char> _scan_temp;
    std::size_t _scan_temp_bytes = 0;
};

} // namespace

std::unique_ptr<GzipCompressor> GpuDevice::gzip_compressor(int algorithm) const {
    return std::make_unique<NvcompGzipCompressor>(*this, algorithm);
}
