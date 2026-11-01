// madspace_gpu_monitor <device index>: samples the SM activity of a CUDA device
// with CUPTI PM sampling, for the scheduling of the MadNIS trainings (see
// cupti_activity_monitor.hpp). It runs as a process of its own because CUPTI
// profiling slows down the kernel launches from several threads of the process
// it is initialized in. Writes "ready" or "error <reason>" first, then every
// 200 ms "<active cycles> <elapsed cycles>" (sm__cycles_active.avg and
// gr__cycles_elapsed.max summed over the samples since the previous line). The
// counters are device-wide, so they include the work of every process. Exits
// when its standard input is closed.

#include <cuda.h>
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_result.h>
#include <cupti_target.h>
#include <dlfcn.h>

#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

// The CUPTI functions used here, looked up in the library loaded at runtime
struct CuptiFunctions {
#define CUPTI_FUNCTION(name) decltype(&::name) name = nullptr;
    CUPTI_FUNCTION(cuptiGetResultString)
    CUPTI_FUNCTION(cuptiProfilerInitialize)
    CUPTI_FUNCTION(cuptiProfilerDeInitialize)
    CUPTI_FUNCTION(cuptiDeviceGetChipName)
    CUPTI_FUNCTION(cuptiProfilerHostInitialize)
    CUPTI_FUNCTION(cuptiProfilerHostDeinitialize)
    CUPTI_FUNCTION(cuptiProfilerHostConfigAddMetrics)
    CUPTI_FUNCTION(cuptiProfilerHostGetConfigImageSize)
    CUPTI_FUNCTION(cuptiProfilerHostGetConfigImage)
    CUPTI_FUNCTION(cuptiProfilerHostEvaluateToGpuValues)
    CUPTI_FUNCTION(cuptiPmSamplingGetCounterAvailability)
    CUPTI_FUNCTION(cuptiPmSamplingEnable)
    CUPTI_FUNCTION(cuptiPmSamplingDisable)
    CUPTI_FUNCTION(cuptiPmSamplingSetConfig)
    CUPTI_FUNCTION(cuptiPmSamplingStart)
    CUPTI_FUNCTION(cuptiPmSamplingStop)
    CUPTI_FUNCTION(cuptiPmSamplingDecodeData)
    CUPTI_FUNCTION(cuptiPmSamplingGetCounterDataSize)
    CUPTI_FUNCTION(cuptiPmSamplingCounterDataImageInitialize)
    CUPTI_FUNCTION(cuptiPmSamplingGetCounterDataInfo)
#undef CUPTI_FUNCTION
};

// Loads CUPTI once; throws with the reason if it is not available
const CuptiFunctions& cupti() {
    static const CuptiFunctions functions = [] {
        std::vector<std::string> candidates;
        if (const char* path = std::getenv("MADSPACE_CUPTI_LIBRARY")) {
            candidates.push_back(path);
        }
        candidates.push_back("libcupti.so.12");
        candidates.push_back("libcupti.so");
#ifdef MADSPACE_CUPTI_LIBRARY_PATH
        candidates.push_back(MADSPACE_CUPTI_LIBRARY_PATH);
#endif
        void* lib = nullptr;
        for (auto& candidate : candidates) {
            // never closed: the monitors may live until the end of the process
            if ((lib = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL))) {
                break;
            }
        }
        if (!lib) {
            throw std::runtime_error(
                "CUPTI library not found (set MADSPACE_CUPTI_LIBRARY to libcupti.so)"
            );
        }
        CuptiFunctions functions;
        auto load = [&](auto& function, const char* name) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
                dlsym(lib, name)
            );
            if (!function) {
                throw std::runtime_error(
                    std::format("CUPTI has no {} (PM sampling needs CUDA 12.6)", name)
                );
            }
        };
#define CUPTI_FUNCTION(name) load(functions.name, #name);
        CUPTI_FUNCTION(cuptiGetResultString)
        CUPTI_FUNCTION(cuptiProfilerInitialize)
        CUPTI_FUNCTION(cuptiProfilerDeInitialize)
        CUPTI_FUNCTION(cuptiDeviceGetChipName)
        CUPTI_FUNCTION(cuptiProfilerHostInitialize)
        CUPTI_FUNCTION(cuptiProfilerHostDeinitialize)
        CUPTI_FUNCTION(cuptiProfilerHostConfigAddMetrics)
        CUPTI_FUNCTION(cuptiProfilerHostGetConfigImageSize)
        CUPTI_FUNCTION(cuptiProfilerHostGetConfigImage)
        CUPTI_FUNCTION(cuptiProfilerHostEvaluateToGpuValues)
        CUPTI_FUNCTION(cuptiPmSamplingGetCounterAvailability)
        CUPTI_FUNCTION(cuptiPmSamplingEnable)
        CUPTI_FUNCTION(cuptiPmSamplingDisable)
        CUPTI_FUNCTION(cuptiPmSamplingSetConfig)
        CUPTI_FUNCTION(cuptiPmSamplingStart)
        CUPTI_FUNCTION(cuptiPmSamplingStop)
        CUPTI_FUNCTION(cuptiPmSamplingDecodeData)
        CUPTI_FUNCTION(cuptiPmSamplingGetCounterDataSize)
        CUPTI_FUNCTION(cuptiPmSamplingCounterDataImageInitialize)
        CUPTI_FUNCTION(cuptiPmSamplingGetCounterDataInfo)
#undef CUPTI_FUNCTION
        return functions;
    }();
    return functions;
}

void check(CUptiResult result, const char* call) {
    if (result == CUPTI_SUCCESS) {
        return;
    }
    if (result == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) {
        throw std::runtime_error(
            "GPU profiling is restricted to administrators "
            "(NVreg_RestrictProfilingToAdminUsers)"
        );
    }
    const char* message = nullptr;
    cupti().cuptiGetResultString(result, &message);
    throw std::runtime_error(
        std::format(
            "{} failed: {}{}",
            call,
            message ? message : "unknown error",
            // what CUPTI reports when another profiler holds the counters
            result == CUPTI_ERROR_UNKNOWN ? " (is another profiler running?)" : ""
        )
    );
}

#define CUPTI_CALL(call, params) check(cupti().call(&params), #call)

// Samples sm__cycles_active.avg (cycles in which an average SM has a warp) and
// gr__cycles_elapsed.max every sampling_interval GPU cycles
class PmSampler {
public:
    // ~7 ms at the 1.4 GHz of an A100
    static constexpr std::uint64_t sampling_interval = 10'000'000;
    static constexpr std::size_t hardware_buffer_size = 32 << 20;
    static constexpr std::size_t max_samples = 1024;

    explicit PmSampler(int device_index) {
        CUpti_Profiler_Initialize_Params params{CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        CUPTI_CALL(cuptiProfilerInitialize, params);
        _profiler_initialized = true;
        try {
            set_up(device_index);
        } catch (...) {
            tear_down();
            throw;
        }
    }

    ~PmSampler() { tear_down(); }

    // Decodes the samples collected since the previous call, adds them to the
    // sums of active and elapsed cycles
    void update(double& active_cycles, double& elapsed_cycles) {
        while (true) {
            CUpti_PmSampling_DecodeData_Params decode{
                CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE
            };
            decode.pPmSamplingObject = _sampler;
            decode.pCounterDataImage = _counter_data.data();
            decode.counterDataImageSize = _counter_data.size();
            CUPTI_CALL(cuptiPmSamplingDecodeData, decode);

            CUpti_PmSampling_GetCounterDataInfo_Params info{
                CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE
            };
            info.pCounterDataImage = _counter_data.data();
            info.counterDataImageSize = _counter_data.size();
            CUPTI_CALL(cuptiPmSamplingGetCounterDataInfo, info);
            for (std::size_t index = 0; index < info.numCompletedSamples; ++index) {
                add_sample(index, active_cycles, elapsed_cycles);
            }
            reset_counter_data();
            if (decode.decodeStopReason !=
                CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL) {
                break;
            }
        }
    }

private:
    void set_up(int device_index) {
        CUpti_Device_GetChipName_Params chip{CUpti_Device_GetChipName_Params_STRUCT_SIZE};
        chip.deviceIndex = device_index;
        CUPTI_CALL(cuptiDeviceGetChipName, chip);

        CUpti_PmSampling_GetCounterAvailability_Params availability{
            CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE
        };
        availability.deviceIndex = device_index;
        CUPTI_CALL(cuptiPmSamplingGetCounterAvailability, availability);
        _availability_image.resize(availability.counterAvailabilityImageSize);
        availability.pCounterAvailabilityImage = _availability_image.data();
        CUPTI_CALL(cuptiPmSamplingGetCounterAvailability, availability);

        CUpti_Profiler_Host_Initialize_Params host{
            CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE
        };
        host.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
        host.pChipName = chip.pChipName;
        host.pCounterAvailabilityImage = _availability_image.data();
        CUPTI_CALL(cuptiProfilerHostInitialize, host);
        _host = host.pHostObject;

        CUpti_Profiler_Host_ConfigAddMetrics_Params add_metrics{
            CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE
        };
        add_metrics.pHostObject = _host;
        add_metrics.ppMetricNames = _metrics.data();
        add_metrics.numMetrics = _metrics.size();
        CUPTI_CALL(cuptiProfilerHostConfigAddMetrics, add_metrics);
        CUpti_Profiler_Host_GetConfigImageSize_Params config_size{
            CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE
        };
        config_size.pHostObject = _host;
        CUPTI_CALL(cuptiProfilerHostGetConfigImageSize, config_size);
        _config_image.resize(config_size.configImageSize);
        CUpti_Profiler_Host_GetConfigImage_Params config{
            CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE
        };
        config.pHostObject = _host;
        config.pConfigImage = _config_image.data();
        config.configImageSize = _config_image.size();
        CUPTI_CALL(cuptiProfilerHostGetConfigImage, config);

        CUpti_PmSampling_Enable_Params enable{CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
        enable.deviceIndex = device_index;
        CUPTI_CALL(cuptiPmSamplingEnable, enable);
        _sampler = enable.pPmSamplingObject;

        CUpti_PmSampling_SetConfig_Params set_config{
            CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE
        };
        set_config.pPmSamplingObject = _sampler;
        set_config.configSize = _config_image.size();
        set_config.pConfig = _config_image.data();
        set_config.hardwareBufferSize = hardware_buffer_size;
        set_config.samplingInterval = sampling_interval;
        set_config.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL;
        CUPTI_CALL(cuptiPmSamplingSetConfig, set_config);

        CUpti_PmSampling_GetCounterDataSize_Params data_size{
            CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE
        };
        data_size.pPmSamplingObject = _sampler;
        data_size.pMetricNames = _metrics.data();
        data_size.numMetrics = _metrics.size();
        data_size.maxSamples = max_samples;
        CUPTI_CALL(cuptiPmSamplingGetCounterDataSize, data_size);
        _counter_data.resize(data_size.counterDataSize);
        reset_counter_data();

        CUpti_PmSampling_Start_Params start{CUpti_PmSampling_Start_Params_STRUCT_SIZE};
        start.pPmSamplingObject = _sampler;
        CUPTI_CALL(cuptiPmSamplingStart, start);
        _started = true;
    }

    void tear_down() {
        // without exceptions: called from the destructor
        if (_started) {
            CUpti_PmSampling_Stop_Params stop{CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
            stop.pPmSamplingObject = _sampler;
            cupti().cuptiPmSamplingStop(&stop);
            _started = false;
        }
        if (_sampler) {
            CUpti_PmSampling_Disable_Params disable{
                CUpti_PmSampling_Disable_Params_STRUCT_SIZE
            };
            disable.pPmSamplingObject = _sampler;
            cupti().cuptiPmSamplingDisable(&disable);
            _sampler = nullptr;
        }
        if (_host) {
            CUpti_Profiler_Host_Deinitialize_Params deinit{
                CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE
            };
            deinit.pHostObject = _host;
            cupti().cuptiProfilerHostDeinitialize(&deinit);
            _host = nullptr;
        }
        if (_profiler_initialized) {
            CUpti_Profiler_DeInitialize_Params deinit{
                CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE
            };
            cupti().cuptiProfilerDeInitialize(&deinit);
            _profiler_initialized = false;
        }
    }

    void reset_counter_data() {
        CUpti_PmSampling_CounterDataImage_Initialize_Params init{
            CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE
        };
        init.pPmSamplingObject = _sampler;
        init.counterDataSize = _counter_data.size();
        init.pCounterData = _counter_data.data();
        CUPTI_CALL(cuptiPmSamplingCounterDataImageInitialize, init);
    }

    void add_sample(std::size_t sample_index, double& active_cycles, double& elapsed_cycles) {
        double values[2];
        CUpti_Profiler_Host_EvaluateToGpuValues_Params evaluate{
            CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE
        };
        evaluate.pHostObject = _host;
        evaluate.pCounterDataImage = _counter_data.data();
        evaluate.counterDataImageSize = _counter_data.size();
        evaluate.ppMetricNames = _metrics.data();
        evaluate.numMetrics = _metrics.size();
        evaluate.rangeIndex = sample_index;
        evaluate.pMetricValues = values;
        CUPTI_CALL(cuptiProfilerHostEvaluateToGpuValues, evaluate);
        // the first sample spans the time before the sampling started
        if (values[1] > 100. * sampling_interval) {
            return;
        }
        active_cycles += values[0];
        elapsed_cycles += values[1];
    }

    std::vector<const char*> _metrics{"sm__cycles_active.avg", "gr__cycles_elapsed.max"};
    bool _profiler_initialized = false;
    bool _started = false;
    CUpti_Profiler_Host_Object* _host = nullptr;
    CUpti_PmSampling_Object* _sampler = nullptr;
    // CUPTI keeps pointers into these for as long as the sampling runs
    std::vector<std::uint8_t> _availability_image;
    std::vector<std::uint8_t> _config_image;
    std::vector<std::uint8_t> _counter_data;
};

} // namespace

int main(int argc, char** argv) {
    // when the parent dies, or stops reading
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    signal(SIGPIPE, SIG_IGN);
    if (argc != 2) {
        std::printf("error usage: madspace_gpu_monitor <device index>\n");
        return 1;
    }
    int device_index = std::atoi(argv[1]);
    std::unique_ptr<PmSampler> sampler;
    try {
        if (cuInit(0) != CUDA_SUCCESS) {
            throw std::runtime_error("cuInit failed");
        }
        sampler = std::make_unique<PmSampler>(device_index);
    } catch (const std::exception& e) {
        std::printf("error %s\n", e.what());
        return 1;
    }
    std::printf("ready\n");
    std::fflush(stdout);
    try {
        while (true) {
            pollfd input{STDIN_FILENO, POLLIN, 0};
            // stdin is only ever closed: any event ends the monitor
            if (poll(&input, 1, 200) != 0) {
                break;
            }
            double active_cycles = 0., elapsed_cycles = 0.;
            sampler->update(active_cycles, elapsed_cycles);
            if (std::printf("%.17g %.17g\n", active_cycles, elapsed_cycles) < 0 ||
                std::fflush(stdout) != 0) {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::printf("error %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    return 0;
}
