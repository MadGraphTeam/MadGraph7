#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace madspace {

class ThreadPool {
public:
    using JobFunc = std::function<std::optional<std::size_t>()>;
    ThreadPool(int thread_count = -1);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    void set_thread_count(int new_count);
    std::size_t thread_count() const { return _thread_count; }
    void submit(JobFunc job);
    void submit(std::vector<JobFunc>& jobs);
    std::optional<std::size_t> wait();
    std::vector<std::size_t> wait_multiple();
    std::size_t add_listener(std::function<void(std::size_t)> listener);
    void remove_listener(std::size_t id);

    static std::size_t thread_index() { return _thread_index; }

private:
    static inline thread_local std::size_t _thread_index = 0;
    static const std::size_t QUEUE_SIZE_PER_THREAD = 16384;

    void thread_loop(std::size_t index);
    bool fill_done_cache();

    std::mutex _mutex;
    std::condition_variable _cv_run, _cv_done;
    std::size_t _thread_count;
    std::vector<std::thread> _threads;
    std::deque<JobFunc> _job_queue;
    std::deque<std::size_t> _done_queue;
    std::vector<std::size_t> _done_buffer;
    std::size_t _busy_threads = 0;
    std::size_t _listener_id = 0;
    std::unordered_map<std::size_t, std::function<void(std::size_t)>> _listeners;
};

class ResultQueue {
public:
    void push(std::size_t result);
    std::size_t wait();
    std::vector<std::size_t> wait_multiple();

private:
    void fill_done_cache();

    std::mutex _mutex;
    std::condition_variable _cv;
    std::deque<std::size_t> _queue;
    std::vector<std::size_t> _buffer;
};

template <typename T>
class ThreadResource {
public:
    ThreadResource() = default;
    ThreadResource(
        ThreadPool& pool,
        std::function<T()> constructor,
        std::optional<std::function<void(T&)>> destructor = std::nullopt
    ) :
        _pool(&pool),
        _constructor(std::move(constructor)),
        _destructor(destructor),
        _listener_id(pool.add_listener([this](std::size_t thread_count) {
            while (_resources.size() < thread_count) {
                _resources.emplace_back();
            }
        })) {
        for (std::size_t i = 0; i == 0 || i < pool.thread_count(); ++i) {
            _resources.emplace_back();
        }
        get();
    }
    ~ThreadResource() {
        reset();
    }
    ThreadResource(ThreadResource&& other) noexcept :
        _pool(std::move(other._pool)),
        _resources(std::move(other._resources)),
        _constructor(std::move(other._constructor)),
        _listener_id(std::move(other._listener_id)),
        _destructor(std::move(other._destructor)) {
        other._pool = nullptr;
    }

    ThreadResource& operator=(ThreadResource&& other) noexcept {
        reset();
        _pool = std::move(other._pool);
        _resources = std::move(other._resources);
        _constructor = std::move(other._constructor);
        _listener_id = std::move(other._listener_id);
        _destructor = std::move(other._destructor);
        other._pool = nullptr;
        return *this;
    }
    ThreadResource(const ThreadResource&) = delete;
    ThreadResource& operator=(const ThreadResource&) = delete;
    T& get() const {
        auto& [flag, item] = _resources.at(ThreadPool::thread_index());
        std::call_once(flag, [&] {
            std::unique_lock<std::mutex> lock(construction_mutex());
            item.emplace(_constructor());
        });
        return *item;
    }
    void reset() {
        if (_pool) {
            if (_destructor) {
                for (auto& [flag, item] : _resources) {
                    if (item) {
                        _destructor.value()(*item);
                    }
                }
            }
            _pool->remove_listener(_listener_id);
        }
    }

private:
    static std::mutex& construction_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    ThreadPool* _pool = nullptr;
    mutable std::deque<std::pair<std::once_flag, std::optional<T>>> _resources;
    std::function<T()> _constructor;
    std::size_t _listener_id;
    std::optional<std::function<void(T&)>> _destructor;
};

inline ThreadPool& default_thread_pool() {
    static ThreadPool instance;
    return instance;
}

} // namespace madspace
