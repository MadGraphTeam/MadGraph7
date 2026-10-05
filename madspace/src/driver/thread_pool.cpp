#include "madspace/driver/thread_pool.hpp"

#include "madspace/util.hpp"

using namespace madspace;

ThreadPool::ThreadPool(int thread_count) { set_thread_count(thread_count); }

ThreadPool::~ThreadPool() { set_thread_count(0); }

void ThreadPool::set_thread_count(int new_count) {
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _thread_count = new_count < 0 ? std::thread::hardware_concurrency() : new_count;
    }

    if (_threads.size() < _thread_count) {
        for (int i = _threads.size(); i < _thread_count; ++i) {
            _threads.emplace_back(&ThreadPool::thread_loop, this, i);
        }
    } else if (_threads.size() > _thread_count) {
        _cv_run.notify_all();
        std::for_each(
            _threads.begin() + _thread_count, _threads.end(), [](auto& thread) {
                thread.join();
            }
        );
        _threads.erase(_threads.begin() + _thread_count, _threads.end());
    }

    for (auto& [id, listener] : _listeners) {
        listener(_thread_count);
    }
}

void ThreadPool::submit(JobFunc job) {
    // Spin until there is space in the queue. The queue should be sufficiently large
    // such that this never happens.
    std::unique_lock<std::mutex> lock(_mutex);
    _job_queue.push_back(job);
    if (!_done_queue.empty()) {
        _done_buffer.insert(
            _done_buffer.begin(), _done_queue.rbegin(), _done_queue.rend()
        );
        _done_queue.clear();
    }
    lock.unlock();
    _cv_run.notify_one();
}

void ThreadPool::submit(std::vector<JobFunc>& jobs) {
    if (jobs.empty()) {
        return;
    }
    std::unique_lock<std::mutex> lock(_mutex);
    for (auto& job : jobs) {
        _job_queue.push_back(std::move(job));
    }
    if (!_done_queue.empty()) {
        _done_buffer.insert(
            _done_buffer.begin(), _done_queue.rbegin(), _done_queue.rend()
        );
        _done_queue.clear();
    }
    lock.unlock();
    _cv_run.notify_all();
}

void ThreadPool::rethrow_job_exception(std::unique_lock<std::mutex>& lock) {
    _job_queue.clear();
    _cv_done.wait(lock, [&] { return _busy_threads == 0; });
    auto exception = _exception;
    _exception = nullptr;
    _done_queue.clear();
    _done_buffer.clear();
    std::rethrow_exception(exception);
}

bool ThreadPool::fill_done_cache() {
    if (!_done_buffer.empty()) {
        return true;
    }

    std::unique_lock<std::mutex> lock(_mutex);
    if (_exception) {
        rethrow_job_exception(lock);
    }
    if (_done_queue.empty()) {
        if (_job_queue.empty() && _busy_threads == 0) {
            return false;
        }
        _cv_done.wait(lock, [&] { return !_done_queue.empty() || _exception; });
        if (_exception) {
            rethrow_job_exception(lock);
        }
    }
    _done_buffer.insert(_done_buffer.begin(), _done_queue.rbegin(), _done_queue.rend());
    _done_queue.clear();
    return true;
}

std::optional<std::size_t> ThreadPool::wait() {
    if (!fill_done_cache()) {
        return std::nullopt;
    }
    std::size_t result = _done_buffer.back();
    _done_buffer.pop_back();
    return result;
}

std::vector<std::size_t> ThreadPool::wait_multiple() {
    if (!fill_done_cache()) {
        return {};
    }
    std::vector<std::size_t> ret(_done_buffer.rbegin(), _done_buffer.rend());
    _done_buffer.clear();
    return ret;
}

std::size_t ThreadPool::add_listener(std::function<void(std::size_t)> listener) {
    _listeners[_listener_id] = listener;
    return _listener_id++;
}

void ThreadPool::remove_listener(std::size_t id) {
    if (auto search = _listeners.find(id); search != _listeners.end()) {
        _listeners.erase(search);
    } else {
        throw std::invalid_argument("Listener id not found");
    }
}

void ThreadPool::thread_loop(std::size_t index) {
    _thread_index = index;
    std::unique_lock<std::mutex> lock(_mutex);
    while (true) {
        _cv_run.wait(lock, [&] {
            return !_job_queue.empty() || index >= _thread_count;
        });
        if (index >= _thread_count) {
            return;
        }
        auto job = _job_queue.front();
        _job_queue.pop_front();
        ++_busy_threads;
        lock.unlock();
        std::optional<std::size_t> result;
        try {
            result = job();
        } catch (...) {
            // A job must never let an exception escape: it would unwind out of
            // the thread function and terminate the process. Hand the first one
            // to whoever is waiting, and drop the jobs still queued -- the
            // waiter is about to unwind and they capture its locals.
            lock.lock();
            if (!_exception) {
                _exception = std::current_exception();
            }
            _job_queue.clear();
            --_busy_threads;
            _cv_done.notify_all();
            continue;
        }
        lock.lock();
        --_busy_threads;
        if (result) {
            _done_queue.push_back(*result);
        }
        // notify_all, not notify_one: rethrow_job_exception() waits here for
        // _busy_threads to drain, and a job returning nullopt must wake it too.
        _cv_done.notify_all();
    }
}

void ResultQueue::push(std::size_t result) {
    std::unique_lock<std::mutex> lock(_mutex);
    _queue.push_back(result);
    _cv.notify_one();
}

std::size_t ResultQueue::wait() {
    fill_done_cache();
    std::size_t result = _buffer.back();
    _buffer.pop_back();
    return result;
}

std::vector<std::size_t> ResultQueue::wait_multiple() {
    fill_done_cache();
    std::vector<std::size_t> ret(_buffer.rbegin(), _buffer.rend());
    _buffer.clear();
    return ret;
}

void ResultQueue::fill_done_cache() {
    if (!_buffer.empty()) {
        return;
    }
    std::unique_lock<std::mutex> lock(_mutex);
    if (_queue.empty()) {
        _cv.wait(lock, [&] { return !_queue.empty(); });
    }
    _buffer.insert(_buffer.begin(), _queue.rbegin(), _queue.rend());
    _queue.clear();
}
