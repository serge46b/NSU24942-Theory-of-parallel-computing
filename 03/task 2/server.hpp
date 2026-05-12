#pragma once
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

struct TaskDesc {
    enum Kind { Sin, Sqrt, Pow } kind;
    double x{};
    double y{}; // степень для Pow
};

template <typename T>
class TaskServer {
public:
    explicit TaskServer(std::size_t pool_size = std::thread::hardware_concurrency())
        : pool_size_(pool_size ? pool_size : 1)
    {}

    void start()
    {
        if (running_.exchange(true))
            return;
        stop_requested_ = false;
        workers_.reserve(pool_size_);
        for (std::size_t i = 0; i < pool_size_; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    void stop()
    {
        if (!running_)
            return;
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            stop_requested_ = true;
        }
        q_cv_.notify_all();
        for (auto& w : workers_)
            if (w.joinable())
                w.join();
        workers_.clear();
        running_ = false;
        stop_requested_ = false;
    }

    ~TaskServer()
    {
        if (running_)
            stop();
    }

    std::size_t add_task(TaskDesc t)
    {
        if (!running_)
            throw std::runtime_error("TaskServer: start() before add_task()");
        const std::size_t id = ++next_id_;
        {
            std::lock_guard<std::mutex> lk(res_mu_);
            results_[id] = std::nullopt;
        }
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            task_queue_.emplace(id, t);
        }
        q_cv_.notify_one();
        return id;
    }

    T request_result(std::size_t id)
    {
        std::unique_lock<std::mutex> lk(res_mu_);
        res_cv_.wait(lk, [&] {
            auto it = results_.find(id);
            return it != results_.end() && it->second.has_value();
        });
        return *results_[id];
    }

private:
    void worker_loop()
    {
        for (;;) {
            std::pair<std::size_t, TaskDesc> job{};
            {
                std::unique_lock<std::mutex> lk(q_mu_);
                q_cv_.wait(lk, [&] { return stop_requested_ || !task_queue_.empty(); });
                if (stop_requested_ && task_queue_.empty())
                    return;
                if (task_queue_.empty())
                    continue;
                job = task_queue_.front();
                task_queue_.pop();
            }
            const auto [id, td] = job;
            T v{};
            switch (td.kind) {
            case TaskDesc::Sin:
                v = static_cast<T>(std::sin(td.x));
                break;
            case TaskDesc::Sqrt:
                v = static_cast<T>(std::sqrt(td.x));
                break;
            case TaskDesc::Pow:
                v = static_cast<T>(std::pow(td.x, td.y));
                break;
            }
            {
                std::lock_guard<std::mutex> lk(res_mu_);
                results_[id] = v;
            }
            res_cv_.notify_all();
        }
    }

    std::atomic<std::size_t> next_id_{0};
    std::atomic<bool> running_{false};
    bool stop_requested_{false};
    std::size_t pool_size_;
    std::vector<std::thread> workers_;

    std::mutex q_mu_;
    std::condition_variable q_cv_;
    std::queue<std::pair<std::size_t, TaskDesc>> task_queue_;

    std::mutex res_mu_;
    std::condition_variable res_cv_;
    std::unordered_map<std::size_t, std::optional<T>> results_;
};
