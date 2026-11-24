#include "kvcohort/scheduler.hpp"

#include <algorithm>
#include <utility>

namespace kvcohort {

Scheduler::Scheduler(SchedulerOptions options) : options_(std::move(options)) {
  if (options_.workers == 0) options_.workers = 1;
  workers_.reserve(options_.workers);
  for (std::size_t i = 0; i < options_.workers; ++i)
    workers_.emplace_back([this] { WorkerLoop(); });
}

Scheduler::~Scheduler() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  ready_cv_.notify_all();
  for (auto& worker : workers_)
    if (worker.joinable()) worker.join();
}

std::size_t Scheduler::QueuedTasksLocked() const {
  std::size_t total = 0;
  for (const auto& queue : queues_) total += queue.size();
  return total;
}

bool Scheduler::TrySubmit(TaskClass cls, std::uint64_t weight_bytes, std::function<void()> task) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_) return false;
  if (QueuedTasksLocked() >= options_.max_queued_tasks ||
      queued_bytes_ + weight_bytes > options_.max_queued_bytes) {
    ++stats_.rejected;
    return false;
  }
  const auto index = static_cast<std::size_t>(cls);
  queues_[index].push_back(Item{cls, weight_bytes, std::move(task), std::chrono::steady_clock::now()});
  queued_bytes_ += weight_bytes;
  ++stats_.submitted;
  ++stats_.submitted_by_class[index];
  ready_cv_.notify_one();
  return true;
}

std::optional<Scheduler::Item> Scheduler::TakeLocked() {
  const auto now = std::chrono::steady_clock::now();
  const auto gc_index = static_cast<std::size_t>(TaskClass::gc);

  // 保底：GC 队首等待超过阈值就插队，否则持续的高频需求读会把它饿死。
  if (!queues_[gc_index].empty()) {
    const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - queues_[gc_index].front().enqueued)
                            .count();
    if (waited >= static_cast<long long>(options_.gc_aging_ms)) {
      ++stats_.aged;
      Item item = std::move(queues_[gc_index].front());
      queues_[gc_index].pop_front();
      queued_bytes_ -= item.weight;
      return item;
    }
  }
  // 固定优先级：demand > prefetch > write > gc。
  for (auto& queue : queues_) {
    if (queue.empty()) continue;
    Item item = std::move(queue.front());
    queue.pop_front();
    queued_bytes_ -= item.weight;
    return item;
  }
  return std::nullopt;
}

void Scheduler::WorkerLoop() {
  for (;;) {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_cv_.wait(lock, [this] { return stopping_ || QueuedTasksLocked() > 0; });
    if (stopping_ && QueuedTasksLocked() == 0) return;

    auto item = TakeLocked();
    if (!item) continue;
    ++active_;

    lock.unlock();
    try {
      item->task();
    } catch (...) {
      // 任务体不应抛异常；真抛了也不能让工作线程退出，否则队列会被永久卡住。
    }
    lock.lock();

    --active_;
    ++stats_.completed;
    ++stats_.executed_by_class[static_cast<std::size_t>(item->cls)];
    if (active_ == 0 && QueuedTasksLocked() == 0) idle_cv_.notify_all();
  }
}

void Scheduler::WaitIdle() {
  std::unique_lock<std::mutex> lock(mutex_);
  idle_cv_.wait(lock, [this] { return active_ == 0 && QueuedTasksLocked() == 0; });
}

SchedulerStats Scheduler::GetStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  SchedulerStats snapshot = stats_;
  snapshot.queued_tasks = QueuedTasksLocked();
  snapshot.queued_bytes = queued_bytes_;
  snapshot.active = active_;
  return snapshot;
}

}  // namespace kvcohort
