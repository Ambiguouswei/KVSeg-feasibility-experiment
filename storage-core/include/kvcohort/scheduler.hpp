#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace kvcohort {

// 四类任务按固定优先级取用：需求读 > 预取 > 缓存写入 > GC。
// 优先级是「取任务时的顺序」，不是线程优先级。
enum class TaskClass : std::uint8_t { demand = 0, prefetch = 1, write = 2, gc = 3 };
constexpr std::size_t kTaskClasses = 4;

struct SchedulerOptions {
  std::size_t workers = 4;
  // 同时按任务数与字节数限制排队量，避免大对象绕过请求数上限。
  std::size_t max_queued_tasks = 1024;
  std::uint64_t max_queued_bytes = 256ull * 1024 * 1024;
  // GC 保底：队首等待超过该毫秒数就插到最前，防止高频需求读把回收饿死。
  // 设为 0 表示「GC 永远优先」，仅用于测试。
  std::uint64_t gc_aging_ms = 500;
};

struct SchedulerStats {
  std::uint64_t submitted = 0;
  std::uint64_t completed = 0;
  std::uint64_t rejected = 0;   // 因排队上限被拒
  std::uint64_t aged = 0;       // 因保底被提前处理的 GC 任务数
  std::uint64_t queued_tasks = 0;
  std::uint64_t queued_bytes = 0;
  std::uint64_t active = 0;
  std::uint64_t submitted_by_class[kTaskClasses] = {0, 0, 0, 0};
  std::uint64_t executed_by_class[kTaskClasses] = {0, 0, 0, 0};
};

// 有界优先级执行器。任务体不应阻塞太久（当前实现里它们只做本地文件 I/O）。
class Scheduler {
 public:
  explicit Scheduler(SchedulerOptions options = {});
  ~Scheduler();
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  // weight_bytes 用于字节级背压；返回 false 表示被上限拒绝，调用方应回 busy。
  bool TrySubmit(TaskClass cls, std::uint64_t weight_bytes, std::function<void()> task);
  void WaitIdle();
  SchedulerStats GetStats() const;

 private:
  struct Item {
    TaskClass cls;
    std::uint64_t weight;
    std::function<void()> task;
    std::chrono::steady_clock::time_point enqueued;
  };

  std::optional<Item> TakeLocked();
  std::size_t QueuedTasksLocked() const;
  void WorkerLoop();

  SchedulerOptions options_;  // 构造时归一化（workers=0 会被抬到 1），因此不能是 const
  mutable std::mutex mutex_;
  std::condition_variable ready_cv_;
  std::condition_variable idle_cv_;
  std::deque<Item> queues_[kTaskClasses];
  std::uint64_t queued_bytes_ = 0;
  std::size_t active_ = 0;
  bool stopping_ = false;
  SchedulerStats stats_;
  std::vector<std::thread> workers_;
};

}  // namespace kvcohort
