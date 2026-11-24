#include "kvcohort/scheduler.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace kvcohort;

std::mutex g_report_mutex;
int g_failed = 0;
int g_total = 0;

void Require(bool condition, const char* message) {
  ++g_total;
  if (condition) return;
  std::lock_guard<std::mutex> lock(g_report_mutex);
  ++g_failed;
  std::fprintf(stderr, "FAIL %s\n", message);
}

// 让唯一的工作线程先卡住，从而确定性地把后续任务积压在队列里。
class Gate {
 public:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return open_; });
  }
  void Open() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      open_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool open_ = false;
};

struct Recorder {
  std::mutex mutex;
  std::vector<TaskClass> order;
  void Add(TaskClass cls) {
    std::lock_guard<std::mutex> lock(mutex);
    order.push_back(cls);
  }
  std::vector<TaskClass> Snapshot() {
    std::lock_guard<std::mutex> lock(mutex);
    return order;
  }
};

SchedulerOptions SingleWorker(std::uint64_t gc_aging_ms) {
  SchedulerOptions options;
  options.workers = 1;
  options.max_queued_tasks = 64;
  options.max_queued_bytes = 1ull << 20;
  options.gc_aging_ms = gc_aging_ms;
  return options;
}

// 固定优先级：demand > prefetch > write > gc。
void PriorityOrder() {
  Scheduler scheduler(SingleWorker(1000000));  // 关闭保底，纯看优先级
  Gate gate;
  std::atomic<bool> started{false};
  Require(scheduler.TrySubmit(TaskClass::demand, 0, [&] {
            started.store(true);
            gate.Wait();
          }),
          "blocker submitted");
  while (!started.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));

  Recorder recorder;
  scheduler.TrySubmit(TaskClass::gc, 0, [&] { recorder.Add(TaskClass::gc); });
  scheduler.TrySubmit(TaskClass::write, 0, [&] { recorder.Add(TaskClass::write); });
  scheduler.TrySubmit(TaskClass::prefetch, 0, [&] { recorder.Add(TaskClass::prefetch); });
  scheduler.TrySubmit(TaskClass::demand, 0, [&] { recorder.Add(TaskClass::demand); });

  gate.Open();
  scheduler.WaitIdle();

  const auto order = recorder.Snapshot();
  Require(order.size() == 4, "all four tasks ran");
  if (order.size() == 4) {
    Require(order[0] == TaskClass::demand, "demand runs first");
    Require(order[1] == TaskClass::prefetch, "prefetch second");
    Require(order[2] == TaskClass::write, "write third");
    Require(order[3] == TaskClass::gc, "gc last when aging is disabled");
  }
}

// GC 保底：等待超阈值后被提前处理，避免被需求读饿死。
void GcAging() {
  Scheduler scheduler(SingleWorker(0));  // 立即到期
  Gate gate;
  std::atomic<bool> started{false};
  Require(scheduler.TrySubmit(TaskClass::demand, 0, [&] {
            started.store(true);
            gate.Wait();
          }),
          "blocker submitted");
  while (!started.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));

  Recorder recorder;
  scheduler.TrySubmit(TaskClass::gc, 0, [&] { recorder.Add(TaskClass::gc); });
  scheduler.TrySubmit(TaskClass::demand, 0, [&] { recorder.Add(TaskClass::demand); });

  gate.Open();
  scheduler.WaitIdle();

  const auto order = recorder.Snapshot();
  Require(order.size() == 2, "both tasks ran");
  if (order.size() == 2) {
    Require(order[0] == TaskClass::gc, "aged gc overtakes demand");
    Require(order[1] == TaskClass::demand, "demand runs after aged gc");
  }
  Require(scheduler.GetStats().aged >= 1, "aging counter incremented");
}

// 任务数上限：超出后拒绝而不是无限排队。
void TaskBound() {
  SchedulerOptions options = SingleWorker(1000000);
  options.max_queued_tasks = 2;
  Scheduler scheduler(options);
  Gate gate;
  std::atomic<bool> started{false};
  Require(scheduler.TrySubmit(TaskClass::demand, 0, [&] {
            started.store(true);
            gate.Wait();
          }),
          "blocker submitted");
  while (!started.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));

  Require(scheduler.TrySubmit(TaskClass::write, 0, [] {}), "1st queued task accepted");
  Require(scheduler.TrySubmit(TaskClass::write, 0, [] {}), "2nd queued task accepted");
  Require(!scheduler.TrySubmit(TaskClass::write, 0, [] {}), "3rd queued task rejected");
  Require(scheduler.GetStats().rejected >= 1, "rejected counter incremented");

  gate.Open();
  scheduler.WaitIdle();
}

// 字节上限：大对象不能绕过任务数上限。
void ByteBound() {
  SchedulerOptions options = SingleWorker(1000000);
  options.max_queued_bytes = 100;
  Scheduler scheduler(options);
  Gate gate;
  std::atomic<bool> started{false};
  Require(scheduler.TrySubmit(TaskClass::demand, 0, [&] {
            started.store(true);
            gate.Wait();
          }),
          "blocker submitted");
  while (!started.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));

  Require(scheduler.TrySubmit(TaskClass::write, 60, [] {}), "60 bytes accepted");
  Require(!scheduler.TrySubmit(TaskClass::write, 60, [] {}), "second 60 bytes rejected");
  gate.Open();
  scheduler.WaitIdle();
}

// 多线程下全部任务都应完成，且计数自洽。
void AllTasksComplete() {
  SchedulerOptions options;
  options.workers = 4;
  Scheduler scheduler(options);
  std::atomic<int> ran{0};
  constexpr int kTasks = 200;
  for (int i = 0; i < kTasks; ++i)
    Require(scheduler.TrySubmit(static_cast<TaskClass>(i % 4), 0, [&] { ran.fetch_add(1); }),
            "task accepted");
  scheduler.WaitIdle();
  Require(ran.load() == kTasks, "every submitted task ran");
  const auto stats = scheduler.GetStats();
  Require(stats.completed == static_cast<std::uint64_t>(kTasks), "completed matches submissions");
  Require(stats.queued_tasks == 0, "queue drained");
  std::uint64_t executed = 0;
  for (std::size_t i = 0; i < kTaskClasses; ++i) executed += stats.executed_by_class[i];
  Require(executed == stats.completed, "per-class counters sum to completed");
}

}  // namespace

int main() {
  PriorityOrder();
  GcAging();
  TaskBound();
  ByteBound();
  AllTasksComplete();
  std::printf("scheduler contracts: %d passed, %d failed\n", g_total - g_failed, g_failed);
  return g_failed == 0 ? 0 : 1;
}
