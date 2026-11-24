#include "kvcohort/protocol.hpp"
#include "kvcohort/scheduler.hpp"
#include "kvcohort/store.hpp"

#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace kvcohort;

std::atomic<bool> g_stop{false};
constexpr int kPollTimeoutMs = 200;

[[noreturn]] void IoFail(const std::string& what) {
  throw Error(ErrorCode::io, what + ": " + std::strerror(errno));
}

proto::Status StatusFor(ErrorCode code) {
  switch (code) {
    case ErrorCode::invalid: return proto::Status::invalid;
    case ErrorCode::missing: return proto::Status::missing;
    case ErrorCode::conflict: return proto::Status::conflict;
    case ErrorCode::capacity: return proto::Status::capacity;
    case ErrorCode::corruption: return proto::Status::corruption;
    case ErrorCode::busy: return proto::Status::busy;
    case ErrorCode::io: return proto::Status::io;
    case ErrorCode::declined: return proto::Status::declined;
  }
  return proto::Status::io;
}

BlockKey ReadKey(proto::Reader& in) {
  BlockKey key;
  key.space = std::string(in.str());
  key.digest = std::string(in.str());
  key.parent = std::string(in.str());
  return key;
}

std::string StatsJson(const Stats& s, const SchedulerStats& q) {
  std::string out = "{";
  const auto add = [&out](const char* name, std::uint64_t value, bool first) {
    if (!first) out += ",";
    out += "\"";
    out += name;
    out += "\":";
    out += std::to_string(value);
  };
  add("objects", s.objects, true);
  add("live_payload_bytes", s.live_payload_bytes, false);
  add("segment_bytes", s.segment_bytes, false);
  add("allocated_bytes", s.allocated_bytes, false);
  add("segments", s.segments, false);
  add("active_segments", s.active_segments, false);
  add("gc_moved_bytes", s.gc_moved_bytes, false);
  add("demand_read_calls", s.demand_read_calls, false);
  add("demand_read_bytes", s.demand_read_bytes, false);
  add("payload_write_calls", s.payload_write_calls, false);
  add("payload_write_bytes", s.payload_write_bytes, false);
  add("reclaimed_file_bytes", s.reclaimed_file_bytes, false);
  add("dedup_verify_read_calls", s.dedup_verify_read_calls, false);
  add("admission_declined", s.admission_declined, false);
  add("restore_declined_calls", s.restore_declined_calls, false);
  const auto addq = [&out](const char* name, std::uint64_t value) {
    out += ",\"";
    out += name;
    out += "\":";
    out += std::to_string(value);
  };
  addq("queue_queued_tasks", q.queued_tasks);
  addq("queue_queued_bytes", q.queued_bytes);
  addq("queue_active", q.active);
  addq("queue_submitted", q.submitted);
  addq("queue_completed", q.completed);
  addq("queue_rejected", q.rejected);
  addq("queue_gc_aged", q.aged);
  out += "}";
  return out;
}

// 操作到四个队列的映射：只有 GC 与写入是可延迟的，读永远是需求读。
TaskClass ClassFor(proto::Op op) {
  switch (op) {
    case proto::Op::put: return TaskClass::write;
    case proto::Op::gc: return TaskClass::gc;
    default: return TaskClass::demand;
  }
}

// 跨连接等待调度结果的槽位。
struct Slot {
  void Set(std::string value) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      response = std::move(value);
      ready = true;
    }
    cv.notify_all();
  }
  std::string Wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return ready; });
    return response;
  }
  std::mutex mutex;
  std::condition_variable cv;
  bool ready = false;
  std::string response;
};

// 返回响应体；异常由调用方转换成状态码。
std::string Dispatch(Store& store, const Scheduler& scheduler, proto::Op op, proto::Reader& in) {
  switch (op) {
    case proto::Op::ping:
      return std::string("kvcohort/daemon ") + std::to_string(proto::kVersion) + "\n";

    case proto::Op::put: {
      const auto key = ReadKey(in);
      PlacementHint hint;
      hint.lifetime = static_cast<Lifetime>(in.u8());
      hint.prefix_group = std::string(in.str());
      const auto payload = in.str();
      const auto generation = store.Put(key, payload, hint);
      proto::Writer out;
      out.u64(generation);
      return out.take();
    }

    case proto::Op::get: {
      const auto key = ReadKey(in);
      auto ticket = store.Pin(key);
      if (!ticket) throw Error(ErrorCode::missing, "object is not present");
      return ticket->Read();
    }

    case proto::Op::lookup: {
      const auto count = in.u32();
      std::vector<BlockKey> chain;
      chain.reserve(count);
      for (std::uint32_t i = 0; i < count; ++i) chain.push_back(ReadKey(in));
      const auto tickets = store.LookupPrefix(chain);
      const auto payloads = store.ReadBatch(tickets);
      proto::Writer out;
      out.u32(static_cast<std::uint32_t>(payloads.size()));
      for (const auto& bytes : payloads) out.str(bytes);
      return out.take();
    }

    case proto::Op::retire: {
      proto::Writer out;
      out.u8(store.Retire(ReadKey(in)) ? 1 : 0);
      return out.take();
    }

    case proto::Op::seal:
      store.Seal();
      return {};

    case proto::Op::gc: {
      proto::Writer out;
      out.u8(store.CollectGarbage() ? 1 : 0);
      return out.take();
    }

    case proto::Op::stats: {
      proto::Writer out;
      out.str(StatsJson(store.GetStats(), scheduler.GetStats()));
      return out.take();
    }

    case proto::Op::shutdown:
      g_stop.store(true);
      return {};
  }
  throw Error(ErrorCode::invalid, "unknown opcode");
}

void Serve(int fd, Store& store, Scheduler& scheduler) {
  while (!g_stop.load()) {
    std::uint8_t header[4];
    if (!proto::RecvAll(fd, header, sizeof(header))) break;
    proto::Reader header_reader(std::string_view(reinterpret_cast<const char*>(header), sizeof(header)));
    const auto length = header_reader.u32();
    // 畸形长度直接断链，不做任何分配。
    if (length < 2 || length > proto::kMaxFrame) break;

    std::string payload(length, '\0');
    if (!proto::RecvAll(fd, payload.data(), payload.size())) break;

    // 先解析头部以便排队，再交给调度器执行；等待期间本连接阻塞，
    // 因此优先级是跨连接的（别的连接的需求读可以插到本连接的写之前）。
    proto::Reader head(payload);
    std::string response;
    proto::Op op = proto::Op::ping;
    try {
      const auto version = head.u8();
      if (version != proto::kVersion)
        throw Error(ErrorCode::invalid, "protocol version mismatch");
      op = static_cast<proto::Op>(head.u8());
    } catch (const std::exception& e) {
      response = proto::EncodeFrame(proto::Status::invalid, e.what());
      if (!proto::SendAll(fd, response)) break;
      continue;
    }

    const auto weight = (op == proto::Op::put) ? static_cast<std::uint64_t>(payload.size()) : 0ull;
    auto slot = std::make_shared<Slot>();
    const bool accepted = scheduler.TrySubmit(ClassFor(op), weight,
        [&store, &scheduler, op, payload, slot] {
          std::string result;
          try {
            proto::Reader in(payload);
            in.u8();
            in.u8();
            result = proto::EncodeFrame(proto::Status::ok, Dispatch(store, scheduler, op, in));
          } catch (const Error& e) {
            result = proto::EncodeFrame(StatusFor(e.code), e.what());
          } catch (const std::exception& e) {
            result = proto::EncodeFrame(proto::Status::io, e.what());
          }
          slot->Set(std::move(result));
        });
    if (!accepted) {
      // 排队上限触发背压：明确回 busy，而不是无限排队或静默丢弃。
      response = proto::EncodeFrame(proto::Status::busy, "scheduler queue is full");
    } else {
      response = slot->Wait();
    }
    if (!proto::SendAll(fd, response)) break;
  }
  ::close(fd);
}

void OnSignal(int) { g_stop.store(true); }

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: kvd ROOT [--socket PATH] [--layout NAME] "
                   "[--segment-kib N] [--payload-budget-kib N] [--calibrate-every N] "
                   "[--workers N] [--gc-aging-ms N] [--queued-tasks N] [--direct-io 0|1]\n";
      return 2;
    }
    const std::string root = argv[1];
    std::string socket_path;
    Options options;
    SchedulerOptions scheduler_options;
    for (int i = 2; i < argc; i += 2) {
      if (i + 1 >= argc) throw Error(ErrorCode::invalid, "option requires a value");
      const std::string flag = argv[i];
      const std::string value = argv[i + 1];
      if (flag == "--socket") socket_path = value;
      else if (flag == "--layout") options.layout = ParseLayout(value);
      else if (flag == "--segment-kib") options.segment_bytes = std::stoull(value) * 1024;
      else if (flag == "--payload-budget-kib") options.payload_budget_bytes = std::stoull(value) * 1024;
      else if (flag == "--calibrate-every") options.calibrate_every_n_writes = std::stoull(value);
      else if (flag == "--workers") scheduler_options.workers = static_cast<std::size_t>(std::stoull(value));
      else if (flag == "--gc-aging-ms") scheduler_options.gc_aging_ms = std::stoull(value);
      else if (flag == "--queued-tasks") scheduler_options.max_queued_tasks = static_cast<std::size_t>(std::stoull(value));
      else if (flag == "--direct-io") options.direct_io = std::stoull(value) != 0;
      else throw Error(ErrorCode::invalid, "unknown option: " + flag);
    }
    // 参数归一化，与回放器保持一致：
    // 1) 对象不能大于所在段；
    // 2) 载荷预算必须严格大于 GC 预留，否则准入永远无法通过。
    if (options.max_object_bytes > options.segment_bytes)
      options.max_object_bytes = options.segment_bytes;
    if (options.payload_budget_bytes != 0 && options.gc_reserve_bytes >= options.payload_budget_bytes)
      options.gc_reserve_bytes = options.segment_bytes;

    if (socket_path.empty()) socket_path = root + "/kvd.sock";
    if (socket_path.size() >= sizeof(sockaddr_un::sun_path))
      throw Error(ErrorCode::invalid, "socket path exceeds AF_UNIX limit");

    ::signal(SIGPIPE, SIG_IGN);
    ::signal(SIGINT, OnSignal);
    ::signal(SIGTERM, OnSignal);

    // Store 必须在监听之前构造：它负责独占数据根，失败要尽早暴露。
    Store store(root, options);
    // 调度器放在 Store 之后、工作线程之前：析构顺序要求它是最后被销毁的。
    Scheduler scheduler(scheduler_options);

    ::unlink(socket_path.c_str());
    const int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) IoFail("create listening socket");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (::bind(listen_fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0)
      IoFail("bind " + socket_path);
    ::chmod(socket_path.c_str(), 0600);
    if (::listen(listen_fd, 64) != 0) IoFail("listen");

    std::cout << "kvd listening on " << socket_path << " (workers=" << scheduler_options.workers
              << ", gc_aging_ms=" << scheduler_options.gc_aging_ms << ")" << std::endl;

    std::mutex workers_mutex;
    std::vector<std::thread> workers;
    while (!g_stop.load()) {
      pollfd entry{};
      entry.fd = listen_fd;
      entry.events = POLLIN;
      const int ready = ::poll(&entry, 1, kPollTimeoutMs);
      if (ready < 0) {
        if (errno == EINTR) continue;
        IoFail("poll");
      }
      if (ready == 0) continue;
      const int client_fd = ::accept(listen_fd, nullptr, nullptr);
      if (client_fd < 0) {
        if (errno == EINTR) continue;
        IoFail("accept");
      }
      std::lock_guard<std::mutex> lock(workers_mutex);
      workers.emplace_back([client_fd, &store, &scheduler] { Serve(client_fd, store, scheduler); });
    }

    ::close(listen_fd);
    ::unlink(socket_path.c_str());
    {
      std::lock_guard<std::mutex> lock(workers_mutex);
      for (auto& worker : workers)
        if (worker.joinable()) worker.join();
    }
    std::cout << "kvd stopped" << std::endl;
    return 0;
  } catch (const Error& e) {
    std::cerr << "kvd error " << static_cast<int>(e.code) << ": " << e.what() << '\n';
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "kvd error: " << e.what() << '\n';
    return 1;
  }
}
