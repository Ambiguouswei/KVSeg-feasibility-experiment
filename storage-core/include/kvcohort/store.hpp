#pragma once

#include "kvcohort/policy.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace kvcohort {

enum class ErrorCode { invalid, missing, conflict, capacity, corruption, busy, io, declined };

class Error : public std::runtime_error {
 public:
  Error(ErrorCode error_code, const std::string& message) : std::runtime_error(message), code(error_code) {}
  ErrorCode code;
};

// Caller supplies exact, versioned model/representation identities. Authentication
// belongs to the future daemon adapter, not this trusted in-process library.
std::string MakeNamespace(std::string_view tenant, std::string_view model_revision,
                          std::string_view representation, std::string_view extra_identity = {});

struct BlockKey {
  std::string space;
  std::string digest;
  std::string parent;
  std::string id() const;
};

BlockKey MakeBlockKey(const std::string& space, const std::string& parent,
                      const std::vector<std::uint32_t>& token_ids);

struct Options {
  std::uint64_t segment_bytes = 4 * 1024 * 1024;
  std::uint64_t max_object_bytes = 1024 * 1024;
  std::uint64_t capacity_bytes = 64 * 1024 * 1024;
  std::uint64_t gc_reserve_bytes = 4 * 1024 * 1024;
  std::uint64_t metadata_reserve_bytes = 8 * 1024 * 1024;
  // Optional bound on committed segment payload, including dead/pinned bytes.
  // Normal writes leave gc_reserve_bytes unused within this budget too.
  std::uint64_t payload_budget_bytes = 0;
  Layout layout = Layout::append;
  std::uint64_t max_open_segments = 8;
  // 准入检查每 N 次成功写入才做一次全树占用扫描。
  // 0（默认）= 每次都精确扫描，与 v0.2.0 完全一致——库的默认行为必须保守，
  // 因为估算只累加载荷字节，抓不到 RocksDB 的 WAL 与压缩增长。
  // 实测路径（回放器）会显式设为 64 以消除写路径上的 O(文件数) 开销。
  std::uint64_t calibrate_every_n_writes = 0;
  // 实验开关：GC 目标段是否重新成为该 cohort 的活动段。
  // false = v0.2.0 行为（目标段封口，不再追加）。用途是把「布局效应」与
  // 「GC 封段这一实现细节」分离——prefix 布局的低搬移很可能来自后者。
  bool gc_target_appendable = false;
  // 准入 / 读取决策 / GC 评分参数。默认全部关闭，等价于 v0.2.0 行为。
  PolicyConfig policy;
  // 直接 I/O：绕过页缓存，使 pread 真正落到设备。
  // O_DIRECT 要求偏移、长度、缓冲区均按 4096 对齐；实现内部用对齐缓冲处理
  // 非对齐请求（会多读/多写少量字节），因此段文件尾部会被补齐到 4096 的倍数。
  bool direct_io = false;
  // Tests only: injected after well-defined persistence boundaries.
  std::function<void(std::string_view)> failpoint;
};

struct Stats {
  std::uint64_t objects = 0;
  std::uint64_t live_payload_bytes = 0;
  std::uint64_t segment_bytes = 0;
  std::uint64_t allocated_bytes = 0;
  std::uint64_t retired_segment_bytes = 0;
  std::uint64_t segments = 0;
  std::uint64_t gc_moved_bytes = 0;
  std::uint64_t active_segments = 0;
  std::uint64_t demand_read_calls = 0;
  std::uint64_t demand_read_bytes = 0;
  std::uint64_t payload_write_calls = 0;
  std::uint64_t payload_write_bytes = 0;
  std::uint64_t gc_read_calls = 0;
  std::uint64_t gc_read_bytes = 0;
  std::uint64_t gc_write_calls = 0;
  std::uint64_t gc_write_bytes = 0;
  std::uint64_t reclaimed_file_bytes = 0;
  // 去重命中时对既有对象的完整性重读；与需求读分开计量。
  std::uint64_t dedup_verify_read_calls = 0;
  std::uint64_t dedup_verify_read_bytes = 0;
  // 准入拒绝（无复用证据）与恢复放弃（恢复比重算更贵）的计数。
  std::uint64_t admission_declined = 0;
  std::uint64_t restore_declined_calls = 0;
  std::uint64_t restore_bytes_avoided = 0;
};

struct BatchReadOptions {
  std::uint64_t max_batch_bytes = 1024 * 1024;
  std::uint64_t max_gap_bytes = 0;
  std::uint64_t max_total_output_bytes = 64 * 1024 * 1024;
};

struct SegmentFile;

class ReadTicket {
 public:
  ReadTicket() = default;
  std::string Read() const;
  std::uint64_t generation() const { return generation_; }
  std::uint64_t size() const { return length_; }
  explicit operator bool() const { return static_cast<bool>(file_); }
 private:
  friend class Store;
  std::shared_ptr<SegmentFile> file_;
  std::uint64_t offset_ = 0;
  std::uint64_t length_ = 0;
  std::uint64_t generation_ = 0;
  bool direct_ = false;
  std::string checksum_;
};

class Store {
 public:
  explicit Store(const std::filesystem::path& root, Options options = {});
  ~Store();
  Store(const Store&) = delete;
  Store& operator=(const Store&) = delete;
  // A successful Put is durable. First stage deliberately has no volatile mode.
  std::uint64_t Put(const BlockKey& key, std::string_view payload, const PlacementHint& hint = {});
  std::optional<ReadTicket> Pin(const BlockKey& key);
  std::vector<ReadTicket> LookupPrefix(const std::vector<BlockKey>& chain);
  // Tickets remain pinned throughout sorted/coalesced physical reads. Results
  // preserve request order. Counts are pread calls, not block-device operations.
  std::vector<std::string> ReadBatch(const std::vector<ReadTicket>& tickets, BatchReadOptions options = {});
  bool Retire(const BlockKey& key);
  void Seal();
  bool CollectGarbage();
  std::uint64_t ReapRetired();
  // 低成本：只读内存计数器与段目录，并复用缓存的占用估算，可放在热路径上。
  Stats GetStats() const;
  // 精确：扫描 obj/ 前缀并重新统计根目录占用，代价 O(对象数 + 文件数)。
  Stats GetStatsExact() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvcohort
