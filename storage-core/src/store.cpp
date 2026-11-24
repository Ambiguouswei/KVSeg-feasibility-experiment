#include "kvcohort/store.hpp"

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>
#include <openssl/sha.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <utility>

#include <cstdlib>
#include <cstring>

namespace kvcohort {
namespace fs = std::filesystem;
namespace {
// 冻结的磁盘格式标识：文件名称与内容都是 on-disk 契约的一部分，
// 不随项目改名而变化，否则 v0.2.0 生成的数据根将无法打开。
constexpr std::string_view kMarker = "PREFIX_KVFS_PROTOTYPE_V1\n";

constexpr std::uint64_t kDirectAlign = 4096;
constexpr std::uint64_t AlignUp(std::uint64_t value) {
  return (value + kDirectAlign - 1) / kDirectAlign * kDirectAlign;
}

// O_DIRECT 要求缓冲区按块对齐；用自己的 RAII 包一层，避免到处手写 posix_memalign。
class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t bytes) : size_(bytes) {
    if (bytes == 0) return;
    void* raw = nullptr;
    if (::posix_memalign(&raw, kDirectAlign, bytes) != 0) throw Error(ErrorCode::io, "posix_memalign failed");
    data_ = static_cast<char*>(raw);
  }
  ~AlignedBuffer() { std::free(data_); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  char* data() { return data_; }
  std::size_t size() const { return size_; }

 private:
  char* data_ = nullptr;
  std::size_t size_ = 0;
};

struct IoTally {
  std::atomic<std::uint64_t> calls{0}, bytes{0};
};
struct IoCounters {
  IoTally demand, payload, gc_read, gc_write;
  // 去重命中时对既有对象的完整性重读；单独计量，不混入需求读。
  IoTally verify;
};

[[noreturn]] void Io(const std::string& what) {
  throw Error(ErrorCode::io, what + ": " + std::strerror(errno));
}
void Check(const rocksdb::Status& status) {
  if (!status.ok()) throw Error(status.IsCorruption() ? ErrorCode::corruption : ErrorCode::io, status.ToString());
}
std::string Digest(std::string_view bytes) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> hash{};
  SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), hash.data());
  std::string result;
  constexpr char hex[] = "0123456789abcdef";
  for (auto c : hash) { result += hex[c >> 4]; result += hex[c & 15]; }
  return result;
}
bool IsDigest(std::string_view s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
void Field(std::string& out, std::string_view value) {
  const auto length = static_cast<std::uint64_t>(value.size());
  for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<char>((length >> (8 * i)) & 255));
  out.append(value);
}
void Validate(const BlockKey& key) {
  if (!IsDigest(key.space) || !IsDigest(key.digest) || (!key.parent.empty() && !IsDigest(key.parent)))
    throw Error(ErrorCode::invalid, "invalid versioned namespace or block digest");
}
void Sync(int fd) { if (::fsync(fd) != 0) Io("fsync"); }
void SyncDir(const fs::path& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) Io("open directory");
  const int code = ::fsync(fd); const int saved = errno; ::close(fd); errno = saved;
  if (code != 0) Io("fsync directory");
}
void WriteAll(int fd, std::string_view bytes, std::uint64_t offset, IoTally* tally = nullptr,
              bool direct = false) {
  if (direct) {
    // 载荷拷进对齐缓冲，尾部补零后整块写出。多出的尾部永远位于 used 之后，
    // 恢复时会被截断，不会被当作数据读出。
    const auto aligned = static_cast<std::size_t>(AlignUp(bytes.size()));
    AlignedBuffer buffer(aligned);
    std::memset(buffer.data(), 0, aligned);
    if (!bytes.empty()) std::memcpy(buffer.data(), bytes.data(), bytes.size());
    std::size_t done = 0;
    while (done < aligned) {
      const auto n = ::pwrite(fd, buffer.data() + done, aligned - done, static_cast<off_t>(offset + done));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) { if (n == 0) errno = EIO; Io("pwrite segment"); }
      done += static_cast<std::size_t>(n);
    }
    if (tally) {
      tally->calls.fetch_add(1, std::memory_order_relaxed);
      tally->bytes.fetch_add(static_cast<std::uint64_t>(bytes.size()), std::memory_order_relaxed);
    }
    return;
  }
  while (!bytes.empty()) {
    if (tally) tally->calls.fetch_add(1,std::memory_order_relaxed);
    const auto n = ::pwrite(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) { if (n == 0) errno = EIO; Io("pwrite segment"); }
    if (tally) tally->bytes.fetch_add(static_cast<std::uint64_t>(n),std::memory_order_relaxed);
    bytes.remove_prefix(static_cast<std::size_t>(n)); offset += static_cast<std::uint64_t>(n);
  }
}
std::string ReadAll(int fd, std::uint64_t offset, std::uint64_t length, IoTally* tally = nullptr,
                    bool direct = false) {
  if (length > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw Error(ErrorCode::corruption, "object is too large for address space");
  if (direct) {
    // 前后各补齐到 4096 边界再读，然后只保留请求的那一段。
    const std::uint64_t start = offset / kDirectAlign * kDirectAlign;
    const std::uint64_t end = AlignUp(offset + length);
    const auto span = static_cast<std::size_t>(end - start);
    AlignedBuffer buffer(span);
    std::size_t done = 0;
    while (done < span) {
      const auto n = ::pread(fd, buffer.data() + done, span - done, static_cast<off_t>(start + done));
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) Io("pread segment");
      if (n == 0) throw Error(ErrorCode::corruption, "short object read");
      done += static_cast<std::size_t>(n);
    }
    if (tally) {
      tally->calls.fetch_add(1, std::memory_order_relaxed);
      tally->bytes.fetch_add(length, std::memory_order_relaxed);
    }
    return std::string(buffer.data() + static_cast<std::size_t>(offset - start),
                       static_cast<std::size_t>(length));
  }
  std::string bytes(static_cast<std::size_t>(length), '\0');
  std::size_t done = 0;
  while (done < bytes.size()) {
    if (tally) tally->calls.fetch_add(1,std::memory_order_relaxed);
    const auto n = ::pread(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(offset + done));
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) Io("pread segment");
    if (n == 0) throw Error(ErrorCode::corruption, "short object read");
    if (tally) tally->bytes.fetch_add(static_cast<std::uint64_t>(n),std::memory_order_relaxed);
    done += static_cast<std::size_t>(n);
  }
  return bytes;
}
std::string SegmentName(std::uint64_t id) {
  std::ostringstream out; out << std::setw(20) << std::setfill('0') << id << ".seg"; return out.str();
}
std::string SegKey(std::uint64_t id) { return "seg/" + SegmentName(id); }
std::string RefPrefix(std::uint64_t id) { return "ref/" + SegmentName(id) + "/"; }
bool Starts(std::string_view s, std::string_view prefix) { return s.substr(0, prefix.size()) == prefix; }
std::uint64_t Allocated(const fs::path& root) {
  std::uint64_t bytes = 0;
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    struct stat st{}; if (::lstat(entry.path().c_str(), &st) != 0) { if (errno == ENOENT) continue; Io("stat allocation"); }
    if (S_ISLNK(st.st_mode)) throw Error(ErrorCode::invalid, "symlink inside owned data root");
    if (!S_ISREG(st.st_mode)) continue;
    bytes += static_cast<std::uint64_t>(st.st_blocks) * 512;
  }
  return bytes;
}
struct RootLock {
  explicit RootLock(const fs::path& path) {
    fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) Io("open root lock");
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
      ::close(fd); fd = -1; throw Error(ErrorCode::busy, "data root already owned by another Store or read ticket");
    }
  }
  ~RootLock() { if (fd >= 0) ::close(fd); }
  int fd = -1;
};
struct Record {
  std::uint64_t segment = 0, offset = 0, length = 0, generation = 0;
  std::string checksum, parent;
  std::string Encode() const {
    std::ostringstream out;
    out << "v1 " << segment << ' ' << offset << ' ' << length << ' ' << generation << ' ' << checksum << ' ' << (parent.empty() ? "-" : parent);
    return out.str();
  }
  static Record Decode(const std::string& data) {
    Record r; std::string version, extra; std::istringstream in(data);
    if (!(in >> version >> r.segment >> r.offset >> r.length >> r.generation >> r.checksum >> r.parent) ||
        version != "v1" || !IsDigest(r.checksum) || (r.parent != "-" && !IsDigest(r.parent)) || (in >> extra))
      throw Error(ErrorCode::corruption, "invalid object manifest");
    if (r.parent == "-") r.parent.clear();
    return r;
  }
};
struct SegmentMeta {
  char state = 'a';  // active, sealed, retired, or GC staging
  std::uint64_t used = 0;
  std::string cohort = "legacy";
  std::string Encode() const { return "v2 " + std::string(1, state) + " " + std::to_string(used) + " " + cohort; }
  static SegmentMeta Decode(const std::string& data) {
    SegmentMeta m; std::string v, extra; std::istringstream in(data);
    if (!(in >> v >> m.state >> m.used) || (v != "v1" && v != "v2") || std::string("asrt").find(m.state) == std::string::npos)
      throw Error(ErrorCode::corruption, "invalid segment catalog");
    if (v=="v2" && !(in >> m.cohort)) throw Error(ErrorCode::corruption,"missing segment cohort");
    if (m.cohort.empty() || m.cohort.size()>200 || !std::all_of(m.cohort.begin(),m.cohort.end(),[](char c){
          return (c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='/';}) || (in >> extra))
      throw Error(ErrorCode::corruption, "invalid segment catalog");
    return m;
  }
};
}  // namespace

struct SegmentFile {
  ~SegmentFile() { if (fd >= 0) ::close(fd); }
  int fd = -1;
  fs::path path;
  std::shared_ptr<RootLock> owner;  // root remains exclusively owned until reads finish
  std::shared_ptr<IoCounters> counters;
};

std::string MakeNamespace(std::string_view tenant, std::string_view model, std::string_view representation, std::string_view extra) {
  if (tenant.empty() || model.empty() || representation.empty())
    throw Error(ErrorCode::invalid, "tenant, exact model revision and representation must be nonempty");
  std::string encoded = "namespace:v1";
  for (auto value : {tenant, model, representation, extra}) Field(encoded, value);
  return Digest(encoded);
}
BlockKey MakeBlockKey(const std::string& space, const std::string& parent, const std::vector<std::uint32_t>& tokens) {
  if (!IsDigest(space) || (!parent.empty() && !IsDigest(parent)) || tokens.empty() || tokens.size() > 1024 * 1024)
    throw Error(ErrorCode::invalid, "invalid block identity");
  std::string encoded = "prefix:v1"; Field(encoded, space); Field(encoded, parent);
  for (auto token : tokens) for (unsigned i = 0; i < 4; ++i) encoded.push_back(static_cast<char>((token >> (8 * i)) & 255));
  return {space, Digest(encoded), parent};
}
std::string BlockKey::id() const { Validate(*this); return space + "/" + digest; }
std::string ReadTicket::Read() const {
  if (!file_) throw Error(ErrorCode::invalid, "empty read ticket");
  auto bytes = ReadAll(file_->fd, offset_, length_, &file_->counters->demand, direct_);
  if (Digest(bytes) != checksum_) throw Error(ErrorCode::corruption, "payload SHA256 mismatch");
  return bytes;
}

struct Store::Impl {
  fs::path root, data;
  Options options;
  std::shared_ptr<RootLock> owner;
  std::unique_ptr<rocksdb::DB> db;
  std::unique_ptr<PlacementPolicy> policy;
  std::shared_ptr<IoCounters> counters = std::make_shared<IoCounters>();
  mutable std::mutex mutex;
  std::map<std::uint64_t, SegmentMeta> catalog;
  std::map<std::uint64_t, std::shared_ptr<SegmentFile>> files;
  std::map<std::string,std::uint64_t> active_by_cohort;
  std::map<std::uint64_t,std::uint64_t> last_write;
  std::uint64_t next_segment = 1, next_generation = 1, moved = 0, clock = 0, reclaimed = 0;
  // 增量维护的存活统计，使 GetStats() 不必扫描 obj/ 前缀。
  std::uint64_t live_objects_ = 0, live_payload_ = 0;
  // 占用估算：快照 + 自上次校准以来的增量。
  mutable std::uint64_t write_seq_ = 0;
  mutable bool alloc_calibrated_ = false;
  mutable std::uint64_t alloc_snapshot_ = 0, alloc_snapshot_writes_ = 0;
  mutable std::uint64_t alloc_added_ = 0, alloc_freed_ = 0;
  // 准入证据：前缀组摘要 -> 被命中次数（有界近似）。
  std::map<std::string,std::uint8_t> read_evidence_;
  std::uint64_t admission_declined_ = 0;
  std::uint64_t declined_restores_ = 0, declined_bytes_ = 0;
  bool poisoned = false;

  Impl(const fs::path& requested, Options opts) : root(fs::absolute(requested).lexically_normal()), data(root / "segments"), options(std::move(opts)) {
    if (options.segment_bytes == 0 || options.max_object_bytes == 0 || options.max_object_bytes > options.segment_bytes ||
        options.segment_bytes > (1ULL << 32) || options.capacity_bytes <= options.gc_reserve_bytes ||
        options.capacity_bytes - options.gc_reserve_bytes <= options.metadata_reserve_bytes ||
        options.gc_reserve_bytes < options.segment_bytes || options.max_open_segments==0 || options.max_open_segments>64 ||
        (options.payload_budget_bytes && options.payload_budget_bytes<=options.gc_reserve_bytes))
      throw Error(ErrorCode::invalid, "invalid segment, object, capacity or reserve limits");
    policy=MakePlacementPolicy(options.layout);
    if (fs::is_symlink(root)) throw Error(ErrorCode::invalid, "data root must not be a symlink");
    const auto marker = root / "KVFS_FORMAT";
    if (fs::exists(root) && !fs::exists(marker) && !fs::is_empty(root))
      throw Error(ErrorCode::invalid, "refusing a nonempty directory without our format marker");
    if (fs::create_directories(root)) fs::permissions(root, fs::perms::owner_all, fs::perm_options::replace);
    owner = std::make_shared<RootLock>(root / "OWNER.lock");
    if (!fs::exists(marker)) {
      const int fd = ::open(marker.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
      if (fd < 0) Io("create format marker");
      try { WriteAll(fd, kMarker, 0); Sync(fd); } catch (...) { ::close(fd); throw; }
      ::close(fd); SyncDir(root); SyncDir(root.parent_path());
    } else {
      int fd = ::open(marker.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
      if (fd < 0) Io("open format marker");
      std::string got;
      try { got = ReadAll(fd, 0, kMarker.size()); } catch (...) { ::close(fd); throw; }
      ::close(fd);
      if (got != kMarker || fs::file_size(marker) != kMarker.size()) throw Error(ErrorCode::invalid, "unsupported format marker");
    }
    for (const auto& path : {data, root / "metadata"}) {
      if (fs::is_symlink(path)) throw Error(ErrorCode::invalid, "owned subdirectory must not be a symlink");
      fs::create_directories(path);
    }
    SyncDir(root);
    rocksdb::Options db_options;
    db_options.create_if_missing = true;
    db_options.paranoid_checks = true;
    db_options.max_background_jobs = 2;
    db_options.write_buffer_size = 256 * 1024;
    db_options.max_write_buffer_number = 2;
    db_options.target_file_size_base = 1024 * 1024;
    db_options.max_total_wal_size = 1024 * 1024;
    db_options.keep_log_file_num = 2;
    db_options.max_log_file_size = 256 * 1024;
    rocksdb::DB* raw = nullptr;
    Check(rocksdb::DB::Open(db_options, (root / "metadata").string(), &raw)); db.reset(raw);
    std::string format;
    const auto status = db->Get(rocksdb::ReadOptions(), "cfg/format", &format);
    const auto expected = "v1 " + std::to_string(options.segment_bytes);
    if (status.IsNotFound()) {
      rocksdb::WriteBatch batch; batch.Put("cfg/format", expected); batch.Put("cfg/next_segment", "1"); batch.Put("cfg/next_generation", "1"); Commit(batch);
    } else { Check(status); if (format != expected) throw Error(ErrorCode::invalid, "segment format/options differ from persisted configuration"); }
    std::string layout;
    const auto layout_status=db->Get(rocksdb::ReadOptions(),"cfg/layout",&layout);
    if(layout_status.IsNotFound()) {
      rocksdb::WriteBatch batch;batch.Put("cfg/layout","layout:v1 "+policy->name());Commit(batch);
    } else {
      Check(layout_status);
      if(layout!="layout:v1 "+policy->name()) throw Error(ErrorCode::invalid,"layout differs from persisted policy; use a separate data root for each strategy");
    }
    next_segment = Counter("cfg/next_segment"); next_generation = Counter("cfg/next_generation");
    std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(rocksdb::ReadOptions()));
    for (it->Seek("seg/"); it->Valid() && Starts(it->key().ToStringView(), "seg/"); it->Next()) {
      const auto key = it->key().ToString();
      if (key.size() != 28 || key.substr(24) != ".seg") throw Error(ErrorCode::corruption, "invalid segment catalog key");
      const auto id = static_cast<std::uint64_t>(std::stoull(key.substr(4, 20)));
      auto m = SegmentMeta::Decode(it->value().ToString());
      if (m.used > options.segment_bytes || id >= next_segment) throw Error(ErrorCode::corruption, "invalid committed segment bounds");
      catalog[id] = m;
    }
    Check(it->status()); it.reset();
    for (auto& [id, m] : catalog) {
      const auto path = data / SegmentName(id);
      if (m.state == 'r' || m.state == 't') continue;
      if (m.state == 'a') {
        if(m.cohort!="legacy") {
          if(active_by_cohort.count(m.cohort)) throw Error(ErrorCode::corruption,"multiple active segments in one cohort");
          active_by_cohort[m.cohort]=id;last_write[id]=++clock;
        }
      }
      if (!fs::exists(path) || fs::file_size(path) < m.used) throw Error(ErrorCode::corruption, "missing or truncated committed segment");
      auto f = OpenFile(path, false);
      // Discard payload written before a failed metadata publication.
      // 直接 I/O 下，尾部对齐读会触及 used 之后的字节，因此把文件补齐到 4096 的倍数；
      // 这些字节永远位于 used 之外，不会被当作对象内容读出。
      const auto target_size = options.direct_io ? AlignUp(m.used) : m.used;
      if (fs::file_size(path) != target_size) { if (::ftruncate(f->fd, static_cast<off_t>(target_size)) != 0) Io("truncate uncommitted tail"); Sync(f->fd); }
      files[id] = std::move(f);
    }
    for(auto& [id,m]:catalog) if(m.state=='a'&&m.cohort=="legacy") SealOne(id);
    while(active_by_cohort.size()>options.max_open_segments) SealOldest();
    for (auto& [id, m] : catalog) if (m.state == 't') m.state = 'r';
    Reap();
    // 种子化存活统计，使 GetStats() 之后无需扫描 obj/ 前缀。
    { std::unique_ptr<rocksdb::Iterator> oit(db->NewIterator(rocksdb::ReadOptions()));
      for (oit->Seek("obj/"); oit->Valid() && Starts(oit->key().ToStringView(), "obj/"); oit->Next()) {
        ++live_objects_; live_payload_ += Record::Decode(oit->value().ToString()).length;
      }
      Check(oit->status()); }
    // Scan only segment filenames, never all tensor contents.
    for (const auto& entry : fs::directory_iterator(data)) {
      if (entry.is_symlink() || !entry.is_regular_file()) throw Error(ErrorCode::invalid, "unexpected entry in segment directory");
      const auto name = entry.path().filename().string();
      if (name.size() != 24 || name.substr(20) != ".seg" || !std::all_of(name.begin(), name.begin() + 20, [](char c) { return c >= '0' && c <= '9'; }))
        throw Error(ErrorCode::invalid, "unknown file in owned segment directory");
      const auto id = static_cast<std::uint64_t>(std::stoull(name.substr(0, 20)));
      if (!catalog.count(id)) { fs::remove(entry.path()); SyncDir(data); }
    }
  }
  std::uint64_t Counter(const std::string& key) const {
    std::string value; Check(db->Get(rocksdb::ReadOptions(), key, &value));
    std::size_t consumed = 0; const auto n = std::stoull(value, &consumed);
    if (consumed != value.size() || n == 0) throw Error(ErrorCode::corruption, "invalid durable counter");
    return static_cast<std::uint64_t>(n);
  }
  void Ready() const { if (poisoned) throw Error(ErrorCode::io, "previous I/O failure: close and reopen the cache before further operations"); }
  // v0.2.0 每次 Put 都递归扫全根统计 st_blocks，是写路径的主要开销。
  // 改为「快照 + 增量」，每 calibrate_every_n_writes 次写入重新校准一次。
  // 一个对象的前缀组 = 它的父块摘要（根块用自身摘要）。
  static const std::string& EvidenceKey(const BlockKey& key) {
    return key.parent.empty() ? key.digest : key.parent;
  }
  // 记录一次「该前缀被读过」。仅在开启准入时维护，避免无意义的内存增长。
  void NoteReuse(const BlockKey& key) {
    if (!options.policy.admit_enabled) return;
    auto& slot = read_evidence_[EvidenceKey(key)];
    if (slot < std::numeric_limits<std::uint8_t>::max()) ++slot;
    if (read_evidence_.size() > options.policy.admit_evidence_capacity) read_evidence_.clear();
  }
  // 准入判据：该前缀组此前至少被命中 admit_min_reuse 次才允许落盘。
  // 没有复用证据的数据（例如 no-reuse 负载）应当被拒绝写入。
  bool Admit(const BlockKey& key) {
    if (!options.policy.admit_enabled) return true;
    // 根块是整条前缀链的共享锚点：它没有父前缀，按定义永远拿不到「父块被读过」的证据。
    // 若不给豁免，冷启动时任何链的第一个块都会被拒，整个缓存永远无法建立。
    if (key.parent.empty()) return true;
    const auto it = read_evidence_.find(EvidenceKey(key));
    const std::uint32_t seen = it == read_evidence_.end() ? 0U : static_cast<std::uint32_t>(it->second);
    if (seen < options.policy.admit_min_reuse) { ++admission_declined_; return false; }
    return true;
  }
  // 恢复判据：预计恢复时间与预计重算时间比较。任一速率未标定则退回「总是恢复」。
  bool ShouldRestore(std::uint64_t bytes, std::uint32_t blocks) {
    if (!options.policy.restore_decision_enabled) return true;
    if (options.policy.restore_bytes_per_us <= 0.0 || options.policy.prefill_us_per_token <= 0.0) return true;
    const double restore_us = static_cast<double>(bytes) / options.policy.restore_bytes_per_us +
                              options.policy.restore_queue_us;
    const double recompute_us = static_cast<double>(blocks) *
                                static_cast<double>(options.policy.tokens_per_block) *
                                options.policy.prefill_us_per_token;
    if (restore_us >= recompute_us) { ++declined_restores_; declined_bytes_ += bytes; return false; }
    return true;
  }
  std::uint64_t AllocatedEstimate(bool force = false) const {
    // 首次调用必须校准：此时 write_seq_ 与快照序号都是 0，差值条件不成立，
    // 若不显式标记，就会返回快照 0，让准入检查把整块盘当成空闲。
    const bool stale = force || !alloc_calibrated_ ||
                       options.calibrate_every_n_writes == 0 ||
                       write_seq_ - alloc_snapshot_writes_ >= options.calibrate_every_n_writes;
    if (stale) {
      alloc_snapshot_ = Allocated(root);
      alloc_calibrated_ = true;
      alloc_snapshot_writes_ = write_seq_;
      alloc_added_ = 0;
      alloc_freed_ = 0;
      return alloc_snapshot_;
    }
    return alloc_added_ > alloc_freed_ ? alloc_snapshot_ + (alloc_added_ - alloc_freed_) : alloc_snapshot_;
  }
  void Commit(rocksdb::WriteBatch& batch) {
    rocksdb::WriteOptions write; write.sync = true; Check(db->Write(write, &batch));
  }
  void Fault(std::string_view name) { if (options.failpoint) options.failpoint(name); }
  std::shared_ptr<SegmentFile> OpenFile(const fs::path& path, bool create) {
    auto file = std::make_shared<SegmentFile>(); file->owner = owner; file->path = path;
    file->counters=counters;
    file->fd = ::open(path.c_str(),
                      O_RDWR | O_CLOEXEC | O_NOFOLLOW | (options.direct_io ? O_DIRECT : 0) |
                          (create ? O_CREAT | O_EXCL : 0),
                      0600);
    if (file->fd < 0) Io("open segment");
    if (create) { Sync(file->fd); SyncDir(data); }
    return file;
  }
  void Capacity(std::uint64_t payload_extra, bool gc) const {
    if(options.payload_budget_bytes) {
      std::uint64_t current=0;for(const auto& [id,m]:catalog){(void)id;current+=m.used;}
      const auto limit=options.payload_budget_bytes-(gc?0:options.gc_reserve_bytes);
      if(current>limit||payload_extra>limit-current) throw Error(ErrorCode::capacity,"committed payload budget exhausted, including dead and pinned bytes");
    }
    const auto extra=((payload_extra+4095)/4096)*4096+8192;
    const auto used = AllocatedEstimate();
    const auto reserve = options.metadata_reserve_bytes + (gc ? 0 : options.gc_reserve_bytes);
    if (reserve > options.capacity_bytes || used > options.capacity_bytes - reserve || extra > options.capacity_bytes - reserve - used)
      throw Error(ErrorCode::capacity, "admission budget exhausted; retire and collect objects, or skip cache storage");
  }
  std::uint64_t Create(char state,const std::string& cohort) {
    if (next_segment == std::numeric_limits<std::uint64_t>::max()) throw Error(ErrorCode::capacity, "segment identifier exhausted");
    const auto id = next_segment;
    auto f = OpenFile(data / SegmentName(id), true);
    SegmentMeta m{state, 0,cohort};
    rocksdb::WriteBatch batch; batch.Put(SegKey(id), m.Encode()); batch.Put("cfg/next_segment", std::to_string(id + 1)); Commit(batch);
    catalog[id] = m; files[id] = std::move(f); ++next_segment;
    return id;
  }
  std::optional<Record> Find(const BlockKey& key) const {
    Validate(key); std::string value;
    const auto s = db->Get(rocksdb::ReadOptions(), "obj/" + key.id(), &value);
    if (s.IsNotFound()) return std::nullopt;
    Check(s); auto r = Record::Decode(value);
    if (r.parent != key.parent || r.length > options.max_object_bytes || !catalog.count(r.segment))
      throw Error(ErrorCode::corruption, "object identity or bounds disagree with manifest");
    const auto& m = catalog.at(r.segment);
    if ((m.state != 'a' && m.state != 's') || r.offset > m.used || r.length > m.used - r.offset)
      throw Error(ErrorCode::corruption, "object maps outside live committed segment");
    return r;
  }
  std::uint64_t Reap() {
    std::uint64_t bytes = 0;
    for (auto it = catalog.begin(); it != catalog.end();) {
      const auto id = it->first;
      if (it->second.state != 'r' || (files.count(id) && files.at(id).use_count() > 1)) { ++it; continue; }
      const auto path = data / SegmentName(id);
      if (fs::exists(path)) { bytes += fs::file_size(path); fs::remove(path); SyncDir(data); }
      rocksdb::WriteBatch batch; batch.Delete(SegKey(id)); Commit(batch);
      files.erase(id); it = catalog.erase(it);
    }
    reclaimed+=bytes; alloc_freed_+=bytes;
    return bytes;
  }
  void SealOne(std::uint64_t id) {
    auto m=catalog.at(id);if(m.state!='a')return;m.state='s';
    rocksdb::WriteBatch batch;batch.Put(SegKey(id),m.Encode());Commit(batch);catalog[id]=m;
    active_by_cohort.erase(m.cohort);last_write.erase(id);
  }
  void SealOldest() {
    if(last_write.empty())throw Error(ErrorCode::corruption,"active stream tracking is empty");
    const auto victim=std::min_element(last_write.begin(),last_write.end(),[](const auto& a,const auto& b){return a.second<b.second;})->first;
    SealOne(victim);
  }
  void SealActive() {
    rocksdb::WriteBatch batch;
    for(const auto& [cohort,id]:active_by_cohort){(void)cohort;auto m=catalog.at(id);m.state='s';batch.Put(SegKey(id),m.Encode());}
    if(!active_by_cohort.empty())Commit(batch);
    for(const auto& [cohort,id]:active_by_cohort){(void)cohort;catalog[id].state='s';}
    active_by_cohort.clear();last_write.clear();
  }
  std::uint64_t Writable(const std::string& cohort,std::uint64_t bytes) {
    auto it=active_by_cohort.find(cohort);
    if(it!=active_by_cohort.end()&&bytes>options.segment_bytes-catalog.at(it->second).used){SealOne(it->second);it=active_by_cohort.end();}
    std::uint64_t id=0;
    if(it==active_by_cohort.end()) {
      if(active_by_cohort.size()>=options.max_open_segments)SealOldest();
      id=Create('a',cohort);active_by_cohort[cohort]=id;
    } else id=it->second;
    last_write[id]=++clock;return id;
  }
};

Store::Store(const fs::path& root, Options options) : impl_(std::make_unique<Impl>(root, std::move(options))) {}
Store::~Store() = default;

std::uint64_t Store::Put(const BlockKey& key, std::string_view payload,const PlacementHint& hint) {
  auto& x = *impl_; std::lock_guard<std::mutex> lock(x.mutex); x.Ready(); Validate(key);
  if (payload.empty() || payload.size() > x.options.max_object_bytes) throw Error(ErrorCode::invalid, "payload must be nonempty and within max_object_bytes");
  const auto checksum = Digest(payload);
  const auto cohort=x.policy->Cohort(key.space,key.parent.empty()?key.digest:key.parent,hint);
  if (auto old = x.Find(key)) {
    if (old->length != payload.size() || old->checksum != checksum) throw Error(ErrorCode::conflict, "immutable object already exists with different payload");
    // Do not acknowledge a silently corrupted existing object as a successful put.
    if (Digest(ReadAll(x.files.at(old->segment)->fd, old->offset, old->length,
                       &x.counters->verify, x.options.direct_io)) != checksum)
      throw Error(ErrorCode::corruption, "existing payload checksum mismatch");
    return old->generation;
  }
  if (!x.Admit(key))
    throw Error(ErrorCode::declined, "prefix group has no reuse evidence; skip cache storage");
  x.Capacity(payload.size(), false);
  try {
    const auto id=x.Writable(cohort,payload.size());auto m=x.catalog.at(id);
    Record r{id, m.used, static_cast<std::uint64_t>(payload.size()), x.next_generation, checksum, key.parent};
    if (x.next_generation == std::numeric_limits<std::uint64_t>::max()) throw Error(ErrorCode::capacity, "generation exhausted");
    WriteAll(x.files.at(id)->fd, payload, r.offset,&x.counters->payload, x.options.direct_io); Sync(x.files.at(id)->fd); x.Fault("put_after_payload_sync");
    m.used += r.length;
    rocksdb::WriteBatch batch;
    batch.Put("obj/" + key.id(), r.Encode()); batch.Put(RefPrefix(id) + key.id(), ""); batch.Put(SegKey(id), m.Encode());
    batch.Put("cfg/next_generation", std::to_string(r.generation + 1)); x.Commit(batch);
    x.Fault("put_after_metadata_commit"); x.catalog[id] = m; ++x.next_generation;
    ++x.write_seq_; ++x.live_objects_; x.live_payload_ += r.length;
    x.alloc_added_ += ((r.length + 4095) / 4096) * 4096 + 4096;
    return r.generation;
  } catch (...) { x.poisoned = true; throw; }
}

std::optional<ReadTicket> Store::Pin(const BlockKey& key) {
  auto& x = *impl_; std::lock_guard<std::mutex> lock(x.mutex); x.Ready();
  auto r = x.Find(key); if (!r) return std::nullopt;
  x.NoteReuse(key);
  ReadTicket ticket; ticket.file_ = x.files.at(r->segment); ticket.offset_ = r->offset; ticket.length_ = r->length;
  ticket.generation_ = r->generation; ticket.checksum_ = r->checksum; ticket.direct_ = x.options.direct_io; return ticket;
}

std::vector<ReadTicket> Store::LookupPrefix(const std::vector<BlockKey>& chain) {
  auto& x = *impl_; std::lock_guard<std::mutex> lock(x.mutex); x.Ready();
  for (std::size_t i = 0; i < chain.size(); ++i) {
    Validate(chain[i]);
    if ((i == 0 && !chain[i].parent.empty()) || (i > 0 && (chain[i].space != chain[0].space || chain[i].parent != chain[i - 1].digest)))
      throw Error(ErrorCode::invalid, "lookup requires one ordered, root-anchored prefix chain");
  }
  std::vector<ReadTicket> result;
  std::uint64_t total_bytes = 0;
  for (const auto& key : chain) {
    auto r = x.Find(key); if (!r) break;
    x.NoteReuse(key);
    ReadTicket ticket; ticket.file_ = x.files.at(r->segment); ticket.offset_ = r->offset; ticket.length_ = r->length;
    ticket.generation_ = r->generation; ticket.checksum_ = r->checksum; ticket.direct_ = x.options.direct_io;
    total_bytes += r->length;
    result.push_back(std::move(ticket));
  }
  if (!result.empty() &&
      !x.ShouldRestore(total_bytes, static_cast<std::uint32_t>(result.size())))
    return {};
  return result;
}

std::vector<std::string> Store::ReadBatch(const std::vector<ReadTicket>& tickets,BatchReadOptions options) {
  auto& x=*impl_;
  if(options.max_batch_bytes==0||options.max_batch_bytes>64*1024*1024||options.max_gap_bytes>options.max_batch_bytes||
     options.max_total_output_bytes==0||options.max_total_output_bytes>256*1024*1024)
    throw Error(ErrorCode::invalid,"invalid bounded batch read options");
  std::uint64_t total=0;
  for(const auto& ticket:tickets) {
    if(!ticket.file_||ticket.file_->owner!=x.owner)throw Error(ErrorCode::invalid,"batch contains an empty or foreign-root ticket");
    if(ticket.length_>options.max_batch_bytes||ticket.length_>options.max_total_output_bytes-total)
      throw Error(ErrorCode::invalid,"batch exceeds byte limits");
    total+=ticket.length_;
  }
  std::vector<std::size_t> order(tickets.size());std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](std::size_t a,std::size_t b) {
    if(tickets[a].file_!=tickets[b].file_)return tickets[a].file_->path<tickets[b].file_->path;
    return tickets[a].offset_<tickets[b].offset_;
  });
  std::vector<std::string> result(tickets.size());
  for(std::size_t begin=0;begin<order.size();) {
    const auto& first=tickets[order[begin]];const auto start=first.offset_;
    auto end=start+first.length_;std::size_t next=begin+1;
    for(;next<order.size();++next) {
      const auto& item=tickets[order[next]];
      if(item.file_!=first.file_||(item.offset_>end&&item.offset_-end>options.max_gap_bytes))break;
      const auto proposed=std::max(end,item.offset_+item.length_);
      if(proposed-start>options.max_batch_bytes)break;
      end=proposed;
    }
    const auto bytes=ReadAll(first.file_->fd,start,end-start,&first.file_->counters->demand,first.direct_);
    for(auto i=begin;i<next;++i) {
      const auto index=order[i];const auto& item=tickets[index];
      auto piece=bytes.substr(static_cast<std::size_t>(item.offset_-start),static_cast<std::size_t>(item.length_));
      if(Digest(piece)!=item.checksum_)throw Error(ErrorCode::corruption,"batch payload SHA256 mismatch");
      result[index]=std::move(piece);
    }
    begin=next;
  }
  return result;
}

bool Store::Retire(const BlockKey& key) {
  auto& x = *impl_; std::lock_guard<std::mutex> lock(x.mutex); x.Ready();
  auto r = x.Find(key); if (!r) return false;
  try {
    rocksdb::WriteBatch batch; batch.Delete("obj/" + key.id()); batch.Delete(RefPrefix(r->segment) + key.id()); x.Commit(batch);
    --x.live_objects_; x.live_payload_ -= r->length; return true;
  } catch (...) { x.poisoned = true; throw; }
}
void Store::Seal() {
  auto& x = *impl_; std::lock_guard<std::mutex> lock(x.mutex); x.Ready();
  try { x.SealActive(); } catch (...) { x.poisoned = true; throw; }
}

bool Store::CollectGarbage() {
  auto& x = *impl_; std::lock_guard<std::mutex> lock(x.mutex); x.Ready();
  try {
    x.Reap();
    struct Live { std::string id; Record record; };
    std::uint64_t victim = 0; double best_score = 0.0;
    bool headroom_blocked=false;
    std::vector<Live> chosen;
    for (const auto& [id, m] : x.catalog) {
      if (m.state != 's' && m.state != 'a') continue;
      std::vector<Live> live; std::uint64_t size = 0;
      const auto prefix = RefPrefix(id);
      std::unique_ptr<rocksdb::Iterator> it(x.db->NewIterator(rocksdb::ReadOptions()));
      for (it->Seek(prefix); it->Valid() && Starts(it->key().ToStringView(), prefix); it->Next()) {
        const auto object_id = it->key().ToString().substr(prefix.size());
        std::string value; Check(x.db->Get(rocksdb::ReadOptions(), "obj/" + object_id, &value));
        auto r = Record::Decode(value);
        if (r.segment != id || r.offset > m.used || r.length > m.used - r.offset || r.length > x.options.max_object_bytes)
          throw Error(ErrorCode::corruption, "reverse index disagrees with object map");
        size += r.length; live.push_back({object_id, std::move(r)});
      }
      Check(it->status());
      if (size > m.used) throw Error(ErrorCode::corruption, "live bytes exceed segment bounds");
      const auto dead = m.used - size;
      if(dead&&size) {
        try {x.Capacity(size,true);}
        catch(const Error& e){if(e.code!=ErrorCode::capacity)throw;headroom_blocked=true;continue;}
      }
      // max-dead 与 v0.2.0 完全一致；min-cost 按「每搬一字节能腾出多少」评分。
      const double score = x.options.policy.gc_score == GcScore::min_cost
          ? static_cast<double>(dead) / (1.0 + static_cast<double>(size))
          : static_cast<double>(dead);
      if (score > best_score) { victim = id; best_score = score; chosen = std::move(live); }
    }
    if (!victim) {
      if(headroom_blocked)throw Error(ErrorCode::capacity,"no GC victim fits current copy headroom");
      return false;
    }
    if(x.catalog.at(victim).state=='a')x.SealOne(victim);
    std::uint64_t live_size = 0; for (const auto& v : chosen) live_size += v.record.length;
    if (!chosen.empty()) x.Capacity(live_size, true);
    std::uint64_t target = 0;
    const auto cohort=x.catalog.at(victim).cohort;
    if (!chosen.empty()) target = x.Create('t',cohort);
    std::sort(chosen.begin(),chosen.end(),[](const Live& a,const Live& b){return a.record.offset<b.record.offset;});
    std::uint64_t offset = 0;
    rocksdb::WriteBatch batch;
    for (auto& v : chosen) {
      auto bytes = ReadAll(x.files.at(victim)->fd, v.record.offset, v.record.length,
                           &x.counters->gc_read, x.options.direct_io);
      if (Digest(bytes) != v.record.checksum) throw Error(ErrorCode::corruption, "GC source checksum mismatch");
      WriteAll(x.files.at(target)->fd, bytes, offset,&x.counters->gc_write, x.options.direct_io);
      v.record.segment = target; v.record.offset = offset; offset += v.record.length;
      batch.Put("obj/" + v.id, v.record.Encode()); batch.Delete(RefPrefix(victim) + v.id); batch.Put(RefPrefix(target) + v.id, "");
    }
    if (target) Sync(x.files.at(target)->fd);
    x.Fault("gc_after_payload_sync");
    auto retired = x.catalog.at(victim); retired.state = 'r'; batch.Put(SegKey(victim), retired.Encode());
    if (target) batch.Put(SegKey(target), SegmentMeta{'s', offset,cohort}.Encode());
    x.Commit(batch); x.Fault("gc_after_metadata_commit");
    x.catalog[victim] = retired; if (target) x.catalog[target] = {'s', offset,cohort}; x.moved += live_size;
    if (x.options.gc_target_appendable && target) {
      // 让搬迁目标重新成为活动段：先封掉该 cohort 现有活动段，再持久化状态切换。
      // 两次提交之间崩溃时目标段仍是 sealed，行为退化为默认路径，不会产生悬空映射。
      auto current = x.active_by_cohort.find(cohort);
      if (current != x.active_by_cohort.end() && current->second != target) x.SealOne(current->second);
      SegmentMeta reopened = x.catalog.at(target); reopened.state = 'a';
      rocksdb::WriteBatch reopen; reopen.Put(SegKey(target), reopened.Encode()); x.Commit(reopen);
      x.catalog[target] = reopened; x.active_by_cohort[cohort] = target; x.last_write[target] = ++x.clock;
      while (x.active_by_cohort.size() > x.options.max_open_segments) x.SealOldest();
    }
    x.Reap(); return true;
  } catch (const Error& e) {
    // Lack of GC headroom is an expected admission outcome, not uncertain I/O.
    if (e.code != ErrorCode::capacity) x.poisoned = true;
    throw;
  } catch (...) { x.poisoned = true; throw; }
}

std::uint64_t Store::ReapRetired() {
  auto& x = *impl_; std::lock_guard<std::mutex> lock(x.mutex); x.Ready();
  try { return x.Reap(); } catch (...) { x.poisoned = true; throw; }
}

Stats Store::GetStats() const {
  auto& x = *impl_; std::lock_guard<std::mutex> lock(x.mutex); x.Ready(); Stats stats;
  stats.segments = x.catalog.size(); stats.gc_moved_bytes = x.moved;
  stats.active_segments=x.active_by_cohort.size();stats.reclaimed_file_bytes=x.reclaimed;
  stats.demand_read_calls=x.counters->demand.calls.load(std::memory_order_relaxed);
  stats.demand_read_bytes=x.counters->demand.bytes.load(std::memory_order_relaxed);
  stats.payload_write_calls=x.counters->payload.calls.load(std::memory_order_relaxed);
  stats.payload_write_bytes=x.counters->payload.bytes.load(std::memory_order_relaxed);
  stats.gc_read_calls=x.counters->gc_read.calls.load(std::memory_order_relaxed);
  stats.gc_read_bytes=x.counters->gc_read.bytes.load(std::memory_order_relaxed);
  stats.gc_write_calls=x.counters->gc_write.calls.load(std::memory_order_relaxed);
  stats.gc_write_bytes=x.counters->gc_write.bytes.load(std::memory_order_relaxed);
  stats.dedup_verify_read_calls=x.counters->verify.calls.load(std::memory_order_relaxed);
  stats.dedup_verify_read_bytes=x.counters->verify.bytes.load(std::memory_order_relaxed);
  for (const auto& [id, m] : x.catalog) { (void)id; stats.segment_bytes += m.used; if (m.state == 'r') stats.retired_segment_bytes += m.used; }
  stats.objects = x.live_objects_; stats.live_payload_bytes = x.live_payload_;
  stats.admission_declined = x.admission_declined_;
  stats.restore_declined_calls = x.declined_restores_;
  stats.restore_bytes_avoided = x.declined_bytes_;
  stats.allocated_bytes = x.AllocatedEstimate();
  return stats;
}

Stats Store::GetStatsExact() const {
  auto& x = *impl_; std::lock_guard<std::mutex> lock(x.mutex); x.Ready(); Stats stats;
  stats.segments = x.catalog.size(); stats.gc_moved_bytes = x.moved;
  stats.active_segments=x.active_by_cohort.size();stats.reclaimed_file_bytes=x.reclaimed;
  stats.demand_read_calls=x.counters->demand.calls.load(std::memory_order_relaxed);
  stats.demand_read_bytes=x.counters->demand.bytes.load(std::memory_order_relaxed);
  stats.payload_write_calls=x.counters->payload.calls.load(std::memory_order_relaxed);
  stats.payload_write_bytes=x.counters->payload.bytes.load(std::memory_order_relaxed);
  stats.gc_read_calls=x.counters->gc_read.calls.load(std::memory_order_relaxed);
  stats.gc_read_bytes=x.counters->gc_read.bytes.load(std::memory_order_relaxed);
  stats.gc_write_calls=x.counters->gc_write.calls.load(std::memory_order_relaxed);
  stats.gc_write_bytes=x.counters->gc_write.bytes.load(std::memory_order_relaxed);
  stats.dedup_verify_read_calls=x.counters->verify.calls.load(std::memory_order_relaxed);
  stats.dedup_verify_read_bytes=x.counters->verify.bytes.load(std::memory_order_relaxed);
  for (const auto& [id, m] : x.catalog) { (void)id; stats.segment_bytes += m.used; if (m.state == 'r') stats.retired_segment_bytes += m.used; }
  std::unique_ptr<rocksdb::Iterator> it(x.db->NewIterator(rocksdb::ReadOptions()));
  for (it->Seek("obj/"); it->Valid() && Starts(it->key().ToStringView(), "obj/"); it->Next()) {
    ++stats.objects; stats.live_payload_bytes += Record::Decode(it->value().ToString()).length;
  }
  Check(it->status());
  stats.admission_declined = x.admission_declined_;
  stats.restore_declined_calls = x.declined_restores_;
  stats.restore_bytes_avoided = x.declined_bytes_;
  stats.allocated_bytes = Allocated(x.root);
  return stats;
}
}  // namespace kvcohort
