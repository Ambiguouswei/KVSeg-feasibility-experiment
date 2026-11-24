#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

// 系统头必须出现在全局作用域：放进 namespace 会把所有 POSIX 声明嵌进来。
#if defined(__unix__) || defined(__APPLE__)
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace kvcohort {
namespace proto {

// 线上协议版本。改变帧布局时必须 +1，且客户端与服务端各自校验。
constexpr std::uint8_t kVersion = 1;

// 单帧上限，防止畸形长度头导致的无界分配。
constexpr std::uint32_t kMaxFrame = 64u * 1024u * 1024u;

enum class Op : std::uint8_t {
  ping = 1,
  put = 2,
  get = 3,
  lookup = 4,
  retire = 5,
  seal = 6,
  gc = 7,
  stats = 8,
  shutdown = 9,
};

// 与 kvcohort::ErrorCode 一一对应；ok 之外的取值直接映射回异常。
enum class Status : std::uint8_t {
  ok = 0,
  invalid = 1,
  missing = 2,
  conflict = 3,
  capacity = 4,
  corruption = 5,
  busy = 6,
  io = 7,
  declined = 8,
};

struct Error : std::runtime_error {
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

class Writer {
 public:
  void u8(std::uint8_t value) { buf_.push_back(static_cast<char>(value)); }
  void u32(std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) buf_.push_back(static_cast<char>((value >> (8 * i)) & 0xFFu));
  }
  void u64(std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) buf_.push_back(static_cast<char>((value >> (8 * i)) & 0xFFu));
  }
  void str(std::string_view text) {
    if (text.size() > 0xFFFFFFFFull) throw Error("string too long for wire format");
    u32(static_cast<std::uint32_t>(text.size()));
    buf_.append(text.data(), text.size());
  }
  void raw(std::string_view bytes) { buf_.append(bytes.data(), bytes.size()); }
  const std::string& bytes() const { return buf_; }
  std::string take() { return std::move(buf_); }

 private:
  std::string buf_;
};

class Reader {
 public:
  explicit Reader(std::string_view data) : data_(data) {}

  std::uint8_t u8() {
    need(1);
    return static_cast<std::uint8_t>(data_[pos_++]);
  }
  std::uint32_t u32() {
    need(4);
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i)
      value |= static_cast<std::uint32_t>(static_cast<unsigned char>(data_[pos_ + i])) << (8 * i);
    pos_ += 4;
    return value;
  }
  std::uint64_t u64() {
    need(8);
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i)
      value |= static_cast<std::uint64_t>(static_cast<unsigned char>(data_[pos_ + i])) << (8 * i);
    pos_ += 8;
    return value;
  }
  std::string_view str() {
    const auto length = u32();
    need(length);
    auto view = data_.substr(pos_, length);
    pos_ += length;
    return view;
  }
  std::string_view rest() {
    auto view = data_.substr(pos_);
    pos_ = data_.size();
    return view;
  }
  bool done() const { return pos_ == data_.size(); }
  std::size_t remaining() const { return data_.size() - pos_; }

 private:
  void need(std::size_t count) const {
    if (count > data_.size() - pos_) throw Error("truncated frame payload");
  }
  std::string_view data_;
  std::size_t pos_ = 0;
};

// 帧布局：[u32 body_len][u8 status][body...]
// body_len 只计 status + body，便于先分发再解析。
inline std::string EncodeFrame(Status status, std::string_view body) {
  Writer out;
  out.u32(static_cast<std::uint32_t>(body.size() + 1));
  out.u8(static_cast<std::uint8_t>(status));
  out.raw(body);
  return out.take();
}

#if defined(__unix__) || defined(__APPLE__)
// 阻塞读写的完整收发。返回 false 表示对端在读完之前关闭了连接。
inline bool SendAll(int fd, std::string_view data) {
  std::size_t sent = 0;
  while (sent < data.size()) {
    const auto n = ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
    if (n > 0) { sent += static_cast<std::size_t>(n); continue; }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

inline bool RecvAll(int fd, void* buffer, std::size_t count) {
  auto* out = static_cast<char*>(buffer);
  std::size_t got = 0;
  while (got < count) {
    const auto n = ::recv(fd, out + got, count - got, 0);
    if (n > 0) { got += static_cast<std::size_t>(n); continue; }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}
#endif

}  // namespace proto
}  // namespace kvcohort
