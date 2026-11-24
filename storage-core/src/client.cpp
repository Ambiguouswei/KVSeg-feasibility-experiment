#include "kvcohort/client.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

namespace kvcohort {
namespace {

[[noreturn]] void IoFail(const std::string& what) {
  throw Error(ErrorCode::io, what + ": " + std::strerror(errno));
}

ErrorCode MapStatus(proto::Status status) {
  switch (status) {
    case proto::Status::invalid: return ErrorCode::invalid;
    case proto::Status::missing: return ErrorCode::missing;
    case proto::Status::conflict: return ErrorCode::conflict;
    case proto::Status::capacity: return ErrorCode::capacity;
    case proto::Status::corruption: return ErrorCode::corruption;
    case proto::Status::busy: return ErrorCode::busy;
    case proto::Status::io: return ErrorCode::io;
    case proto::Status::declined: return ErrorCode::declined;
    case proto::Status::ok: break;
  }
  return ErrorCode::io;
}

void WriteKey(proto::Writer& out, const BlockKey& key) {
  out.str(key.space);
  out.str(key.digest);
  out.str(key.parent);
}

}  // namespace

Client::Client(const std::string& socket_path) {
  fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd_ < 0) IoFail("create client socket");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(address.sun_path))
    throw Error(ErrorCode::invalid, "socket path exceeds AF_UNIX limit");
  std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
  if (::connect(fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
    const int saved = errno;
    ::close(fd_);
    fd_ = -1;
    errno = saved;
    IoFail("connect to " + socket_path);
  }
}

Client::~Client() {
  if (fd_ >= 0) ::close(fd_);
}

std::string Client::Call(proto::Op op, std::string_view body) {
  if (fd_ < 0) throw Error(ErrorCode::io, "client is not connected");

  proto::Writer request;
  request.u32(static_cast<std::uint32_t>(body.size() + 2));
  request.u8(proto::kVersion);
  request.u8(static_cast<std::uint8_t>(op));
  request.raw(body);
  if (!proto::SendAll(fd_, request.bytes()))
    throw Error(ErrorCode::io, "failed to send request");

  std::uint8_t header[4];
  if (!proto::RecvAll(fd_, header, sizeof(header)))
    throw Error(ErrorCode::io, "connection closed while reading response header");
  proto::Reader header_reader(std::string_view(reinterpret_cast<const char*>(header), sizeof(header)));
  const auto length = header_reader.u32();
  if (length == 0 || length > proto::kMaxFrame)
    throw Error(ErrorCode::corruption, "invalid response frame length");

  std::string payload(length, '\0');
  if (!proto::RecvAll(fd_, payload.data(), payload.size()))
    throw Error(ErrorCode::io, "connection closed while reading response body");

  const auto status = static_cast<proto::Status>(static_cast<std::uint8_t>(payload[0]));
  std::string_view rest(payload.data() + 1, payload.size() - 1);
  if (status != proto::Status::ok) throw Error(MapStatus(status), std::string(rest));
  return std::string(rest);
}

void Client::Ping() { Call(proto::Op::ping, {}); }

std::uint64_t Client::Put(const BlockKey& key, std::string_view payload,
                          const PlacementHint& hint) {
  proto::Writer out;
  WriteKey(out, key);
  out.u8(static_cast<std::uint8_t>(hint.lifetime));
  out.str(hint.prefix_group);
  out.str(payload);
  const auto response = Call(proto::Op::put, out.bytes());
  proto::Reader in(response);
  return in.u64();
}

std::string Client::Get(const BlockKey& key) {
  proto::Writer out;
  WriteKey(out, key);
  return Call(proto::Op::get, out.bytes());
}

std::vector<std::string> Client::LookupPrefix(const std::vector<BlockKey>& chain) {
  proto::Writer out;
  out.u32(static_cast<std::uint32_t>(chain.size()));
  for (const auto& key : chain) WriteKey(out, key);
  const auto response = Call(proto::Op::lookup, out.bytes());
  proto::Reader in(response);
  const auto count = in.u32();
  std::vector<std::string> result;
  result.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) result.emplace_back(in.str());
  return result;
}

bool Client::Retire(const BlockKey& key) {
  proto::Writer out;
  WriteKey(out, key);
  const auto response = Call(proto::Op::retire, out.bytes());
  proto::Reader in(response);
  return in.u8() != 0;
}

void Client::Seal() { Call(proto::Op::seal, {}); }

bool Client::CollectGarbage() {
  const auto response = Call(proto::Op::gc, {});
  proto::Reader in(response);
  return in.u8() != 0;
}

std::string Client::Stats() {
  const auto response = Call(proto::Op::stats, {});
  proto::Reader in(response);
  return std::string(in.str());
}

void Client::Shutdown() {
  Call(proto::Op::shutdown, {});
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

}  // namespace kvcohort
