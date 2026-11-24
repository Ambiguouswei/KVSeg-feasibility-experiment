#pragma once

#include "kvcohort/protocol.hpp"
#include "kvcohort/store.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kvcohort {

// 通过 Unix 域套接字访问守护进程的客户端。
// 语义与进程内 Store 一致，但每次调用都是一次完整往返：服务端在同一请求内
// 完成「查询 + 取得票据 + 读取 + 释放票据」，因此票据不跨请求存活。
class Client {
 public:
  explicit Client(const std::string& socket_path);
  ~Client();
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  void Ping();
  // 成功返回代次；同键不同内容抛 conflict，准入拒绝抛 declined。
  std::uint64_t Put(const BlockKey& key, std::string_view payload, const PlacementHint& hint = {});
  // 未命中抛 missing。
  std::string Get(const BlockKey& key);
  // 返回从根开始的连续命中段；遇到首个缺口即止，长度可能小于 chain。
  std::vector<std::string> LookupPrefix(const std::vector<BlockKey>& chain);
  bool Retire(const BlockKey& key);
  void Seal();
  bool CollectGarbage();
  // 服务端已格式化为 JSON 一行，避免客户端解析依赖。
  std::string Stats();
  void Shutdown();

  int fd() const { return fd_; }

 private:
  std::string Call(proto::Op op, std::string_view body);
  int fd_ = -1;
};

}  // namespace kvcohort
