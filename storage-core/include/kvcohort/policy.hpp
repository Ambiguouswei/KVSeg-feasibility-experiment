#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace kvcohort {

enum class Layout { append, prefix, lifetime, joint };
enum class Lifetime { unknown, short_lived, long_lived };
// GC 候选打分方式。max_dead = v0.2.0 行为（按无效字节最多选）；
// min_cost = 按「平均每搬一字节能腾出多少空间」选，偏向搬移成本低的段。
enum class GcScore { max_dead, min_cost };

struct PlacementHint {
  // Optional stable ancestor digest, chosen by the adapter, not a pathname.
  std::string prefix_group;
  Lifetime lifetime = Lifetime::unknown;
};

std::string LayoutName(Layout layout);
Layout ParseLayout(std::string_view name);
std::string LifetimeName(Lifetime lifetime);
Lifetime ParseLifetime(std::string_view name);
std::string GcScoreName(GcScore score);
GcScore ParseGcScore(std::string_view name);

// 决策参数集中定义，每项都有明确含义与单位。
//
// 约定（重要）：默认值必须精确复现 v0.2.0 的行为。任何新增机制默认关闭，
// 于是「打开某个机制」本身构成它自己的对照组，消融实验才有意义。
struct PolicyConfig {
  // ---- 准入（存入）----
  // 依据是该对象所属前缀组此前被命中过的次数：没有复用证据的数据不落盘。
  // 证据表是有界近似（见 admit_evidence_capacity），不承诺精确计数。
  bool admit_enabled = false;
  std::uint32_t admit_min_reuse = 1;
  std::uint64_t admit_evidence_capacity = 65536;

  // ---- 读取决策（恢复 vs 重算）----
  // 两侧都必须有实测值才能比较；任一为 0 表示未标定，退回「能恢复就恢复」。
  // restore_bytes_per_us : 实测恢复带宽（字节/微秒）
  // prefill_us_per_token  : 实测重算速率（微秒/token）
  bool restore_decision_enabled = false;
  double restore_bytes_per_us = 0.0;
  double prefill_us_per_token = 0.0;
  double restore_queue_us = 0.0;
  std::uint32_t tokens_per_block = 32;

  // ---- GC 评分 ----
  GcScore gc_score = GcScore::max_dead;
};

class PlacementPolicy {
 public:
  virtual ~PlacementPolicy() = default;
  virtual std::string name() const = 0;
  // All strategies isolate physical streams by namespace. They differ only in
  // whether prefix and lifetime labels participate in the cohort key.
  virtual std::string Cohort(std::string_view space, std::string_view default_group,
                             const PlacementHint& hint) const = 0;
};

std::unique_ptr<PlacementPolicy> MakePlacementPolicy(Layout layout);

}  // namespace kvcohort
