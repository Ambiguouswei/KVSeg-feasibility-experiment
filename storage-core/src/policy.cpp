#include "kvcohort/policy.hpp"
#include "kvcohort/store.hpp"

#include <algorithm>

namespace kvcohort {
std::string LayoutName(Layout layout) {
  switch (layout) {
    case Layout::append: return "append";
    case Layout::prefix: return "prefix";
    case Layout::lifetime: return "lifetime";
    case Layout::joint: return "joint";
  }
  throw Error(ErrorCode::invalid, "invalid layout enum");
}
Layout ParseLayout(std::string_view name) {
  for (auto layout : {Layout::append, Layout::prefix, Layout::lifetime, Layout::joint})
    if (LayoutName(layout) == name) return layout;
  throw Error(ErrorCode::invalid, "layout must be append, prefix, lifetime or joint");
}
std::string LifetimeName(Lifetime lifetime) {
  switch (lifetime) {
    case Lifetime::unknown: return "unknown";
    case Lifetime::short_lived: return "short";
    case Lifetime::long_lived: return "long";
  }
  throw Error(ErrorCode::invalid, "invalid lifetime enum");
}
Lifetime ParseLifetime(std::string_view name) {
  for (auto lifetime : {Lifetime::unknown, Lifetime::short_lived, Lifetime::long_lived})
    if (LifetimeName(lifetime) == name) return lifetime;
  throw Error(ErrorCode::invalid, "lifetime must be unknown, short or long");
}
std::string GcScoreName(GcScore score) {
  switch (score) {
    case GcScore::max_dead: return "max-dead";
    case GcScore::min_cost: return "min-cost";
  }
  throw Error(ErrorCode::invalid, "invalid gc score enum");
}
GcScore ParseGcScore(std::string_view name) {
  for (auto score : {GcScore::max_dead, GcScore::min_cost})
    if (GcScoreName(score) == name) return score;
  throw Error(ErrorCode::invalid, "gc score must be max-dead or min-cost");
}
namespace {
bool Digest(std::string_view text) {
  return text.size() == 64 && std::all_of(text.begin(),text.end(),[](char c) {
    return (c>='0'&&c<='9')||(c>='a'&&c<='f');
  });
}
class GroupedPolicy final : public PlacementPolicy {
 public:
  explicit GroupedPolicy(Layout layout) : layout_(layout) {}
  std::string name() const override { return LayoutName(layout_); }
  std::string Cohort(std::string_view space,std::string_view fallback,const PlacementHint& hint) const override {
    if (!Digest(space) || !Digest(fallback) || (!hint.prefix_group.empty() && !Digest(hint.prefix_group)))
      throw Error(ErrorCode::invalid,"placement namespace and group must be SHA256 digests");
    const auto life=LifetimeName(hint.lifetime);
    std::string result(space);result+="/"+name();
    if(layout_==Layout::prefix||layout_==Layout::joint)result+="/"+(hint.prefix_group.empty()?std::string(fallback):hint.prefix_group);
    if(layout_==Layout::lifetime||layout_==Layout::joint)result+="/"+life;
    return result;
  }
 private:
  Layout layout_;
};
}  // namespace
std::unique_ptr<PlacementPolicy> MakePlacementPolicy(Layout layout) {
  (void)LayoutName(layout);return std::make_unique<GroupedPolicy>(layout);
}
}  // namespace kvcohort
