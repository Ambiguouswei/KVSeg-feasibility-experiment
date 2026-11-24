#include "kvcohort/store.hpp"

#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;
namespace {
void Require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void Expect(kvcohort::ErrorCode code, F f) {
  try { f(); } catch (const kvcohort::Error& e) { Require(e.code == code, e.what()); return; }
  throw std::runtime_error("expected kvc error was not raised");
}
struct Temp {
  fs::path base, root;
  Temp() {
    const char* setting = std::getenv("KVCOHORT_TEST_ROOT");
    if (!setting || !*setting) throw std::runtime_error("KVCOHORT_TEST_ROOT must name a dedicated test directory");
    auto parent = fs::absolute(setting); fs::create_directories(parent);
    auto pattern = (parent / "case-XXXXXX").string();
    if (!::mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
    base = pattern; root = base / "store";
  }
  ~Temp() { std::error_code ignored; fs::remove_all(base, ignored); }
};
kvcohort::Options Options() {
  kvcohort::Options o; o.segment_bytes = 32768; o.max_object_bytes = 16384;
  o.capacity_bytes = 8 * 1024 * 1024; o.gc_reserve_bytes = 32768; o.metadata_reserve_bytes = 512 * 1024; return o;
}
std::string Ns(std::string_view tenant = "tenant") { return kvcohort::MakeNamespace(tenant, "model@sha256:exact", "fp16-kv-layout-v1-tp1"); }
kvcohort::BlockKey Key(std::uint32_t token) { return kvcohort::MakeBlockKey(Ns(), "", {token}); }

void IdentityAndImmutable() {
  Temp t; kvcohort::Store s(t.root, Options()); auto k = Key(1);
  const std::string bytes("binary\0payload", 14);
  const auto generation = s.Put(k, bytes);
  Require(s.Put(k, bytes) == generation, "duplicate put must be idempotent");
  Require(s.Pin(k)->Read() == bytes, "binary payload must round trip");
  Expect(kvcohort::ErrorCode::conflict, [&] { s.Put(k, "different"); });
  auto other = kvcohort::MakeBlockKey(Ns("other-tenant"), "", {1});
  Require(!s.Pin(other), "tenant identity must isolate objects");
  auto layout = kvcohort::MakeNamespace("tenant", "model@sha256:exact", "fp8-layout-v1-tp1");
  Require(!s.Pin(kvcohort::MakeBlockKey(layout, "", {1})), "representation must isolate objects");
  auto model = kvcohort::MakeNamespace("tenant", "model@sha256:other", "fp16-kv-layout-v1-tp1");
  Require(!s.Pin(kvcohort::MakeBlockKey(model, "", {1})), "model revision must isolate objects");
  Expect(kvcohort::ErrorCode::invalid, [&] { s.Put(kvcohort::BlockKey{"../../escape","x",""}, "x"); });
}
void ContiguousPrefix() {
  Temp t; kvcohort::Store s(t.root, Options());
  auto p = Key(1); auto a = kvcohort::MakeBlockKey(Ns(), p.digest, {2}); auto b = kvcohort::MakeBlockKey(Ns(), a.digest, {3});
  s.Put(p, "P"); s.Put(b, "B");
  auto tickets = s.LookupPrefix({p,a,b}); Require(tickets.size() == 1, "must stop at missing middle block");
  s.Put(a, "A"); tickets = s.LookupPrefix({p,a,b}); Require(tickets.size() == 3, "complete chain must hit");
  Expect(kvcohort::ErrorCode::invalid, [&] { s.LookupPrefix({p,b}); });
  auto wrong_context = kvcohort::MakeBlockKey(Ns(), Key(4).digest, {2});
  Require(wrong_context.digest != a.digest, "parent context must affect identity");
}
void ReopenAndExclusiveOwner() {
  Temp t;
  { kvcohort::Store s(t.root, Options()); s.Put(Key(1), "committed");
    Expect(kvcohort::ErrorCode::busy, [&] { kvcohort::Store second(t.root, Options()); }); }
  { kvcohort::Store s(t.root, Options()); Require(s.Pin(Key(1))->Read() == "committed", "durable data lost on reopen"); }
  std::optional<kvcohort::ReadTicket> ticket;
  { kvcohort::Store s(t.root, Options()); ticket = s.Pin(Key(1)); }
  Require(ticket->Read() == "committed", "ticket must remain usable after Store destruction");
  Expect(kvcohort::ErrorCode::busy, [&] { kvcohort::Store s(t.root, Options()); });
  ticket.reset(); kvcohort::Store s(t.root, Options());
}
void PinnedRetirementAndGC() {
  Temp t; kvcohort::Store s(t.root, Options());
  const std::string a(4096, 'A'), b(4096, 'B');
  auto first_generation = s.Put(Key(1), a); s.Put(Key(2), b); s.Seal();
  auto old = s.Pin(Key(1)); Require(s.Retire(Key(1)), "retire existing");
  Require(!s.Pin(Key(1)), "retired object must disappear from new lookups");
  Require(s.CollectGarbage(), "mixed live/dead segment must collect");
  Require(old->Read() == a, "GC invalidated an in-flight read");
  Require(s.Pin(Key(2))->Read() == b, "GC failed to relocate live object");
  Require(s.GetStats().retired_segment_bytes == 8192, "pinned retired segment not accounted");
  auto next_generation = s.Put(Key(1), "new incarnation");
  Require(next_generation > first_generation && old->Read() == a, "generation reuse invalidated stale ticket");
  old.reset(); Require(s.ReapRetired() == 8192, "old segment not freed after pin release");
  Require(s.GetStats().retired_segment_bytes == 0, "retired accounting not cleared");
  Require(s.GetStats().gc_moved_bytes == 4096, "GC moved byte counter incorrect");
}
void ConcurrentPinnedRead() {
  Temp t; kvcohort::Store s(t.root, Options()); const std::string payload(16384, 'R');
  s.Put(Key(1), payload); auto ticket = s.Pin(Key(1)); std::exception_ptr reader_error;
  std::thread reader([&] { try { for (int i=0;i<100;++i) Require(ticket->Read() == payload, "concurrent pinned data changed"); } catch (...) { reader_error=std::current_exception(); } });
  s.Retire(Key(1)); s.CollectGarbage(); reader.join(); if(reader_error) std::rethrow_exception(reader_error);
  ticket.reset(); s.ReapRetired(); Require(s.GetStats().objects == 0, "retired object resurrected");
}
void ChecksumDetectsCorruption() {
  Temp t; kvcohort::Store s(t.root, Options()); s.Put(Key(1), "correct"); auto ticket = s.Pin(Key(1));
  auto segment = *fs::directory_iterator(t.root / "segments"); int fd = ::open(segment.path().c_str(), O_WRONLY);
  Require(fd >= 0, "cannot open test segment"); Require(::pwrite(fd, "X", 1, 0) == 1, "corruption injection failed"); ::close(fd);
  Expect(kvcohort::ErrorCode::corruption, [&] { ticket->Read(); });
  Expect(kvcohort::ErrorCode::corruption, [&] { s.Put(Key(1), "correct"); });
}
void RefusesForeignDirectory() {
  Temp t; fs::create_directory(t.root);
  auto file = t.root / "existing-checkpoint"; int fd = ::open(file.c_str(), O_CREAT | O_WRONLY, 0600); ::write(fd, "keep", 4); ::close(fd);
  Expect(kvcohort::ErrorCode::invalid, [&] { kvcohort::Store s(t.root, Options()); });
  Require(fs::file_size(file) == 4, "foreign content was modified");
  auto link = t.base / "link"; fs::create_directory_symlink(t.root, link);
  Expect(kvcohort::ErrorCode::invalid, [&] { kvcohort::Store s(link, Options()); });
}
void AdmissionBudget() {
  Temp t; auto options = Options(); options.capacity_bytes = 8 * 1024 * 1024;
  kvcohort::Store s(t.root, options); std::vector<kvcohort::BlockKey> keys; bool rejected = false;
  for (std::uint32_t i=0;i<500;++i) {
    try { auto k=Key(i); s.Put(k, std::string(16384,'Q')); keys.push_back(k); }
    catch(const kvcohort::Error& e) { Require(e.code == kvcohort::ErrorCode::capacity, "unexpected budget error"); rejected=true; break; }
  }
  if (!rejected) {
    const auto stats=s.GetStats();
    throw std::runtime_error("budget did not reject: objects="+std::to_string(stats.objects)+", segment_bytes="+std::to_string(stats.segment_bytes)+", allocated="+std::to_string(stats.allocated_bytes));
  }
  Require(!keys.empty(), "budget must admit at least one object");
  Require(s.Pin(keys.front())->Read() == std::string(16384,'Q'), "budget rejection broke existing read");
  const auto before = s.GetStats().segment_bytes;
  for (const auto& key : keys) s.Retire(key);
  while(s.CollectGarbage()) {}
  Require(s.GetStats().objects == 0 && s.GetStats().segment_bytes < before, "full dead segments not reclaimed");
}

void GCPressurePreservesReads() {
  Temp t; kvcohort::Store s(t.root,Options());
  s.Put(Key(1),std::string(4096,'A'));s.Put(Key(2),std::string(4096,'B'));s.Seal();s.Retire(Key(1));
  const auto pressure=t.root/"test-pressure-reservation";
  const int fd=::open(pressure.c_str(),O_CREAT|O_RDWR|O_EXCL,0600);Require(fd>=0,"pressure file open failed");
  const int allocation=::posix_fallocate(fd,0,8*1024*1024);::close(fd);Require(allocation==0,"pressure reservation failed");
  Expect(kvcohort::ErrorCode::capacity,[&]{s.CollectGarbage();});
  Require(s.Pin(Key(2))->Read()==std::string(4096,'B'),"GC capacity rejection broke reads");
  fs::remove(pressure);
  Require(s.CollectGarbage(),"GC failed after reservation released");
}

void RandomizedStateMachineFor(kvcohort::Layout layout) {
  Temp t;auto opts=Options();opts.layout=layout;opts.capacity_bytes=16*1024*1024;
  auto s=std::make_unique<kvcohort::Store>(t.root,opts);std::mt19937 random(0xCAFE);
  std::map<std::uint32_t,std::string> expected;
  std::vector<std::pair<kvcohort::ReadTicket,std::string>> pinned;
  for(int step=0;step<500;++step) {
    const auto id=static_cast<std::uint32_t>(random()%40);const auto key=Key(id);const auto action=random()%5;
    if(action<=1) {
      auto it=expected.find(id);
      const auto value=it==expected.end()?std::string(1024+id,static_cast<char>('a'+step%26)):it->second;
      s->Put(key,value,{key.digest,id%2?kvcohort::Lifetime::short_lived:kvcohort::Lifetime::long_lived});expected[id]=value;
    } else if(action==2) {
      Require(s->Retire(key)==(expected.erase(id)>0),"retirement disagrees with reference model");
    } else if(action==3) {
      s->CollectGarbage();
    } else {
      auto ticket=s->Pin(key);Require(static_cast<bool>(ticket)==(expected.count(id)>0),"lookup disagrees with reference model");
      if(ticket)pinned.emplace_back(*ticket,expected.at(id));
    }
    if(step%25==0) {
      for(const auto& [ticket,value]:pinned)Require(ticket.Read()==value,"old generation changed during mutation sequence");
      for(const auto& [token,value]:expected)Require(s->Pin(Key(token))->Read()==value,"live object disagrees with reference model");
      Require(s->GetStats().objects==expected.size(),"object count disagrees with reference model");
      pinned.clear();s->ReapRetired();
    }
    if(step%100==99) {pinned.clear();s.reset();s=std::make_unique<kvcohort::Store>(t.root,opts);}
  }
  for(const auto& [token,value]:expected)Require(s->Pin(Key(token))->Read()==value,"reopen lost reference-model object");
}
void RandomizedStateMachine() {
  for(auto layout:{kvcohort::Layout::append,kvcohort::Layout::prefix,kvcohort::Layout::lifetime,kvcohort::Layout::joint})RandomizedStateMachineFor(layout);
}

void PolicyCohorts() {
  const auto space=Ns(),group=Key(1).digest,other=Key(2).digest;
  kvcohort::PlacementHint a{group,kvcohort::Lifetime::short_lived},b{group,kvcohort::Lifetime::long_lived},c{other,kvcohort::Lifetime::short_lived};
  for(auto layout:{kvcohort::Layout::append,kvcohort::Layout::prefix,kvcohort::Layout::lifetime,kvcohort::Layout::joint}) {
    auto p=kvcohort::MakePlacementPolicy(layout);
    const auto ca=p->Cohort(space,group,a),cb=p->Cohort(space,group,b),cc=p->Cohort(space,group,c);
    Require((ca!=cb)==(layout==kvcohort::Layout::lifetime||layout==kvcohort::Layout::joint),"lifetime dimension not isolated correctly");
    Require((ca!=cc)==(layout==kvcohort::Layout::prefix||layout==kvcohort::Layout::joint),"prefix dimension not isolated correctly");
    Require(ca!=p->Cohort(Ns("other"),group,a),"physical stream crosses namespaces");
    Expect(kvcohort::ErrorCode::invalid,[&]{p->Cohort(space,group,{"../../outside",kvcohort::Lifetime::unknown});});
    Require(kvcohort::ParseLayout(p->name())==layout,"policy name is not round trippable");
  }
  Expect(kvcohort::ErrorCode::invalid,[]{kvcohort::ParseLayout("not-a-policy");});
}

void LayoutReopenAndBound() {
  for(auto layout:{kvcohort::Layout::append,kvcohort::Layout::prefix,kvcohort::Layout::lifetime,kvcohort::Layout::joint}) {
    Temp t;auto opts=Options();opts.layout=layout;opts.max_open_segments=2;
    {
      kvcohort::Store s(t.root,opts);
      for(std::uint32_t i=1;i<=12;++i) {
        auto key=Key(i);s.Put(key,std::string(256,static_cast<char>(i)),{key.digest,i%2?kvcohort::Lifetime::short_lived:kvcohort::Lifetime::long_lived});
        Require(s.GetStats().active_segments<=2,"active cohort limit exceeded");
      }
    }
    {
      kvcohort::Store s(t.root,opts);
      for(std::uint32_t i=1;i<=12;++i)Require(s.Pin(Key(i))->Read()==std::string(256,static_cast<char>(i)),"policy reopen lost data");
      Require(s.GetStats().active_segments<=2,"reopen violated active cohort limit");
    }
    auto wrong=opts;wrong.layout=layout==kvcohort::Layout::append?kvcohort::Layout::joint:kvcohort::Layout::append;
    Expect(kvcohort::ErrorCode::invalid,[&]{kvcohort::Store s(t.root,wrong);});
    opts.max_open_segments=1;kvcohort::Store s(t.root,opts);Require(s.GetStats().active_segments<=1,"smaller reopen limit not enforced");
  }
}

void BatchedReadOrderAndLimits() {
  Temp t;kvcohort::Store s(t.root,Options());
  for(std::uint32_t i=1;i<=3;++i)s.Put(Key(i),std::string(1024,static_cast<char>('A'+i)));
  auto a=s.Pin(Key(1)),b=s.Pin(Key(2)),c=s.Pin(Key(3));
  auto result=s.ReadBatch({*c,*a,*b});
  Require(result==std::vector<std::string>{std::string(1024,'D'),std::string(1024,'B'),std::string(1024,'C')},"batch did not preserve logical order");
  auto stats=s.GetStats();Require(stats.demand_read_calls==1&&stats.demand_read_bytes==3072,"adjacent objects did not coalesce");
  result=s.ReadBatch({*a,*a});Require(result[0]==result[1],"duplicate-ticket extraction failed");
  Require(s.GetStats().demand_read_calls==2,"duplicate-ticket batch read twice");
  kvcohort::BatchReadOptions limited;limited.max_total_output_bytes=1024;
  Expect(kvcohort::ErrorCode::invalid,[&]{s.ReadBatch({*a,*b},limited);});
  limited={};limited.max_batch_bytes=512;
  Expect(kvcohort::ErrorCode::invalid,[&]{s.ReadBatch({*a},limited);});
  Expect(kvcohort::ErrorCode::invalid,[&]{s.ReadBatch({kvcohort::ReadTicket{}});});
  Temp other;kvcohort::Store foreign(other.root,Options());foreign.Put(Key(1),"other root");
  Expect(kvcohort::ErrorCode::invalid,[&]{s.ReadBatch({*foreign.Pin(Key(1))});});
  s.Retire(Key(1));s.CollectGarbage();Require(s.ReadBatch({*a})[0]==std::string(1024,'B'),"batch lost pinned old range during GC");
}

void BatchGapAmplification() {
  Temp t;kvcohort::Store s(t.root,Options());
  for(std::uint32_t i=1;i<=3;++i)s.Put(Key(i),std::string(1024,static_cast<char>(i)));
  s.Retire(Key(2));auto a=s.Pin(Key(1)),c=s.Pin(Key(3));
  s.ReadBatch({*a,*c});auto before=s.GetStats();
  Require(before.demand_read_calls==2&&before.demand_read_bytes==2048,"zero-gap batch read unrelated bytes");
  kvcohort::BatchReadOptions opts;opts.max_gap_bytes=1024;s.ReadBatch({*a,*c},opts);
  auto after=s.GetStats();Require(after.demand_read_calls-before.demand_read_calls==1&&after.demand_read_bytes-before.demand_read_bytes==3072,"bounded over-read accounting is wrong");
  before=after;opts.max_batch_bytes=2048;s.ReadBatch({*a,*c},opts);after=s.GetStats();
  Require(after.demand_read_calls-before.demand_read_calls==2,"max batch size did not bound coalescing");
}

void PayloadDebtAndGCReserve() {
  Temp t;auto opts=Options();opts.payload_budget_bytes=3*32768;kvcohort::Store s(t.root,opts);
  for(std::uint32_t i=1;i<=4;++i)s.Put(Key(i),std::string(16384,static_cast<char>(i)));
  s.Retire(Key(1));Expect(kvcohort::ErrorCode::capacity,[&]{s.Put(Key(5),std::string(16384,'E'));});
  Require(s.CollectGarbage(),"reserved headroom did not permit mixed-segment GC");
  auto stats=s.GetStats();Require(stats.active_segments==1,"GC sealed an unrelated active stream");
  Require(stats.gc_moved_bytes==16384&&stats.gc_write_bytes==16384&&stats.gc_read_bytes==16384,"GC traffic accounting wrong");
  Require(stats.reclaimed_file_bytes==32768,"old source bytes were not reclaimed");
  s.Put(Key(5),std::string(16384,'E'));Require(s.Pin(Key(2))->Read()==std::string(16384,2),"GC reserve path lost live data");
}

void GCChoosesFeasibleVictim() {
  Temp t;kvcohort::Store s(t.root,Options());
  s.Put(Key(1),std::string(16384,'L'));s.Put(Key(2),std::string(16384,'D'));s.Seal();
  s.Put(Key(3),std::string(4096,'D'));s.Retire(Key(2));s.Retire(Key(3));
  const auto pressure=t.root/"test-pressure-reservation";const int fd=::open(pressure.c_str(),O_CREAT|O_EXCL|O_RDWR,0600);
  Require(fd>=0,"pressure file failed");const int allocation=::posix_fallocate(fd,0,8*1024*1024);::close(fd);
  Require(allocation==0,"pressure reservation failed");
  Require(s.CollectGarbage(),"zero-copy victim was ignored behind an infeasible larger victim");
  Require(s.GetStats().gc_moved_bytes==0&&s.GetStats().reclaimed_file_bytes==4096,"GC did not choose the feasible fully dead segment");
  Require(s.Pin(Key(1))->Read()==std::string(16384,'L'),"GC selection corrupted retained data");
  fs::remove(pressure);Require(s.CollectGarbage(),"mixed victim not collected after pressure release");
}

int RunCrash(const fs::path& root, const std::string& point, bool gc) {
  const auto pid = ::fork(); Require(pid >= 0, "fork failed");
  if (pid == 0) {
    ::execl("/proc/self/exe", "kvc_tests", "--crash", root.c_str(), point.c_str(), gc ? "gc" : "put", static_cast<char*>(nullptr));
    ::_exit(125);
  }
  int status=0; Require(::waitpid(pid,&status,0)==pid, "waitpid failed");
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
void PutCrashRecovery() {
  for (auto point : {"put_after_payload_sync", "put_after_metadata_commit"}) {
    Temp t; { kvcohort::Store s(t.root, Options()); s.Put(Key(1), "old committed object"); }
    Require(RunCrash(t.root,point,false)==77, "put crash failpoint did not execute");
    kvcohort::Store s(t.root,Options()); Require(s.Pin(Key(1))->Read()=="old committed object", "crash lost unrelated object");
    auto result=s.Pin(Key(2));
    if (std::string(point)=="put_after_payload_sync") Require(!result, "uncommitted object became visible");
    else Require(result && result->Read()=="new committed object", "committed object lost after process exit");
    s.Put(Key(3),"append after recovered tail"); Require(s.Pin(Key(3))->Read()=="append after recovered tail", "recovered segment unusable");
  }
}
void GCCrashRecovery() {
  for (auto point : {"gc_after_payload_sync", "gc_after_metadata_commit"}) {
    Temp t; { kvcohort::Store s(t.root,Options()); s.Put(Key(1),std::string(4096,'A')); s.Put(Key(2),std::string(4096,'B')); s.Seal(); s.Retire(Key(1)); }
    Require(RunCrash(t.root,point,true)==77, "GC crash failpoint did not execute");
    kvcohort::Store s(t.root,Options()); Require(!s.Pin(Key(1)), "GC resurrected retired data");
    Require(s.Pin(Key(2))->Read()==std::string(4096,'B'), "GC crash lost live payload");
    Require(s.GetStats().objects==1, "GC crash duplicated object");
    while(s.CollectGarbage()) {}
    Require(s.Pin(Key(2))->Read()==std::string(4096,'B'), "subsequent GC lost recovered object");
  }
}
}  // namespace

int main(int argc,char** argv) {
  if (argc==5 && std::string(argv[1])=="--crash") {
    try {
      auto opts=Options(); const std::string point=argv[3];
      opts.failpoint=[point](std::string_view at){if(at==point)::_exit(77);};
      kvcohort::Store store(argv[2],opts);
      if(std::string(argv[4])=="gc") store.CollectGarbage(); else store.Put(Key(2),"new committed object");
      return 124;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 123;}
  }
  int failed=0; int total=0;
  for (const auto& test : std::vector<std::pair<const char*,std::function<void()>>>{
      {"identity_and_immutable",IdentityAndImmutable},{"contiguous_prefix",ContiguousPrefix},
      {"reopen_and_exclusive_owner",ReopenAndExclusiveOwner},{"pinned_retirement_and_gc",PinnedRetirementAndGC},
      {"concurrent_pinned_read",ConcurrentPinnedRead},{"checksum_corruption",ChecksumDetectsCorruption},
      {"refuse_foreign_directory",RefusesForeignDirectory},{"admission_budget",AdmissionBudget},
      {"gc_pressure_preserves_reads",GCPressurePreservesReads},{"randomized_state_machine",RandomizedStateMachine},
      {"policy_cohorts",PolicyCohorts},{"layout_reopen_and_bound",LayoutReopenAndBound},
      {"batch_read_order_and_limits",BatchedReadOrderAndLimits},{"batch_gap_amplification",BatchGapAmplification},
      {"payload_debt_and_gc_reserve",PayloadDebtAndGCReserve},
      {"gc_chooses_feasible_victim",GCChoosesFeasibleVictim},
      {"put_crash_recovery",PutCrashRecovery},{"gc_crash_recovery",GCCrashRecovery}}) {
    ++total;
    try {test.second();std::cout<<"PASS "<<test.first<<'\n';}
    catch(const std::exception& e){++failed;std::cerr<<"FAIL "<<test.first<<": "<<e.what()<<'\n';}
  }
  std::cout<<"storage contracts: "<<(total-failed)<<" passed, "<<failed<<" failed\n";
  return failed?1:0;
}
