#include "kvcohort/store.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace fs=std::filesystem;
namespace {
using Clock=std::chrono::steady_clock;
constexpr std::uint64_t kTokensPerBlock=32;
struct Definition {
  std::uint32_t parent=0,group=0,tenant=0;
  std::uint64_t bytes=0;
  kvcohort::Lifetime lifetime=kvcohort::Lifetime::unknown;
  kvcohort::BlockKey key;
};
struct Event {std::string operation;std::uint32_t id=0;std::uint64_t count=1;};
struct Trace {std::map<std::uint32_t,Definition> objects;std::vector<Event> events;};
std::uint64_t Number(const std::string& text,std::uint64_t maximum) {
  if(text.empty()||!std::all_of(text.begin(),text.end(),[](char c){return c>='0'&&c<='9';}))throw std::runtime_error("expected an unsigned integer");
  std::size_t used=0;const auto n=std::stoull(text,&used);
  if(used!=text.size()||n>maximum)throw std::runtime_error("integer exceeds trace/option limit");
  return n;
}
double Real(const std::string& text) {
  if(text.empty())throw std::runtime_error("expected a number");
  std::size_t used=0;double v=0;
  try {v=std::stod(text,&used);} catch(const std::exception&){throw std::runtime_error("expected a number");}
  if(used!=text.size()||!(v>=0.0))throw std::runtime_error("expected a non-negative number");
  return v;
}
std::vector<std::string> Fields(const std::string& line) {
  std::istringstream in(line.substr(0,line.find('#')));std::vector<std::string> fields;std::string word;
  while(in>>word)fields.push_back(word);
  return fields;
}
Trace Load(const fs::path& path,std::uint64_t max_object_bytes) {
  if(fs::file_size(path)>64*1024*1024)throw std::runtime_error("trace exceeds 64 MiB input limit");
  std::ifstream file(path);if(!file)throw std::runtime_error("cannot open trace");
  Trace trace;std::string line;std::uint64_t line_number=0;bool header=false;
  // 单租户（tenant=0）沿用旧命名空间名，保证历史轨迹的摘要与已归档证据可复现。
  const auto space_for=[](std::uint32_t tenant){
    return kvcohort::MakeNamespace(tenant?"replay-tenant-"+std::to_string(tenant):std::string("replay-tenant"),
                                   "synthetic-model-v1","opaque-test-payload-block32-v1");
  };
  while(std::getline(file,line)) {
    ++line_number;auto fields=Fields(line);if(fields.empty())continue;
    try {
      if(!header) {if(fields!=std::vector<std::string>{"kvfs-trace-v1"})throw std::runtime_error("missing kvfs-trace-v1 header");header=true;continue;}
      if(trace.events.size()>=200000)throw std::runtime_error("trace exceeds 200000 events");
      const auto& operation=fields[0];
      if(operation=="put") {
        if(fields.size()!=6&&fields.size()!=7)throw std::runtime_error("put requires ID PARENT GROUP LIFETIME BYTES [TENANT]");
        const auto id=static_cast<std::uint32_t>(Number(fields[1],1000000));
        const auto parent=static_cast<std::uint32_t>(Number(fields[2],1000000));
        auto group=static_cast<std::uint32_t>(Number(fields[3],1000000));
        auto lifetime=kvcohort::ParseLifetime(fields[4]);const auto bytes=Number(fields[5],max_object_bytes);
        const auto tenant=static_cast<std::uint32_t>(fields.size()==7?Number(fields[6],1024):0);
        if(!id||!bytes||id==parent||(parent&&!trace.objects.count(parent)))throw std::runtime_error("invalid object ID, size or undefined parent");
        if(parent&&trace.objects.at(parent).tenant!=tenant)throw std::runtime_error("prefix chain must not cross tenants");
        std::uint32_t ancestor=id,root=id;bool group_is_ancestor=group==0;std::uint64_t depth=0;
        while(ancestor) {
          root=ancestor;if(ancestor==group)group_is_ancestor=true;
          ancestor=ancestor==id?parent:trace.objects.at(ancestor).parent;
          if(++depth>512)throw std::runtime_error("prefix chain exceeds 512 blocks");
        }
        if(!group_is_ancestor)throw std::runtime_error("prefix group must be the object itself or an ancestor");
        if(!group)group=root;
        auto it=trace.objects.find(id);
        if(it==trace.objects.end()) {
          std::vector<std::uint32_t> tokens(kTokensPerBlock,id);
          auto key=kvcohort::MakeBlockKey(space_for(tenant),parent?trace.objects.at(parent).key.digest:"",tokens);
          trace.objects.emplace(id,Definition{parent,group,tenant,bytes,lifetime,std::move(key)});
        } else if(it->second.parent!=parent||it->second.group!=group||it->second.tenant!=tenant||it->second.bytes!=bytes||it->second.lifetime!=lifetime)
          throw std::runtime_error("redefinition changes immutable synthetic object metadata");
        trace.events.push_back({operation,id,1});
      } else if(operation=="read"||operation=="retire") {
        if(fields.size()!=2)throw std::runtime_error("read/retire requires one object ID");
        const auto id=static_cast<std::uint32_t>(Number(fields[1],1000000));
        if(!trace.objects.count(id))throw std::runtime_error("read/retire references an undefined object");
        trace.events.push_back({operation,id,1});
      } else if(operation=="gc") {
        if(fields.size()>2)throw std::runtime_error("gc accepts an optional attempt count");
        const auto count=fields.size()==2?Number(fields[1],1024):1;
        if(!count)throw std::runtime_error("gc attempt count must be positive");
        trace.events.push_back({operation,0,count});
      } else if(operation=="seal") {
        if(fields.size()!=1)throw std::runtime_error("seal accepts no fields");
        trace.events.push_back({operation,0,1});
      } else throw std::runtime_error("unknown operation");
    } catch(const std::exception& e) {throw std::runtime_error("trace line "+std::to_string(line_number)+": "+e.what());}
  }
  if(!header)throw std::runtime_error("empty trace");
  return trace;
}
std::string Payload(std::uint32_t id,std::uint64_t size) {
  std::string result(static_cast<std::size_t>(size),'\0');std::uint32_t state=id^0x9e3779b9U;
  for(auto& byte:result){state^=state<<13;state^=state>>17;state^=state<<5;byte=static_cast<char>(state&255);}
  return result;
}
std::uint64_t Micros(Clock::time_point start) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count());
}
void Ratio(std::ostream& out,std::uint64_t top,std::uint64_t bottom) {
  if(bottom)out<<static_cast<double>(top)/static_cast<double>(bottom);else out<<"null";
}
void Run(const Trace& trace,const fs::path& root,const kvcohort::Options& options,const kvcohort::BatchReadOptions& read_options) {
  if(fs::exists(root)&&!fs::is_empty(root))throw std::runtime_error("replay requires a fresh empty data root");
  kvcohort::Store store(root,options);std::set<std::uint32_t> resident;
  std::uint64_t puts=0,admitted=0,rejected=0,declined=0,reads=0,full_hits=0,required_blocks=0,hit_blocks=0,restored_bytes=0;
  std::uint64_t gc_attempts=0,gc_cycles=0,gc_stalls=0,restore_us=0,gc_us=0,max_allocated=0,max_occupied=0;
  const auto start=Clock::now();
  const auto collect=[&]() {
    ++gc_attempts;const auto at=Clock::now();bool result=false;
    try {result=store.CollectGarbage();if(result)++gc_cycles;}
    catch(const kvcohort::Error& e){if(e.code!=kvcohort::ErrorCode::capacity)throw;++gc_stalls;}
    gc_us+=Micros(at);return result;
  };
  for(const auto& event:trace.events) {
    if(event.operation=="put") {
      ++puts;const auto& object=trace.objects.at(event.id);const auto payload=Payload(event.id,object.bytes);
      const kvcohort::PlacementHint hint{trace.objects.at(object.group).key.digest,object.lifetime};
      bool stored=false;
      for(std::uint64_t attempt=0;attempt<1024;++attempt) {
        try {store.Put(object.key,payload,hint);stored=true;break;}
        catch(const kvcohort::Error& e){
          if(e.code==kvcohort::ErrorCode::declined){++declined;break;}
          if(e.code!=kvcohort::ErrorCode::capacity)throw;
          if(!collect())break;
        }
      }
      if(stored){if(resident.insert(event.id).second)++admitted;}else ++rejected;
    } else if(event.operation=="read") {
      ++reads;std::vector<std::uint32_t> ids;
      for(auto id=event.id;id;id=trace.objects.at(id).parent)ids.push_back(id);
      std::reverse(ids.begin(),ids.end());std::vector<kvcohort::BlockKey> chain;
      for(auto id:ids)chain.push_back(trace.objects.at(id).key);
      const auto at=Clock::now();const auto tickets=store.LookupPrefix(chain);const auto values=store.ReadBatch(tickets,read_options);restore_us+=Micros(at);
      required_blocks+=chain.size();hit_blocks+=tickets.size();if(tickets.size()==chain.size())++full_hits;
      for(std::size_t i=0;i<values.size();++i) {
        const auto& object=trace.objects.at(ids[i]);
        if(values[i]!=Payload(ids[i],object.bytes))throw std::runtime_error("restored payload differs from deterministic reference");
        restored_bytes+=values[i].size();
      }
    } else if(event.operation=="retire") {
      const bool expected=resident.erase(event.id)>0;
      if(store.Retire(trace.objects.at(event.id).key)!=expected)throw std::runtime_error("retirement differs from reference state");
    } else if(event.operation=="gc") {
      for(std::uint64_t i=0;i<event.count;++i)if(!collect())break;
    } else store.Seal();
    const auto stats=store.GetStats();max_allocated=std::max(max_allocated,stats.allocated_bytes);max_occupied=std::max(max_occupied,stats.segment_bytes);
    if(stats.active_segments>options.max_open_segments)throw std::runtime_error("active stream bound violated");
  }
  const auto stats=store.GetStatsExact();if(stats.objects!=resident.size())throw std::runtime_error("final object count differs from reference state");
  std::cout<<"{\"schema\":\"kvfs-replay-v1\",\"layout\":\""<<kvcohort::LayoutName(options.layout)<<"\",\"synthetic_only\":true,\"caller_lifetime_hints\":true"
    <<",\"events\":"<<trace.events.size()<<",\"segment_bytes\":"<<options.segment_bytes<<",\"physical_admission_budget_bytes\":"<<options.capacity_bytes
    <<",\"payload_budget_bytes\":"<<options.payload_budget_bytes<<",\"gc_reserve_bytes\":"<<options.gc_reserve_bytes
    <<",\"max_open_segments\":"<<options.max_open_segments<<",\"max_gap_bytes\":"<<read_options.max_gap_bytes
    <<",\"write_events\":"<<puts<<",\"new_admissions\":"<<admitted<<",\"admission_rejections\":"<<rejected<<",\"admission_declined\":"<<declined
    <<",\"read_requests\":"<<reads<<",\"full_prefix_hits\":"<<full_hits<<",\"requested_tokens\":"<<required_blocks*kTokensPerBlock
    <<",\"usable_prefix_tokens\":"<<hit_blocks*kTokensPerBlock<<",\"usable_prefix_hit_ratio\":";
  Ratio(std::cout,hit_blocks,required_blocks);
  std::cout<<",\"restored_bytes\":"<<restored_bytes<<",\"demand_pread_calls\":"<<stats.demand_read_calls<<",\"demand_read_bytes\":"<<stats.demand_read_bytes
    <<",\"read_amplification\":";Ratio(std::cout,stats.demand_read_bytes,restored_bytes);
  std::cout<<",\"new_payload_write_bytes\":"<<stats.payload_write_bytes<<",\"gc_moved_bytes\":"<<stats.gc_moved_bytes
    <<",\"payload_pwrite_calls\":"<<stats.payload_write_calls<<",\"gc_pread_calls\":"<<stats.gc_read_calls<<",\"gc_pwrite_calls\":"<<stats.gc_write_calls
    <<",\"gc_read_bytes\":"<<stats.gc_read_bytes<<",\"gc_write_bytes\":"<<stats.gc_write_bytes<<",\"payload_write_amplification\":";
  Ratio(std::cout,stats.payload_write_bytes+stats.gc_write_bytes,stats.payload_write_bytes);
  std::cout<<",\"reclaimed_file_bytes\":"<<stats.reclaimed_file_bytes<<",\"gc_attempts\":"<<gc_attempts<<",\"gc_cycles\":"<<gc_cycles<<",\"gc_capacity_stalls\":"<<gc_stalls
    <<",\"max_observed_allocated_bytes\":"<<max_allocated<<",\"max_observed_payload_occupied_bytes\":"<<max_occupied
    <<",\"final_objects\":"<<stats.objects<<",\"final_allocated_bytes\":"<<stats.allocated_bytes
    <<",\"gc_score\":\""<<kvcohort::GcScoreName(options.policy.gc_score)<<"\""
    <<",\"restore_declined_calls\":"<<stats.restore_declined_calls<<",\"restore_bytes_avoided\":"<<stats.restore_bytes_avoided
    <<",\"restore_wall_us_total\":"<<restore_us<<",\"gc_wall_us_total\":"<<gc_us<<",\"elapsed_us\":"<<Micros(start)<<"}\n";
}
}  // namespace
int main(int argc,char** argv) {
  try {
    if(argc<4||(argc-4)%2){std::cerr<<"usage: kvc_replay TRACE ROOT LAYOUT [--payload-budget-kib N] [--segment-kib N] [--max-open-segments N] [--max-gap-bytes N] [--calibrate-every N] [--gc-target-appendable 0|1] [--direct-io 0|1] [--admit-min-reuse N] [--gc-score max-dead|min-cost] [--restore-bytes-per-us X] [--prefill-us-per-token Y] [--restore-queue-us Z]\n";return 2;}
    kvcohort::Options options;options.calibrate_every_n_writes=64;options.layout=kvcohort::ParseLayout(argv[3]);options.segment_bytes=64*1024;options.payload_budget_bytes=256*1024;
    kvcohort::BatchReadOptions reads;
    for(int i=4;i<argc;i+=2) {
      const std::string flag=argv[i];const std::string raw=argv[i+1];
      if(flag=="--gc-score"){options.policy.gc_score=kvcohort::ParseGcScore(raw);continue;}
      if(flag=="--restore-bytes-per-us"){options.policy.restore_bytes_per_us=Real(raw);continue;}
      if(flag=="--prefill-us-per-token"){options.policy.prefill_us_per_token=Real(raw);continue;}
      if(flag=="--restore-queue-us"){options.policy.restore_queue_us=Real(raw);continue;}
      const auto value=Number(raw,64*1024*1024);
      if(flag=="--payload-budget-kib")options.payload_budget_bytes=value*1024;
      else if(flag=="--segment-kib") {if(value<4||value>4096)throw std::runtime_error("segment size must be 4 to 4096 KiB");options.segment_bytes=value*1024;}
      else if(flag=="--max-open-segments")options.max_open_segments=value;
      else if(flag=="--max-gap-bytes")reads.max_gap_bytes=value;
      else if(flag=="--calibrate-every")options.calibrate_every_n_writes=value;
      else if(flag=="--gc-target-appendable")options.gc_target_appendable=value!=0;
      else if(flag=="--direct-io")options.direct_io=value!=0;
      else if(flag=="--admit-min-reuse"){options.policy.admit_enabled=true;options.policy.admit_min_reuse=static_cast<std::uint32_t>(value);}
      else throw std::runtime_error("unknown replay option");
    }
    options.policy.restore_decision_enabled =
        options.policy.restore_bytes_per_us>0.0 && options.policy.prefill_us_per_token>0.0;
    options.max_object_bytes=options.segment_bytes;options.gc_reserve_bytes=options.segment_bytes;
    reads.max_batch_bytes=std::max<std::uint64_t>(1024*1024,options.max_object_bytes);
    const auto trace=Load(argv[1],options.max_object_bytes);Run(trace,argv[2],options,reads);return 0;
  } catch(const std::exception& e){std::cerr<<"replay error: "<<e.what()<<'\n';return 1;}
}
