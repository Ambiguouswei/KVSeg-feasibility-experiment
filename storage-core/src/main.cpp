#include "kvcohort/store.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>

namespace {
void Print(const kvcohort::Stats& s) {
  std::cout << "{\"objects\":" << s.objects << ",\"live_payload_bytes\":" << s.live_payload_bytes
            << ",\"segment_bytes\":" << s.segment_bytes << ",\"allocated_bytes\":" << s.allocated_bytes
            << ",\"retired_segment_bytes\":" << s.retired_segment_bytes << ",\"segments\":" << s.segments
            << ",\"gc_moved_bytes_this_process\":" << s.gc_moved_bytes
            << ",\"active_segments\":" << s.active_segments
            << ",\"demand_read_calls\":" << s.demand_read_calls << ",\"demand_read_bytes\":" << s.demand_read_bytes
            << ",\"payload_write_bytes\":" << s.payload_write_bytes << ",\"gc_write_bytes\":" << s.gc_write_bytes
            << ",\"reclaimed_file_bytes\":" << s.reclaimed_file_bytes << "}\n";
}
kvcohort::BlockKey Key(char** args) {
  const auto ns = kvcohort::MakeNamespace(args[3], args[4], args[5]);
  std::vector<std::uint32_t> tokens; std::istringstream in(args[7]); std::string token;
  while (std::getline(in, token, ',')) {
    if (token.empty() || token[0] == '-') throw kvcohort::Error(kvcohort::ErrorCode::invalid, "invalid token ID");
    std::size_t used = 0; auto value = std::stoull(token, &used);
    if (used != token.size() || value > std::numeric_limits<std::uint32_t>::max())
      throw kvcohort::Error(kvcohort::ErrorCode::invalid, "invalid token ID");
    tokens.push_back(static_cast<std::uint32_t>(value));
  }
  return kvcohort::MakeBlockKey(ns, std::string(args[6]) == "-" ? "" : args[6], tokens);
}
void Usage() {
  std::cerr << "usage (optional flags before command: --layout NAME --prefix-group DIGEST --lifetime unknown|short|long):\n"
            << "  kvc stats|gc|smoke ROOT\n"
            << "  kvc put|get ROOT TENANT MODEL_REVISION REPRESENTATION PARENT_OR_DASH TOKEN_CSV FILE\n"
            << "  kvc retire ROOT TENANT MODEL_REVISION REPRESENTATION PARENT_OR_DASH TOKEN_CSV\n"
            << "Use a new empty directory beneath the dedicated project var directory.\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    kvcohort::Options options;kvcohort::PlacementHint hint;
    int first=1;
    while(first<argc&&std::string(argv[first]).rfind("--",0)==0) {
      if(first+1>=argc){Usage();return 2;}
      const std::string flag=argv[first],value=argv[first+1];
      if(flag=="--layout")options.layout=kvcohort::ParseLayout(value);
      else if(flag=="--prefix-group")hint.prefix_group=value;
      else if(flag=="--lifetime")hint.lifetime=kvcohort::ParseLifetime(value);
      else {Usage();return 2;}
      first+=2;
    }
    // Existing positional parser is relative to command at argv[1].
    argv+=first-1;argc-=first-1;
    if (argc < 3) { Usage(); return 2; }
    const std::string operation = argv[1];
    if ((operation == "put" || operation == "get") ? argc != 9 : (operation == "retire" ? argc != 8 : argc != 3)) {
      Usage(); return 2;
    }
    if (operation != "stats" && operation != "gc" && operation != "smoke" && operation != "put" && operation != "get" && operation != "retire") {
      Usage(); return 2;
    }
    kvcohort::Store store(argv[2],options);
    if (operation == "stats") Print(store.GetStats());
    else if (operation == "gc") {
      const bool collected = store.CollectGarbage(); store.ReapRetired();
      std::cout << "{\"collected\":" << (collected ? "true" : "false") << "}\n"; Print(store.GetStats());
    } else if (operation == "put") {
      const auto key = Key(argv); std::ifstream file(argv[8], std::ios::binary);
      if (!file) throw kvcohort::Error(kvcohort::ErrorCode::io, "cannot open input file");
      file.seekg(0, std::ios::end); auto size = file.tellg();
      if (size <= 0 || size > 1024 * 1024) throw kvcohort::Error(kvcohort::ErrorCode::invalid, "CLI payload must be between 1 byte and 1 MiB");
      file.seekg(0); const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
      std::cout << "{\"key\":\"" << key.id() << "\",\"generation\":" << store.Put(key, bytes,hint) << "}\n";
    } else if (operation == "get") {
      const auto key = Key(argv); auto ticket = store.Pin(key);
      if (!ticket) { std::cout << "{\"hit\":false}\n"; return 3; }
      const auto bytes = ticket->Read(); std::ofstream file(argv[8], std::ios::binary | std::ios::trunc);
      if (!file || !file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()))) throw kvcohort::Error(kvcohort::ErrorCode::io, "cannot write output file");
      std::cout << "{\"hit\":true,\"bytes\":" << bytes.size() << "}\n";
    } else if (operation == "retire") {
      std::cout << "{\"retired\":" << (store.Retire(Key(argv)) ? "true" : "false") << "}\n";
    } else {
      const auto ns = kvcohort::MakeNamespace("smoke-tenant", "synthetic-model-revision-v1", "opaque-kv-v1-fp16");
      const auto p = kvcohort::MakeBlockKey(ns, "", {1,2,3,4});
      const auto a = kvcohort::MakeBlockKey(ns, p.digest, {5,6,7,8});
      const auto b = kvcohort::MakeBlockKey(ns, p.digest, {9,10,11,12});
      const std::string common(16384, 'P'), tail_a(8192, 'A'), tail_b(8192, 'B');
      store.Put(p, common,{p.digest,kvcohort::Lifetime::long_lived});
      store.Put(a, tail_a,{p.digest,kvcohort::Lifetime::short_lived});store.Put(b, tail_b,{p.digest,kvcohort::Lifetime::short_lived});
      {
        auto tickets = store.LookupPrefix({p,a});
        const auto restored=store.ReadBatch(tickets);
        if (restored.size() != 2 || restored[0] != common || restored[1] != tail_a)
          throw kvcohort::Error(kvcohort::ErrorCode::corruption, "smoke prefix restore failed");
      }
      store.Seal(); store.Retire(a); store.Retire(b); store.CollectGarbage();
      if (store.Pin(p)->Read() != common || store.Pin(a)) throw kvcohort::Error(kvcohort::ErrorCode::corruption, "smoke GC validation failed");
      std::cout << "{\"smoke\":\"passed\",\"synthetic_only\":true}\n"; Print(store.GetStats());
    }
    return 0;
  } catch (const kvcohort::Error& e) {
    std::cerr << "kvc error " << static_cast<int>(e.code) << ": " << e.what() << '\n'; return 1;
  } catch (const std::exception& e) { std::cerr << "kvc error: " << e.what() << '\n'; return 1; }
}
