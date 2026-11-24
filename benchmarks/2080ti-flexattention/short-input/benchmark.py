import argparse, asyncio, json, time, pathlib, statistics, math
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from kvcohort_vllm.client import Client
from cache_pages import prepare

REV='aa8e72537993ba99e69dfaafa59ed015b17504d1'
SCENARIOS=[(512,1,1),(512,4,1),(512,4,4),(2048,1,1),(2048,4,1),(2048,4,4)]
def prompt(tok,length,group,tail,namespace=0):
    base=tok.encode('A cached document contains facts for answering questions. ',add_special_tokens=False)
    count=length-32
    return [10000+namespace*100+group+(10 if length==2048 else 0)]+(base*((count//len(base))+2))[:count-1]+[tail]+(base*10)[:31]
def stats(socket):
    c=Client(socket)
    try: return c.stats()
    finally: c.close()
async def request(engine,tokens,rid,output=8):
    params=SamplingParams(temperature=0,max_tokens=output,ignore_eos=True,output_kind=RequestOutputKind.DELTA)
    start=time.perf_counter();first=None;cached=None;out=[]
    async for event in engine.generate({'prompt_token_ids':tokens},params,rid):
        if event.num_cached_tokens is not None: cached=event.num_cached_tokens
        if event.outputs and event.outputs[0].token_ids:
            if first is None:first=time.perf_counter()
            out.extend(event.outputs[0].token_ids)
    if first is None or len(out)!=output: raise RuntimeError(f'incomplete {rid}: {len(out)}')
    return {'ttft_seconds':first-start,'completion_seconds':time.perf_counter()-start,'cached_tokens':cached,'output_token_ids':out}
async def batch(engine,tok,length,n,groups,tag,namespace=0):
    ready=asyncio.Event()
    async def one(i):
        await ready.wait()
        return await request(engine,prompt(tok,length,i%groups,30000+i,namespace),f'{tag}-{i}')
    tasks=[asyncio.create_task(one(i)) for i in range(n)]
    await asyncio.sleep(0);start=time.perf_counter();ready.set();result=await asyncio.gather(*tasks);elapsed=time.perf_counter()-start
    return {'batch_elapsed_seconds':elapsed,'output_tokens_per_second':sum(len(x['output_token_ids']) for x in result)/elapsed,'requests':result}
async def run(args):
    p=pathlib.Path(args.output); tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    cfg=None
    if args.mode!='official':
        cfg=KVTransferConfig(kv_connector='KVCohortBenchmarkConnector',kv_role='kv_both',kv_connector_module_path='kvcohort_vllm.benchmark_connector',kv_connector_extra_config={'socket_path':args.socket,'model_revision':REV,'read_only':args.mode!='prime','prefix_checkpoints':[480,2016], 'cpu_cache_bytes':512*1024**2 if args.mode=='cpu' else 0,'prefetch_workers':2})
    engine=AsyncLLM.from_engine_args(AsyncEngineArgs(model=args.model,dtype='float16',tensor_parallel_size=1,gpu_memory_utilization=.80,max_model_len=2112,max_num_seqs=4,max_num_batched_tokens=8192,enable_prefix_caching=True,enable_chunked_prefill=True,enforce_eager=True,disable_log_stats=True,kv_transfer_config=cfg))
    records=[]
    try:
        if args.mode=='prime':
            for length in (512,2048):
                for group in range(4):
                    await engine.reset_prefix_cache(); before=stats(args.socket)
                    t=await request(engine,prompt(tok,length,group,50000),f'prime-{length}-{group}',1)
                    after=stats(args.socket);written=after['payload_write_bytes']-before['payload_write_bytes']
                    if written<1024*1024:raise RuntimeError(f'prime missing data {length} {group}: {written}')
                    records.append({'length':length,'group':group,'write_bytes':written,'request':t});p.write_text(json.dumps(records,indent=2)+'\n')
                    print('PRIME',length,group,written,flush=True)
            return
        if args.mode=='cpu':
            for length in (512,2048):
                for group in range(4):
                    await engine.reset_prefix_cache()
                    x=await request(engine,prompt(tok,length,group,51000),f'cpuseed-{length}-{group}',1)
                    if (x['cached_tokens'] or 0)<length-32:raise RuntimeError(f'cpu preload miss {length} {group}: {x}')
            print('CPU_PRELOAD_COMPLETE',flush=True)
        for length,n,groups in SCENARIOS:
            await engine.reset_prefix_cache()
            await batch(engine,tok,length,n,groups,f'warm-{length}-{n}-{groups}',namespace=5)
            for rep in range(3):
                await engine.reset_prefix_cache()
                page_state=prepare(args.data_root,'cold') if args.mode=='file' else None
                if page_state and page_state['resident_fraction']>.001:raise RuntimeError(f'page cache not cold: {page_state}')
                before=stats(args.socket); result=await asyncio.wait_for(batch(engine,tok,length,n,groups,f'measure-{args.mode}-{length}-{n}-{groups}-{rep}'),600);after=stats(args.socket)
                reads=after['demand_read_bytes']-before['demand_read_bytes']; hits=[x['cached_tokens'] for x in result['requests']]
                row={'mode':args.mode,'length':length,'concurrency':n,'prefix_groups':groups,'repetition':rep,'backend_read_bytes':reads,'page_state':page_state,**result}
                records.append(row);p.write_text(json.dumps(records,indent=2)+'\n');print('MEASURE',json.dumps(row),flush=True)
                if args.mode=='official' and (hits[0]!=0 or reads!=0):raise RuntimeError(f'official first request hit/reads {hits} {reads}')
                if args.mode=='cpu' and (min(x or 0 for x in hits)<length-32 or reads>1024*1024):raise RuntimeError(f'cpu miss {hits} {reads}')
                if args.mode=='file' and (min(x or 0 for x in hits)<length-32 or reads<(length-32)*36*1024):raise RuntimeError(f'file miss {hits} {reads}')
    finally:engine.shutdown()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prime','official','cpu','file'],required=True);ap.add_argument('--model',required=True);ap.add_argument('--socket',required=True);ap.add_argument('--data-root',required=True);ap.add_argument('--output',required=True);asyncio.run(run(ap.parse_args()))
if __name__=='__main__':main()
