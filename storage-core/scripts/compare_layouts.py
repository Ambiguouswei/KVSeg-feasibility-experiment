#!/usr/bin/env python3
"""Run matched layouts sequentially and retain input hash, options and raw JSON."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('trace',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--binary',type=Path,default=Path(__file__).resolve().parents[1]/'build/kvc_replay')
    parser.add_argument('--payload-budget-kib',type=int,default=256)
    parser.add_argument('--segment-kib',type=int,default=64)
    parser.add_argument('--max-open-segments',type=int,default=8)
    parser.add_argument('--max-gap-bytes',type=int,default=0)
    args=parser.parse_args()
    trace=args.trace.resolve();binary=args.binary.resolve();output=args.output.resolve()
    options=['--payload-budget-kib',str(args.payload_budget_kib),'--segment-kib',str(args.segment_kib),
             '--max-open-segments',str(args.max_open_segments),'--max-gap-bytes',str(args.max_gap_bytes)]
    if not trace.is_file() or not binary.is_file():parser.error('trace and replay executable must exist')
    output.mkdir(parents=True,exist_ok=False)
    data=trace.read_bytes();(output/'input.trace').write_bytes(data)
    results=[]
    for layout in ['append','prefix','lifetime','joint']:
        command=[str(binary),str(trace),str(output/layout),layout,*options]
        run=subprocess.run(command,capture_output=True,text=True,timeout=180)
        (output/f'{layout}.stderr.txt').write_text(run.stderr,encoding='utf-8')
        if run.returncode:raise RuntimeError(f'{layout} failed with {run.returncode}: {run.stderr}')
        result=json.loads(run.stdout);result['trace_sha256']=hashlib.sha256(data).hexdigest()
        result['command']=command;(output/f'{layout}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        results.append(result)
    comparable=['events','segment_bytes','physical_admission_budget_bytes','payload_budget_bytes','gc_reserve_bytes','max_open_segments','max_gap_bytes','requested_tokens','trace_sha256']
    if any(any(row[key]!=results[0][key] for key in comparable) for row in results):raise RuntimeError('comparison input or budgets differ')
    columns=['layout','new_admissions','admission_rejections','usable_prefix_hit_ratio','demand_pread_calls','demand_read_bytes',
             'gc_moved_bytes','gc_cycles','payload_write_amplification','max_observed_payload_occupied_bytes','max_observed_allocated_bytes']
    with (output/'summary.tsv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=columns,delimiter='\t',extrasaction='ignore');writer.writeheader();writer.writerows(results)
    print(json.dumps({'output':str(output),'trace_sha256':results[0]['trace_sha256'],'results':[{k:r[k] for k in columns} for r in results]},indent=2))

if __name__=='__main__':main()
