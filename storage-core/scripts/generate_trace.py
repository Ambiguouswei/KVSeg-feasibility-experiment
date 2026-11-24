#!/usr/bin/env python3
"""Generate an explicitly synthetic prefix/lifetime conflict trace, without GPUs."""
import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--rounds',type=int,default=12)
    parser.add_argument('--families',type=int,default=4)
    parser.add_argument('--depth',type=int,default=4)
    parser.add_argument('--hints',choices=['mixed','uniform','unknown'],default='mixed')
    parser.add_argument('--no-reuse',action='store_true')
    parser.add_argument('--tenants',type=int,default=1,
                        help='族数量超过该值时按族轮转分配租户；>1 时 put 行追加第 7 字段 TENANT')
    args=parser.parse_args()
    if not(1<=args.rounds<=64 and 1<=args.families<=16 and 1<=args.depth<=8 and 1<=args.tenants<=16):parser.error('trace dimensions exceed the bounded generator limits')
    lines=['kvfs-trace-v1','# Synthetic opaque payloads, fixed 32 logical tokens/block and 8192 bytes/object.',
           '# Lifetime labels are caller hints; retire events are separately defined. No predictor or future-access oracle is evaluated.']
    life=lambda expected:'unknown' if args.hints=='unknown' else ('long' if args.hints=='uniform' else expected)
    roots=list(range(1,args.families+1))
    tenant_of={root:index%args.tenants for index,root in enumerate(roots)}
    suffix=lambda root:(' '+str(tenant_of[root])) if args.tenants>1 else ''
    for root in roots:lines.append(f'put {root} 0 {root} {life("long")} 8192'+suffix(root))
    next_id=args.families+1
    for _ in range(args.rounds):
        chains={root:[] for root in roots}
        for _ in range(args.depth):
            for root in roots:
                parent=chains[root][-1] if chains[root] else root
                lines.append(f'put {next_id} {parent} {root} {life("short")} 8192'+suffix(root))
                chains[root].append(next_id);next_id+=1
        if not args.no_reuse:
            for _ in range(2):
                for root in roots:lines.append(f'read {chains[root][-1]}')
        for i,root in enumerate(roots):
            if i%2==0:
                lines.extend(f'retire {value}' for value in reversed(chains[root]))
        lines.append('gc 64')
        if not args.no_reuse:
            for root in roots:lines.append(f'read {chains[root][-1]}')
        for i,root in enumerate(roots):
            if i%2==1:
                lines.extend(f'retire {value}' for value in reversed(chains[root]))
        lines.append('gc 64')
    lines.extend(f'retire {root}' for root in roots)
    lines.append('gc 64')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as file:file.write('\n'.join(lines)+'\n')
    print(args.output)

if __name__=='__main__':main()
