#!/usr/bin/env python3
"""Run a matched experiment matrix over (layout x variant x repeat) and summarise.

设计约束（都是为了结论可辩护）：
  * 每次运行使用全新的数据根 —— 回放器本身拒绝复用非空目录，这里预先建好唯一目录。
  * 逐次串行执行，不并发 —— 计数器类指标虽然是确定性的，但墙钟类会被相互干扰，
    串行能保证所有组的耗时处于同一干扰水平。
  * 记录命令行、trace SHA256 与完整 JSON，使任何一行数字都可回溯到一次具体运行。
  * 中途任何一次非零退出立即中止并保留 stderr，不做"忽略失败继续统计"的处理。

用法示例：
  # 隔离「GC 目标段封口」这个混淆变量：4 布局 x 2 种封段行为 x 3 次重复
  python3 scripts/run_matrix.py var/traces/mixed.trace var/matrix/appendable \\
      --sweep appendable --repeats 3

  # 只看命令不执行
  python3 scripts/run_matrix.py var/traces/mixed.trace var/matrix/dry \\
      --sweep gc-score --dry-run
"""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

# 每种扫描给出的变体：(名称, 额外命令行参数)
SWEEPS = {
    "none": [("default", {})],
    "appendable": [
        ("appendable-off", {"--gc-target-appendable": "0"}),
        ("appendable-on", {"--gc-target-appendable": "1"}),
    ],
    "admission": [
        ("no-admission", {}),
        ("admission", {"--admit-min-reuse": "1"}),
    ],
    "gc-score": [
        ("max-dead", {"--gc-score": "max-dead"}),
        ("min-cost", {"--gc-score": "min-cost"}),
    ],
}

# 参与汇总的数值指标；缺失的字段直接跳过，不补零。
METRICS = [
    "new_admissions",
    "admission_rejections",
    "admission_declined",
    "demand_pread_calls",
    "demand_read_bytes",
    "gc_moved_bytes",
    "gc_cycles",
    "payload_write_amplification",
    "usable_prefix_hit_ratio",
    "restore_declined_calls",
    "restore_bytes_avoided",
    "final_objects",
    "final_allocated_bytes",
]

# 摘要表里优先展示的列（顺序即列序）
HEADLINE = [
    "demand_pread_calls",
    "gc_moved_bytes",
    "usable_prefix_hit_ratio",
    "new_admissions",
    "admission_declined",
]


def trace_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_command(binary, trace, root, layout, variant_args, common):
    cmd = [str(binary), str(trace), str(root), layout]
    for key, value in common:
        cmd += [key, str(value)]
    for key, value in variant_args.items():
        cmd += [key, str(value)]
    return cmd


def run_once(cmd, workdir, timeout):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    (workdir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-6:]
        raise RuntimeError(
            f"replay failed (exit {proc.returncode}) in {workdir}\n  " + "\n  ".join(tail)
        )
    try:
        row = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"replay output is not JSON in {workdir}: {exc}") from exc
    (workdir / "run.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def ci95(values):
    """样本均值与 95% 置信半宽（正态近似）。n < 2 时半宽记 0 并在报告里注明。"""
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, 0.0
    sd = statistics.stdev(values)
    return mean, 1.96 * sd / (n ** 0.5)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run layouts x variants x repeats and summarise.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("trace", type=Path, help="轨迹文件（kvfs-trace-v1）")
    ap.add_argument("outdir", type=Path, help="结果目录，必须不存在（避免混入旧结果）")
    ap.add_argument("--binary", type=Path,
                    default=Path(__file__).resolve().parents[1] / "build" / "kvc_replay")
    ap.add_argument("--layouts", default="append,prefix,lifetime,joint")
    ap.add_argument("--sweep", default="none", choices=sorted(SWEEPS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--payload-budget-kib", type=int, default=256)
    ap.add_argument("--segment-kib", type=int, default=64)
    ap.add_argument("--max-open-segments", type=int, default=8)
    ap.add_argument("--max-gap-bytes", type=int, default=0)
    ap.add_argument("--calibrate-every", type=int, default=None)
    ap.add_argument("--extra", action="append", default=[],
                    metavar="FLAG=VALUE", help="附加原样透传给回放器的参数，可重复")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = ap.parse_args(argv)

    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    if not layouts:
        ap.error("--layouts 不能为空")
    if args.repeats < 1:
        ap.error("--repeats 必须 >= 1")

    variants = SWEEPS[args.sweep]

    common = [
        ("--payload-budget-kib", args.payload_budget_kib),
        ("--segment-kib", args.segment_kib),
        ("--max-open-segments", args.max_open_segments),
        ("--max-gap-bytes", args.max_gap_bytes),
    ]
    if args.calibrate_every is not None:
        common.append(("--calibrate-every", args.calibrate_every))
    for item in args.extra:
        if "=" not in item:
            ap.error(f"--extra 需要 FLAG=VALUE 形式：{item}")
        key, value = item.split("=", 1)
        common.append((key, value))

    if not args.dry_run:
        if args.outdir.exists() and any(args.outdir.iterdir()):
            ap.error(f"输出目录非空：{args.outdir}（请换一个新目录）")
        if not args.trace.is_file():
            ap.error(f"轨迹不存在：{args.trace}")
        if not args.binary.is_file():
            ap.error(f"回放器不存在：{args.binary}（先运行 scripts/build.sh）")

    digest = trace_digest(args.trace) if args.trace.is_file() else "(trace missing)"

    rows = []
    total = len(variants) * len(layouts) * args.repeats
    done = 0
    for variant_name, variant_args in variants:
        for layout in layouts:
            for repeat in range(1, args.repeats + 1):
                workdir = args.outdir / variant_name / layout / f"r{repeat}"
                # 数据根必须是全新空目录，且不能与元数据目录混用：
                # 回放器会拒绝任何非空的数据根，而我们要在同目录下写 command.txt / stderr.txt。
                data_root = workdir / "data"
                cmd = build_command(args.binary, args.trace, data_root, layout, variant_args, common)
                done += 1
                if args.dry_run:
                    print(f"[{done}/{total}] " + " ".join(cmd))
                    continue
                print(f"[{done}/{total}] {variant_name}/{layout}/r{repeat}", flush=True)
                row = run_once(cmd, workdir, args.timeout)
                row["_variant"] = variant_name
                row["_layout"] = layout
                row["_repeat"] = repeat
                row["_trace_sha256"] = digest
                rows.append(row)

    if args.dry_run:
        print(f"\n共 {total} 次运行（dry-run，未执行）")
        return 0

    # ---- 一致性检查：所有运行的轨迹必须完全一致 ----
    digests = {r.get("trace_sha256") for r in rows}
    digests |= {r.get("_trace_sha256") for r in rows}
    digests.discard(None)
    if len(digests) != 1:
        raise RuntimeError(f"各次运行的 trace SHA256 不一致：{sorted(digests)}")

    # ---- runs.csv ----
    args.outdir.mkdir(parents=True, exist_ok=True)
    cols = ["_variant", "_layout", "_repeat"] + METRICS
    present = [c for c in cols if any(c in r for r in rows)]
    with (args.outdir / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=present, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in present})

    # ---- 汇总 ----
    summary = {}
    for row in rows:
        summary.setdefault((row["_variant"], row["_layout"]), []).append(row)

    metric_cols = [m for m in METRICS if any(m in r for r in rows)]
    lines = []
    lines.append(f"# 实验矩阵汇总\n")
    lines.append(f"- 轨迹：`{args.trace}`")
    lines.append(f"- trace SHA256：`{digest}`")
    lines.append(f"- 扫描：`{args.sweep}`；布局：{', '.join(layouts)}；重复：{args.repeats}")
    lines.append(f"- 共 {len(rows)} 次运行\n")

    lines.append("## 按 (变体, 布局) 的均值\n")
    head = ["变体", "布局", "n"] + metric_cols
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for (variant, layout), group in summary.items():
        cells = [variant, layout, str(len(group))]
        for metric in metric_cols:
            values = [g[metric] for g in group if isinstance(g.get(metric), (int, float))]
            if not values:
                cells.append("-")
                continue
            mean, half = ci95(values)
            text = f"{mean:.4g}" if isinstance(mean, float) and not float(mean).is_integer() else f"{mean:.0f}"
            cells.append(f"{text} ±{half:.2g}" if len(values) > 1 else text)
        lines.append("| " + " | ".join(cells) + " |")

    # ---- 对比：变体 2 - 变体 1 ----
    if len(variants) == 2:
        first, second = variants[0][0], variants[1][0]
        lines.append(f"\n## 对比：`{second}` − `{first}`（同一布局内）\n")
        head2 = ["布局"] + [m for m in metric_cols if m in HEADLINE or True]
        lines.append("| " + " | ".join(head2) + " |")
        lines.append("|" + "|".join(["---"] * len(head2)) + "|")
        for layout in layouts:
            a = summary.get((first, layout), [])
            b = summary.get((second, layout), [])
            cells = [layout]
            for metric in head2[1:]:
                va = [x[metric] for x in a if isinstance(x.get(metric), (int, float))]
                vb = [x[metric] for x in b if isinstance(x.get(metric), (int, float))]
                if not va or not vb:
                    cells.append("-")
                    continue
                delta = statistics.mean(vb) - statistics.mean(va)
                cells.append(f"{delta:+.4g}")
            lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n---\n")
    lines.append("说明：")
    lines.append("- `±` 为 95% 置信半宽（正态近似）；单次运行时省略。")
    lines.append("- 所有指标都是进程内计数器；`demand_pread_calls` 是 pread 系统调用次数，")
    lines.append("  不等于 SSD 设备 I/O 次数（数据可能命中页缓存，需配合 O_DIRECT 或计入 diskstats）。")
    lines.append("- 运行是串行的，避免组间 I/O 干扰，代价是总耗时随重复数线性增长。")

    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出：{args.outdir/'runs.csv'}")
    print(f"写出：{args.outdir/'summary.md'}")
    print(f"共 {len(rows)} 次运行，全部非零退出次数：0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
