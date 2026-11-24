# kvcohort

面向 LLM KV Cache 的前缀与生命周期感知用户态存储研究原型。

当前版本 **0.2** 包含同步存储核心、可替换布局接口和 CPU 轨迹回放器，尚未提供挂载文件系统。它使用 RocksDB 保存对象与段映射，在普通文件中追加不可变载荷，提供 C++ API 与 CLI。

## 已实现

- SHA256 命名空间绑定租户、精确模型版本、表示版本及额外语义身份；块摘要绑定实际 token ID 与父前缀。
- 不可变对象、重复写入合并与单调代次；完整连续前缀查询遇到缺口停止。
- 先同步数据，再通过同步 RocksDB WriteBatch 发布映射；每次成功 Put 均为持久确认。
- 读取票据固定物理文件和对象代次。逻辑淘汰立即隐藏对象，在途读取仍可完成。
- 同步 GC 复制有效载荷，数据同步后原子切换映射，再等待票据释放并删除旧段。
- 重启丢弃未提交段尾，回收未发布的 GC 目标和孤儿段；无需扫描所有张量。
- 数据根目录标记、拒绝接管非空普通目录、单进程独占与校验失败报错。
- 包含元数据实际分配量的准入检查，为 GC 和元数据预留余量。
- 四种布局：时间追加 `append`、仅前缀 `prefix`、仅生命周期 `lifetime` 和固定联合分组 `joint`，共用正确性与 GC 路径。
- 有界活动段集合，按最近写入顺序封段；布局配置持久化，阻止同一数据根误用另一种策略。
- 根据读票据排序并合并相邻读取，设置批次、间隙与输出字节上限，保持输出顺序并校验每个对象。
- 记录需求读取调用与字节、新载荷写入、GC 读取与写入、回收文件字节等进程内指标。
- 版本化合成轨迹、真实文件后端回放和四种布局的等预算比较脚本。

## 尚未实现

自适应联合代价模型、热度与寿命预测、异步队列、后台自动淘汰、IPC 服务、GPU 传输、vLLM 适配和 FUSE 均在后续阶段。当前 `joint` 使用调用方提供的稳定前缀祖先与粗寿命标签，尚不根据读放大与迁移成本自动选择分组。GC 在可用搬移空间内按无效字节选择候选，不声称已有推理性能提升。

当前准入预算是保守检查，**不是底层硬配额**：RocksDB 后台压缩和元数据增长可能使检查后的实际占用增加。严格的跨引擎物理配额和 ENOSPC 故障注入仍需实现。初版也没有按层的就绪事件、跨层已驻留锚点或自动缓存损坏隔离。调用者负责提供准确身份；租户认证需在未来服务适配层中落实。

## 构建与测试

Linux、C++17、CMake、RocksDB 9.10、OpenSSL。Conda 环境通过名称创建，使用用户默认 envs_dirs；不指定项目内 prefix，不修改 base 环境。

```bash
conda create --name kvcohort-dev --override-channels -c conda-forge \
  rocksdb=9.10 cmake=3.31 ninja pkg-config openssl -y
conda activate kvcohort-dev
bash scripts/build.sh
./build/kvc smoke "$PWD/var/smoke"
./build/kvc stats "$PWD/var/smoke"
```

测试只在 `var/tests/case-*` 创建的临时目录中注入错误，并清理对应测试目录。进程异常测试只退出其自行创建的测试子进程，不操作 GPU、不重启主机，不清空全局缓存。进程崩溃恢复不构成掉电耐久性实测。

存储契约测试包含 18 个案例，其中随机状态机为四种布局分别执行 500 步操作并与独立的内存参考状态比较；写入与 GC 各覆盖两个进程退出边界。另有 4 项回放 CLI 测试，校验输入、读取与搬移计量、容量拒绝及后续恢复。`scripts/build.sh` 通过 CTest 运行两个测试套件。可用 `KVCOHORT_TEST_ROOT="$PWD/var/tests" ./build/kvc_tests` 查看存储逐项结果。

默认数据根准入预算 64 MiB，其中预留 4 MiB GC 空间和 8 MiB 元数据余量；默认段 4 MiB，最大对象 1 MiB。对象上限是为了第一阶段验证，真实张量分片与性能参数尚待引擎接入后确定。

注意 RocksDB 的预分配 WAL 也占实际空间：本机空数据根的初始分配约 4.3 MiB。测试预算必须包含这部分，不应仅按张量大小设置。

## 布局与回放

```bash
python3 scripts/generate_trace.py var/traces/mixed.trace --rounds 12 --hints mixed
python3 scripts/compare_layouts.py var/traces/mixed.trace var/results/mixed \
  --payload-budget-kib 256 --segment-kib 64
```

输出目录须为新目录，脚本保留输入 SHA256、参数、每种布局原始 JSON 和 summary.tsv。默认逐一执行，避免同时竞争 SSD。示例使用真实文件后端，但载荷和访问均为合成，不执行 LLM，不测 TTFT。

`kvc` CLI 在命令前接受 `--layout joint --prefix-group DIGEST --lifetime short`。`--layout` 必须与数据根中保存的策略一致。`Put` 的 `PlacementHint` 不改变对象身份，重复写入不会迁移已存在对象。策略与回放脚本文档不在本次恢复的源码副本中。

## 后续实现顺序

1. 完善元数据增长预算、ENOSPC 与短写故障注入，以及容量长期压力测试。
2. 在现有四种布局基线上实现低开销寿命估计与联合代价控制；加入不准确标签、热点迁移和更多分支轨迹。
3. 实现缓存准入与有界异步任务，避免低复用负载仍无条件写入。
4. 实现受限 FUSE 接口与推理引擎适配，再进行 GPU 端到端评测。

更完整的设计与阶段结果文档不在本次恢复的源码副本中。
