# KVSeg可行性实验

**代码快照日期：2026-11-23**

这个压缩包整理了当前找回的 KV 缓存项目源码、vLLM 集成原型，以及在 2080 Ti 上使用的短输入和长输入实验代码。不同阶段的代码放在不同目录，便于识别用途。

## 目录说明

- `storage-core/`：C++17 存储核心、命令行程序、测试和布局回放脚本，来自恢复后的 `workspace/core` 源码目录。
- `vllm-integration/flash-attention-prototype/`：基于 vLLM 0.10.2 V1 的 FlashAttention 集成原型，含连接器、客户端和验证脚本。这是研究原型，不是可直接用于生产的完整插件。
- `benchmarks/2080ti-flexattention/short-input/`：2080 Ti 短输入实验脚本（512/2048 token，6 个场景），含该实验使用的连接器和缓存页脚本。
- `benchmarks/2080ti-flexattention/long-input/`：2080 Ti 长输入与并发实验脚本（4096/8192 token，7 个场景），含对应连接器和缓存页脚本。

短输入和长输入的 benchmark connector 是分别保留的实验版本；FlashAttention 原型也单独保留。它们接口和适用设置不同，请勿相互替换后就假设实验配置不变。

## 构建存储核心

存储核心使用 Linux、C++17、CMake、RocksDB 9.10 和 OpenSSL。推荐按 `storage-core/README.md` 创建 Conda 环境并构建：

```bash
cd storage-core
bash scripts/build.sh
```

构建脚本会在 `storage-core/build/` 生成构建产物并运行 CTest。若你在原服务器复现实验，先确认 Conda 环境和可用 GPU，再使用独立的新数据目录和 socket 路径。

## 运行基准脚本

两个 benchmark 脚本均提供 `prime`、`official`、`cpu`、`file` 模式，并要求指定本机模型目录、daemon socket、数据目录及输出 JSON 路径。例如：

```bash
python benchmark.py --mode official \
  --model /path/to/local/model \
  --socket /path/to/private/kvd.sock \
  --data-root /path/to/experiment/data \
  --output /path/to/results.json
```

脚本针对原实验环境编写，部分计量依赖 Linux `/proc`、NVMe 分区名和特定 vLLM API。迁移到其他主机时，需要先核对显卡、模型路径、vLLM/PyTorch 版本、daemon 路径和磁盘设备名；包内不含运行环境。

## 包内不包含

模型权重、KV 缓存或测试数据、Conda/Python 环境、编译产物、运行日志和结果文件均未打包。项目源码目录中的详细布局说明等文档在本次恢复副本中不存在，因此不在包内；本说明只描述包内实际文件。
