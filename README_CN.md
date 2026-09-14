# SFLLM: 高性能大语言模型推理框架

[🇺🇸 English](./README.md) | [🇨🇳 中文文档](./README_CN.md)

一个生产就绪的高性能大语言模型推理框架，提供与OpenAI兼容的API接口。

## 项目背景

SFLLM (Serving Framework for Large Language Models) 旨在为大语言模型提供高效可扩展的推理服务。项目专注于通过智能批处理、CUDA优化和内存高效实现来最大化GPU利用率并降低推理延迟。

## 功能特性

- **OpenAI兼容API**: 完全兼容OpenAI API端点，可直接替换使用
- **高性能**: 通过智能请求批处理优化推理性能
- **流式支持**: 实时流式响应，提供更好的用户体验
- **CUDA优化**: CUDA图和自定义内核实现最大性能
- **内存高效**: 优化的KV缓存管理和内存分配
- **生产就绪**: 内置健康检查和错误处理机制
- **Eagle3投机解码**: 采用先进的Eagle3算法进行投机解码，显著提升生成速度
- **重叠调度**: 智能的计算与通信重叠调度，提高整体吞吐量
- **Eagle3 CUDA图加速**: 结合CUDA图优化的Eagle3实现，极致性能表现

## 安装

### 环境要求

- Python 3.8+
- CUDA 11.8+ (GPU加速需要)
- PyTorch 2.0+

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/wejoncy/gemma_serving.git
cd gemma_serving

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 快速开始

### 1. 启动服务器

**基础用法：**
```bash
python python/sfllm/serving/app.py \
  --model /path/to/your/model \
  --port 8081 \
  --dtype float16
```

**启用Eagle3投机解码：**
```bash
python python/sfllm/serving/app.py \
  --model /path/to/your/model \
  --speculative-draft-model-path /path/to/eagle3/draft/model \
  --speculative-algorithm eagle3 \
  --speculative-num-steps 4 \
  --port 8081 \
  --dtype float16
```

**Qwen3.8-27B 文本推理：**

```bash
hf download Qwen/Qwen3.8-27B --local-dir /mnt/data/work/Qwen3.8-27B
python python/sfllm/serving/app.py \
  --model /mnt/data/work/Qwen3.8-27B --dtype bfloat16 \
  --attention-backend fa3 --max-running-requests 8 --cuda-graph-max-bs 8 \
  --port 8081
```

独立的 [`qwen3_8.py`](python/sfllm/models/qwen3_8.py) 入口继承 Qwen3.5 实现。
官方检查点按配置声明的 `Qwen3_5ForConditionalGeneration` 架构加载，
加载语言模型，跳过视觉编码器和 MTP 权重。
该 BF16 配置可在 H100 96 GB 上运行，默认使用 FP32 循环状态和 FlashInfer GDN。
支持通过聊天模板参数控制思考模式，详见 [Qwen3.8 使用说明](docs/qwen3_8.md)。

### 2. 测试API

**聊天补全（流式）**
```bash
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [
      {"role": "user", "content": "你好，你怎么样？"}
    ],
    "stream": true,
    "max_new_tokens": 256,
    "temperature": 0.7
  }'
```

**文本补全**
```bash
curl -X POST "http://localhost:8081/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "prompt": "人工智能的未来是",
    "max_new_tokens": 128,
    "temperature": 0.8
  }'
```

### 3. 健康检查

```bash
curl http://localhost:8081/health
```

## 配置选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--model` | 模型目录路径 | 必需 |
| `--port` | 服务器端口 | 8081 |
| `--dtype` | 模型精度 (float16/float32) | float16 |
| `--max-context-length` | 最大上下文长度 | 4096 |
| `--cuda-graph-max-bs` | CUDA图最大批处理大小 | 32 |
| `--disable-cuda-graph` | 禁用CUDA图 | False |
| `--enable-prefill-cuda-graph` | 捕获 prefill 全图（Triton 或 FA3；GDN 需 FlashInfer） | False |
| `--prefill-cuda-graph-sizes` | prefill 图的总 token 容量档位 | 到 256 步长 32；到 512 步长 64；到 2048 步长 128 |
| `--speculative-algorithm` | 投机解码算法 (`eagle3`、`dflash2` 或 `dspark`) | None |
| `--speculative-draft-model-path` | 投机解码草稿模型路径 | None |
| `--speculative-num-steps` | 投机解码步数 | 4 |
| `--disable-overlap` | 禁用重叠调度 | False |

prefill 根据 batch 的新增 token 总数选择能容纳它的最小捕获档位。
超过 1024 tokens 时，最多补 16 tokens，补充量更大则走 eager；1024 以内
沿用已有档位间距。同一档支持 `--max-running-requests` 范围内的不同请求数
和不等长序列。默认 24 档在小 token 区间更密：129 tokens 使用 160 档（补 31）；
1034 tokens 走 eager；1136 tokens 使用 1152 档（补 16）。
可用 `--prefill-cuda-graph-sizes` 自定义档位。默认上限 2048 仅限制图的使用范围，
不限制请求或上下文长度；更大的 prefill 使用 eager。Qwen3 可通过
`--attention-backend triton --enable-prefill-cuda-graph` 启用，也支持 FA3。
Qwen3.5 等包含 GDN 的模型另外需要 `--linear-attn-prefill-backend flashinfer`。
不会自动切换 backend 或 SSM 精度。token padding 可能改变 GEMM/attention 的
浮点舍入及生成 token，因此不保证与 eager 逐位相同。

## 开源许可

本项目基于MIT许可证开源 - 详见 [LICENSE](./LICENSE) 文件。

## 贡献

我们欢迎贡献！请随时提交问题和拉取请求。

---

**Made with ❤️ by [wejoncy](https://github.com/wejoncy)**
