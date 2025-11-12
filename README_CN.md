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

```bash
python python/sfllm/serving/app.py \
  --model /path/to/your/model \
  --port 8081 \
  --dtype float16
```

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

## 开源许可

本项目基于MIT许可证开源 - 详见 [LICENSE](./LICENSE) 文件。

## 贡献

我们欢迎贡献！请随时提交问题和拉取请求。

---

**Made with ❤️ by [wejoncy](https://github.com/wejoncy)**