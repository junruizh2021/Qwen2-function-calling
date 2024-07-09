# Qwen2-function-calling

## Introduction

Qwen2-7B function calling demo with Intel dGPU accelerating

本项目是为了实现Qwen2-7B在使用Intel dGPU加速推理过程中的稳定工具调用能力

## function calling of Qwen model

目前Qwen2官方提供的function calling 有两种实现途径：

1. 基于 model.generate 接口（续写模式）的 [ReAct Prompting](https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_demo.py) 实现，但由于Qwen模型无法在generate中设置stop_words_ids，需要通过[配置StopWordsLogitsProcesssor实现对stop_words的支持](https://huggingface.co/Qwen/Qwen-72B-Chat/blob/main/modeling_qwen.py#L1250)

2. Qwen-Agent 提供了一个专用封装器，旨在实现通过 dashscope API 与 OpenAI API 进行的函数调用。

## Usage guide

下面是一些python文件和使用说明：

1. **qwen_generation_utils.py** : 配置StopWordsLogitsProcesssor实现对stop_words的支持
2. **react_demo.py** : 改造了Qwen react prompting实现，支持流式输出
，使用bigdl.llm.transformers加速在Intel dGPU上的推理
3. **qwen_langchain.py** : 先使用vllm启动兼容openai api的推理服务，再使用Qwen-Agent进行工具调用，稳定性强。其中，vllm支持在Intel dGPU上的推理加速。

react_demo.py 提供了function calling需要的LLM回答模板。由于稳定性不强，导致一些常识问题会直接回答，无法按模板格式提取Final Answer。
 - 方法1: 每轮对话清空历史记录，放弃上下文 (目前的逻辑) 
 - 方法2: 考虑特殊情况的处理