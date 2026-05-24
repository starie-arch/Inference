#KV Cache 缓存优化


## 1. 实验环境设置

首先设置实验环境，确保结果的可重现性。

```
import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import os

def pick_device() -> torch.device:
    forced = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if forced in {"mps", "cuda", "cpu"}:
        return torch.device(forced)

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reset_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def get_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1024**2
    return 0.0


device = pick_device()

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print(f"✅ 使用设备: {device}")
if device.type == "cuda":
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
elif device.type == "mps":
    print("✅ 当前为 Apple Silicon MPS 模式，将使用 M1 系列芯片加速。")
else:
    print("⚠️ 当前为 CPU 模式，性能实验会较慢。")

# 加载本地 Qwen 模型和分词器
default_model_path = Path(__file__).resolve().parent.parent / "local_models" / "Qwen2.5-0.5B-Instruct"
model_name = Path(os.environ.get("MODEL_NAME", str(default_model_path))).expanduser().resolve()
if not model_name.exists():
    raise FileNotFoundError(f"本地模型路径不存在: {model_name}")
if not (model_name / "config.json").exists():
    raise FileNotFoundError(f"本地模型目录缺少 config.json: {model_name}")

print(f"🚀 加载本地模型: {model_name}")
try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_name), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_name),
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
except Exception as e:
    raise RuntimeError(
        "本地 Qwen 加载失败。请确认当前 Python 环境安装了支持 Qwen2 的 transformers，"
        "建议使用 transformers>=4.43，或切换到项目的 vllm_metal_py312 环境运行。"
    ) from e

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ 模型 {model_name} 加载完成")

generate_length = int(os.environ.get("GENERATE_LENGTH", "100"))
temperature = float(os.environ.get("TEMPERATURE", "0.7"))
print(f"✅ 生成步数 generate_length: {generate_length}")


## 2. KVCache 技术原理

在 Transformer 的自注意力机制中，每个输入序列都需要计算键(Key)和值(Value)向量。对于生成任务，当我们逐步生成 token 时，重复计算先前所有 token 的 KV 值会导致大量冗余计算。

KVCache 的核心思想是将先前计算过的 KV 值存储起来，避免在生成新 token 时重复计算。数学上，自注意力机制可以表示为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q$, $K$, $V$ 分别表示查询(Query)、键(Key)和值(Value)矩阵。在生成过程中，只有最新 token 的 $Q$ 需要与所有先前 token 的 $K$ 和 $V$ 进行计算。

大语言模型生成文本的核心原理是基于深度学习技术，通过训练大规模语料库来学习语言规律，并生成具有相似统计特征的新文本。这些模型的核心是建立一个统计模型，用来估计文本序列中每个词语或字符出现的概率。


## 3. 关闭 KVCache

在第一个实验中，我们完全关闭 KVCache 功能，每次生成新 token 时都重新计算所有先前 token 的 KV 值。这种方法计算效率最低，但可以帮助我们理解 KVCache 的价值。


```
def generate_without_kv_cache(model, input_ids, max_length=50):
    """
    不使用 KVCache 的生成函数
    每次生成新 token 时都重新计算所有先前 token 的 KV 值
    """
    generated = input_ids
    past_key_values = None  # 明确不使用缓存
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(
                generated, 
                past_key_values=past_key_values,
                use_cache=False  # 强制不使用缓存
            )
            
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)
        
        # 始终不使用缓存，所以 past_key_values 保持为 None
        
    return generated

# 准备输入
input_text = "深度学习中的注意力机制是"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 测量显存和延迟
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# -----------------------------
# 单步生成（无 KVCache）
# -----------------------------
start_time = time.time()
latencies = []
mem_usages = []

output_ids = input_ids.clone()

print("🚀 开始生成（关闭 KV Cache）...")

for i in range(generate_length):
    t0 = time.time()
    with torch.no_grad():
        outputs = model(output_ids)
        next_token_logits = outputs.logits[:, -1, :] / temperature
        # 使用采样（避免死循环）
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    output_ids = torch.cat([output_ids, next_token], dim=1)

    latency = time.time() - t0
    latencies.append(latency)
    mem_usages.append(torch.cuda.max_memory_allocated(device) / 1024**2)

end_time = time.time()

# -----------------------------
# 结果统计
# -----------------------------
avg_latency = np.mean(latencies)
max_mem = max(mem_usages)
total_time = end_time - start_time

decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("✅ 实验完成（关闭 KV Cache）")
print(f"平均推理耗时: {avg_latency:.4f} 秒 / token")
print(f"峰值显存使用: {max_mem:.2f} MB")
print(f"总耗时: {total_time:.2f} 秒")
print(f"生成文本片段:\n{decoded[:150]}...")

# -----------------------------
# 可视化：显存与延迟趋势
# -----------------------------
fig, ax1 = plt.subplots(figsize=(7,4))
ax2 = ax1.twinx()

ax1.plot(mem_usages, color="#5DADE2", label="显存 (MB)")
ax2.plot(latencies, color="#E74C3C", label="延迟 (s)")

ax1.set_xlabel("生成步数")
ax1.set_ylabel("显存 (MB)")
ax2.set_ylabel("延迟 (s)")
plt.title("关闭 KV Cache 的推理性能趋势")
ax1.grid(True, linestyle="--", alpha=0.5)
plt.show()
```

    🚀 开始生成（关闭 KV Cache）...
    ✅ 实验完成（关闭 KV Cache）
    平均推理耗时: 0.0111 秒 / token
    峰值显存使用: 430.81 MB
    总耗时: 0.56 秒
    生成文本片段:
    深 度 学 习 中 的 注 意 力 机 制 是 一 种 机 制 吗 ？ 我 是 一 名 在 美 国 的 科 学 家 ， 最 近 在 做 深 度 学 习 的 时 候 ， 发 现 有 一 些 一 些 注 意 力 机 制 需 要 特 别 注 意 ， 比 如 何...



    
![png](Code01KVCache_files/Code01KVCache_5_1.png)
    


这个实验展示了最基础的生成方式，每次都需要重新计算整个序列的注意力，计算复杂度为 $O(n^2)$，其中 n 是序列长度。大语言模型通过概率方法生成文本，即根据输入或上下文为每个可能的词或句子分配一个概率，然后选择概率最高的词或句子，或者从概率分布中采样，来生成输出文本。

## 4. 开启 KVCache

现在，我们启用 KVCache 功能。这将显著减少计算量，因为只需要计算最新 token 的注意力权重。


```
def generate_with_kv_cache(model, input_ids, max_length=50):
    """
    使用 KVCache 的生成函数
    缓存先前计算的 KV 值以避免重复计算
    """
    generated = input_ids
    past_key_values = None  # 初始化为 None
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(
                generated, 
                past_key_values=past_key_values,
                use_cache=True  # 启用缓存
            )
            
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)
        
        # 更新 KVCache 以供下一次迭代使用
        past_key_values = outputs.past_key_values
        
    return generated

# 测量显存和延迟
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# -----------------------------
# 启用 KV Cache 生成
# -----------------------------
output_ids = input_ids.clone()
past_key_values = None
latencies = []
mem_usages = []

print("🚀 开始生成（开启 KV Cache）...")

start_time = time.time()

for i in range(generate_length):
    t0 = time.time()
    with torch.no_grad():
        outputs = model(
            output_ids[:, -1:],  # 只输入上一步生成的 token
            past_key_values=past_key_values,
            use_cache=True
        )
        next_token_logits = outputs.logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        past_key_values = outputs.past_key_values  # 更新缓存
        output_ids = torch.cat([output_ids, next_token], dim=1)

    latency = time.time() - t0
    latencies.append(latency)
    mem_usages.append(torch.cuda.max_memory_allocated(device) / 1024**2)

end_time = time.time()

# -----------------------------
# 结果统计
# -----------------------------
avg_latency = np.mean(latencies)
max_mem = max(mem_usages)
total_time = end_time - start_time
decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("✅ 实验完成（开启 KV Cache）")
print(f"平均推理耗时: {avg_latency:.4f} 秒 / token")
print(f"峰值显存使用: {max_mem:.2f} MB")
print(f"总耗时: {total_time:.2f} 秒")
print(f"生成文本片段:\n{decoded[:150]}...")

# -----------------------------
# 可视化：显存与延迟趋势
# -----------------------------
fig, ax1 = plt.subplots(figsize=(7,4))
ax2 = ax1.twinx()

ax1.plot(mem_usages, color="#58D68D", label="显存 (MB)")
ax2.plot(latencies, color="#C0392B", label="延迟 (s)")

ax1.set_xlabel("生成步数")
ax1.set_ylabel("显存 (MB)")
ax2.set_ylabel("延迟 (s)")
plt.title("开启 KV Cache 的推理性能趋势")
ax1.grid(True, linestyle="--", alpha=0.5)
plt.show()

```

    🚀 开始生成（开启 KV Cache）...
    ✅ 实验完成（开启 KV Cache）
    平均推理耗时: 0.0074 秒 / token
    峰值显存使用: 421.29 MB
    总耗时: 0.37 秒
    生成文本片段:
    深 度 学 习 中 的 注 意 力 机 制 是 一 种 ， 你 会 发 现 自 己 在 一 个 人 心 里 是 那 么 的 孤 独 。 他 会 在 你 身 边 ， 但 你 不 知 道 他 有 多 孤 独 。 有 时 候 你 不 知 道 他 说 的 对 不...



    
![png](Code01KVCache_files/Code01KVCache_7_1.png)
    


使用 KVCache 后，计算复杂度降低到 $O(n)$，因为只需要计算最新 token 与所有缓存 key 的点积。但是，KVCache 可能占用大量显存，尤其是对于长序列。大语言模型具有上下文感知能力，可以根据上下文信息进行文本生成和理解，从而更好地适应不同的语言环境。

