---
title: "Back-of-the-napkin LLM inference"
date: 2026-10-28
comment: true
draft: false
tags: ["math", "llm"]
---

This post will informally go over some simple derivations that roughly estimate the computational complexities and behavior of LLM inference. 

The most time-consuming layers of almost every LLM are its MLPs, or fully-connected layers, and the attention layers. We'll thus focus our analysis on these two, seeing how tweaking certain variables introduces trade-offs.

## foundations
To start, some foundations:
- A general matrix-multiply of shape $[M, N] \times [K, N] = [M, N]$ requires $2MNK$ FLOPs, since it's a multiply-add across $K$, for all $[M, N]$ pairs.
- Compute- vs. memory-bound (see my FlashAttention post for a more in-depth treatment)
    - Compute-bound refers to an implementation of algorithm being so arithmetically intense that the compute resources (i.e. math units) on the hardware can't keep up (i.e. increasing memory bandwidth won't improve your program's performance)
    - Memory-bound refers to the opposite; an implementation that has such little "math" done per byte moved that moving memory across levels of the hardware becomes the bottleneck, causing math units to stall
    - A program being one or the other defines its throughput *ceiling* and its latency *floor*
        - An estimate for the absolute minimum latency (assuming perfect overlap of compute and memory ops. and an optimal algorithm) is:
        $$
        \text{latency} \geq \max\left(
            \underset{\text{compute-bound}}{\underbrace{\frac{\text{FLOPs}}{\text{peak compute}}}},
            \underset{\text{memory-bound}}{\underbrace{\frac{\text{bytes transferred}}{\text{peak bandwidth}}}}\right)
        $$
- Keys and values of all seen tokens in a sequence can be cached in autoregressive generation, where tokens are predicted strictly left to right.
    - Past tokens can't attend to future ones, so these past tokens' keys and values will remain unchanged across all attention layers, regardless of new tokens (more on this later).
- Prefill and decode refer to 2 distinct "phases" during generation that have pretty different properties. Simplifying things a bit:
    - Prefill is the consumption of your initial prompt, along with any preceding system prompt. This is a "one-time" cost. 
    - Decode is the step-by-step generation of new tokens from the model. Generally speaking, an inference engine will run decode steps until the model outputs an "end-of-sequence" token, saying that it's done, or until it hits some max output token limit. The nature of decode makes it difficult to fully utilize compute resources on GPUs, as we'll see soon.

## decode
Let's estimate the FLOPs for decode. We'll assume our engine doesn't use any fancy optimizations and uses *static batching*, where a set number of requests (can still vary in length amongst themselves) are handled together before accepting a new batch of the same size.

Our initial input to the model is thus $X: [b, 1, d]$, where $b$ is the batch size and $d$ is the hidden dimension size of the model. The 1 indicates every decode step is fed just 1 token per batch.

---

### attention layers
With $s$ as the sequence length of the existing KV cache:
$$Q: [b, 1, q, h]$$
$$K, V: [b, s, k, h]$$

For $QK^T$, each query head costs $2bhs$ FLOPs, for a total of $2bhsq$.

$P := \text{softmax}(QK^T)$ using online softmax requires a row-wise max, subtraction, exponentiation, sum, and division, *roughly* costing $5s$ FLOPs per-batch and per-query head.

For $O := PV$, the reduction happens over $s$, instead of $d$ like in $QK^T$, but the FLOPs remain identical: $2bhsq$.

So, for *one* attention layer, we have a total of $2bhsq + 5bqs + 2bhsq$ FLOPs. For our analysis, we'll drop the softmax part, as it's dominated by the 2 matmuls. Taking the ratio $\frac{5bsq}{4bhsq} = \frac{5}{4h}$ estimates that it's a minor fraction of the total FLOPs, since usually $h > 64$.

Now let's count the bytes, using `bf16` as the dtype. $Q, O$ together would then be $4bqh$ bytes, and $K, V$ together would be $4bskh$ bytes. Note we're only counting the bytes transferred between HBM and the GPU, hence why $P$ is not included--thanks to FlashAttention!

The arithmetic intensity (FLOPs/byte) of the entire attention layer is then:

$$ \frac{4bhsq}{4bqh + 4bskh} = \frac{sq}{q + sk} = \frac{s}{1 + \frac{sk}{q}} \approx \frac{q}{k}$$

We dropped the 1 in the denominator because $s$ is almost always very large (think about how many output tokens you've burned on Claude 😉). So, as the number of KV heads drops below that of query heads, arithmetic intensity rises. Intuitively, this makes sense, as if a KV head is shared by more query heads, we're doing more "math per byte". We'd also save on the memory requirements for the KV cache, hence why GQA and MQA are a thing. MQA is the extreme form of GQA where all query heads share the same KV head, but this tends to sacrifice model quality. I won't go into more detail on GQA vs. MQA, but their respective papers are fairly easy to digest.

---

### MLP layers
The fully-connected layers are fairly straightforward. We'll assume a Llama-style model that uses a gated linear unit. 

![img1](/images/posts/napkin-llm/1.png "Fig. 1: Example SwiGLU (swish-gated linear unit, not to scale)")

We can see we have 3 big matmuls. We'll drop the SiLU activation function and elementwise product from the FLOP counts, since these are again dwarfed by the matmuls. Also note the new variable $i$ here, representing the intermediate dimension, which is typically a couple magnitudes greater than $d$ (e.g. in Llama-3 70B, $d = 8192$ and $i = 28672$). 

All 3 matmuls cost a combined $6bdi$ FLOPs, for a total of $2(2bd + 3di)$ bytes in `bf16` (extra $bd$ from the out matrix). The arithmetic intensity is then:

$$ \frac{6bdi}{4bd + 6di} = \frac{6bi}{4b + 6i} = \frac{3i}{2 + \frac{3i}{b}} \approx b $$

Arithmetic intensity of the MLP layers is largely determined by the batch size, not the hidden or intermediate dimensions. Again, intuitively, this makes sense. As batch size increases, the "reuse" of the weight matrices improves and we're performing more compute per byte moved.

---

### the big picture?
We've approximated the arithmetic intensity of the attention layers as $\frac{q}{k}$ and MLP layers as ${b}$. Now what? 

We can now comfortably state that attention layers at decode are memory-bound, regardless of batch size. The highest *realistic* $q/k$ ratios seen in real models are around 8 to 16, which is far below almost every modern GPUs' ridge point.

But what about the MLPs? If we crank up batch size way past a GPU's ridge point, we could at least make this part compute-bound, right? Technically yes, but compute bound ≠ good, and blindly attempting to make these layers compute-bound affects *latency* and *throughput*, which are very important for user interactivity and utilizing the hardware effectively.

---

### tuning batch size
Let's see how tweaking $b$ affects per-request *latency* of attention and MLP layers for a decode step. Remember from above that the latency floor of a memory-bound program can be estimated as $\frac{\text{bytes}}{\text{bandwidth}}$. We already know decode attention is memory bound, and also found from earlier that attention transfers $4bqh + 4bskh$ bytes. Thus, its latency rises directly proportional to $b$.

Now assume a memory-bound MLP, with a $b$ lower than the ridge point. We found that these require $2bd + 6di$ bytes. You might be able to immediately tell that the $6di$ term will dominate, meaning $b$ will play less of a role in determining the latency floor. As an example, let's take the ratios between an MLP using $b$ and another using $2b$.

$$ \frac{8bd + 6di}{\text{bandwidth}} \times \frac{\text{bandwidth}}{4bd + 6di} = \frac{8b + 6i}{4b + 6i} = \frac{4 + \frac{3i}{b}}{2 + \frac{3i}{b}} \approx 1$$

Dropping the constant terms again, we find that latency is *approximately* unaffected* by doubling $b$, unlike attention. Intuitively thinking, since each sequence in a batch requires its own KV cache, the byte count (and thus latency) grows with $b$. For MLPs, every sequence can just share the same weights for the matmuls. 

Now we'll analyze throughput, which is in units of tokens per unit of time, for the entire model--not just individual layers. Since one decode step generates exactly 1 token, this is $b$ tokens. If $t_{\text{MLP}}, t_{\text{Attn}}$ represent the time taken for all MLP and attention layers respectively per step, then throughput is:

$$ \frac{b}{t_{\text{MLP}} + t_{\text{Attn}}} $$

Like we did for latency, we'll double $b$ and take their ratios. With $2b$, throughput is:

$$ \frac{2b}{t_{\text{MLP}} + 2t_{\text{Attn}}} $$

$$ \frac{2b}{t_{\text{MLP}} + 2t_{\text{Attn}}} \times \frac{t_{\text{MLP}} + t_{\text{Attn}}}{b} = \frac{2t_{\text{MLP}} + 2t_{\text{Attn}}}{t_\text{MLP} + 2t_{\text{Attn}}}$$

This tells us a couple things:
- When significantly more time is spent on MLP layers (i.e. $t_{\text{MLP}} \gg t_{\text{Attn}}$), the ratio can be approximated to 2. Doubling batch size in this regime is essentially free; it nearly doubles throughput, and *total latency* per step is barely affected since attention is still a small fraction of the step.
- When $t_{\text{MLP}} = t_{\text{Attn}}$, the ratio drops to 1.33. Throughput still increases here, but the growing latency of the attention layers begins to catch up. This crossover is where the total bytes for the MLP weights equals that of the KV caches for attention layers.
- Lastly, with $t_{\text{MLP}} \ll t_{\text{Attn}}$, the ratio expectedly flatlines to 1.

The below plots throughput against various batch sizes (code used to generate plot in Appendix). 

![img2](/images/posts/napkin-llm/2.png "Fig. 2: Example Llama-3 70B on an H100")

> [!note]+ 
> *I said MLP layer latency is unaffected by doubling $b$ above...
> 
> This only holds because we assumed the MLPs are memory-bound. When $b$ is pushed to make them compute-bound, past the ridge point of the hardware (for reference, an H100's ridge point is ~290 FLOPs/byte), latency becomes directly proportional to $b$. This results in $t_{\text{MLP}}$ doubling with $b$, just like $t_{\text{Attn}}$, further pushing the throughput ratio down to 1.

---

### other variables
Seeing as how the size of the KV cache heavily influences decode latency and throughput, we can analyze the other variables besides $b$ that define its total byte count, and see how we could potentially reduce them:
- $s$ (sequence length)
    - Sliding-window attention retains a fixed subset of recent KV entries.
    - Top-K token selection can dynamically select a relevant subset of KV entries. 
- $k$ (KV heads)
    - Drop the number of KV heads even further at the cost of model complexity
    - DeepSeek's multi-head latent attention (MLA), which replaces the concept of distinct KV heads entirely with a shared compressed latent space.

Another optimization is quantizing the entire cache to a lower precision dtype. Going from `bf16` to `fp8` directly halves the bytes, so performance gains are expected there too.

## Appendix
Code used to generate Fig. 2:
```python
import matplotlib.pyplot as plt

peak_compute = 989_000_000_000_000  # H100: 989 TFLOP/s
peak_bandwidth = 3_350_000_000_000  # H100: 3.35 TB/s
W_size = 100_000_000_000  # Llama3 70B: 100 GB in bf16 (MLP-only)
num_params = W_size / 2
attn_layers = 80
kv_heads = 8
head_dim = 128
seq = 2_048

throughputs = []
no_attn_throughputs = []
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for batch_size in batch_sizes:
    kv_cache_size = batch_size * (attn_layers * seq * kv_heads * head_dim * 2 * 2)
    t_attn = kv_cache_size / peak_bandwidth
    t_mlp = max(W_size / peak_bandwidth, (2 * batch_size * num_params) / peak_compute)
    throughputs.append(batch_size / (t_attn + t_mlp))
    no_attn_throughputs.append(batch_size / t_mlp)

plt.plot(batch_sizes, throughputs, label="full step")
plt.plot(batch_sizes, no_attn_throughputs, label="attention-free (MLP only)")
plt.axhline(
    peak_bandwidth / (attn_layers * seq * kv_heads * head_dim * 2 * 2),
    color="red",
    label="MLP-free (attention only)",
)

plt.legend()
plt.xscale("log", base=2)
plt.xlabel("Batch Size")
plt.ylabel("Throughput (tok/s)")
plt.grid(True)
plt.show()

```