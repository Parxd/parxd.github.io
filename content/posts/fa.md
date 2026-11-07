---
title: "FlashAttention from first principles"
date: 2026-10-04
draft: false
tags: ["CUDA", "GPU", "math"]
---

The "naive" approach to this pre-FlashAttention was running one GEMM (general matrix-multiply) for $P := QK^T$, a softmax kernel (along with scaling by $\frac{1}{\sqrt{d}}$), and another GEMM for $O := PV$. However, for the target hardware we'd like to run this on (massively parallel processors like GPUs), this approach leads to underutilized resources and a much longer runtime for two main reasons:
- Loading/storing memory to/from off-chip DRAM at every intermediate step is very slow. Off-chip memory, while very large (on the order of GBs), has a much lower bandwidth than smaller on-chip memory like the programmable shared memory and non-programmable L1/L2 caches.
![img2](/images/posts/fa/2.png "Example memory hierarchy. Source: Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")
- Repeated kernel launches have overhead. While this overhead isn't horribly large for one kernel (usually in the low microseconds), across multiple attention layers and requests, it can quickly accumulate.

## Motivation
I mentioned that the naive approach leads to *underutilized resources* earlier. To understand, we have to look at the concept of arithmetic intensity. 

At a high level, every *implementation* of a numerical algorithm has an inherent property: its arithmetic intensity&mdash;or the number of floating-point operations performed per byte. For example, a routine that adds two FP32 vectors together has an intensity of 1 floating-point add per 12 bytes (reading 2 FP32 values and writing 1), resulting in $\frac{1}{12}$ FLOPs/byte.

GPUs themselves also have an analagous value. By dividing a GPU's peak compute throughput (simply&mdash;how fast it can perform floating-point/integer operations) by its max memory bandwidth, we get its ridge point. This point is where the performance of an algorithm running on the chip becomes bounded by the compute it needs instead of by its memory requirements.

Any algorithm with an intensity *less* than the point will theoretically not saturate all compute resources of the chip and burn cycles waiting on data; these are called **memory-bound**, while those that exceed the point are bounded by the on-chip resources and are called **compute-bound**.

Now let's look at the progression of memory bandwidth versus compute throughput of recent GPUs.

| GPU (year)       | HBM bandwidth | *Dense* FP16 tensor core throughput  | Ridge point      |
| ---              | :-:           | :-:                                  | :-:              |
| V100 (2017)      | 0.9 TB/s	   | 125 TFLOP/s                          | ~139 FLOPs/byte |
| A100 80GB (2020) | 2.0 TB/s	   | 312 TFLOP/s                          | ~153 FLOPs/byte |
|  H100 SXM (2022) | 3.35 TB/s     | ~989 TFLOP/s                         | ~295 FLOPs/byte |

- The third column is the peak throughput for matrix-multiplying FP16 tensors using tensor core units (TCs).
- The resulting ridge points on the right are thus *only* applicable to algorithms that use TCs.
  - For instance, looking at the example earlier about adding vectors, on an NVIDIA GPU, this would run exclusively on CUDA cores. On the A100 GPU, with its peak 19.49 TFLOPs/s FP32 throughput and 2 TB/s bandwidth, we'd get a ridge point of ~9.7 FLOPs/byte.
  - Quiz: Given the intensity of our vector add example, would it be memory- or compute-bound?
- The overarching point here is that ridge points are not only specific to each chip, but also the compute units used by the algorithm.

Getting back to the table, we can see an increasing gap between compute throughputs and memory bandwidths due to hardware limitations (google "AI memory wall" for more details). This leads to higher and higher ridge points, making it increasingly important to have arithmetically-intense algorithms to fully saturate GPUs' compute resources. Otherwise, we'd have tensor cores sitting idle just waiting on memory to arrive instead of using what we paid for :(

> [!note]- Example: Deriving a chip's compute throughput
> Let's derive the 312 TFLOPs/s figure for the A100 GPU from the table above:
> 1. With 108 SMs, each with 4 tensor cores, we have 432 tensor cores in total.
> 2. According to NVIDIA's [whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), each tensor core (TC) computes a $(M, N, K) = (8, 4, 8)$ FP16 matrix-multiply per cycle.
> 3. A matrix-multiply does a multiply-add of values along all 3 $M, N, K$ modes. Since a multiply-add is 2 FLOPs, each TC can execute 2 times MNK = 512 FLOPs/cycle.
> 4. Across the entire chip, this means 432 TCs times 512 = 221,184 FLOPs/cycle.
> 5. Finally, the A100's peak boost clock rate is 1.41 GHz, meaning the chip can execute 1.41 times 221,184 = **311,869 GFLOPs/second**.

You might be starting to see why the naive attention implementation from earlier is inefficient. Writing out $QK^T$ to HBM, reading it back in for the softmax, and so on is a pretty significant amount of memory movement. This is where FlashAttention comes in: what if we could *fuse* all of these operations and keep these intermediate matrices on-chip to cut the memory movement? With the same FLOPs, this would greatly increase arithmetic intensity. Let's step through how the paper introduced just this algorithm.

## FlashAttention-1

### Online Softmax
Recall the softmax of a vector $x \in \R^n$ of logits is:
$$ {x} = \begin{bmatrix} x_1 & x_2 & \dots & x_n \end{bmatrix} $$
$$ m({x}) = \text{max}({x}) $$
With re-shift term for numerical stability:
$$ f({x}) = \begin{bmatrix} e^{x_1 - m(x)} & e^{x_2 - m(x)} & \dots & e^{x_n - m(x)} \end{bmatrix}$$
$$ \text{softmax}({x}) = \frac{f({x})}{\sum_i e^{x_i - m(x)}} $$

Great, but we need the global maximum $m(x)$ of the entire vector $x$. Recall that attention requires $\text{softmax}(QK^T)$, where the softmax is row-wise, since we want the probabilities of how much each *query* attends to every *key* .

![img1](/images/posts/fa/1.png)

Reading the entire $K^T$ matrix to compute the matmul with $Q$, softmax, and matmul with $V$ onto on-chip memory is impossible. As seen in the earlier diagram, while on-chip memory is fast, it is much smaller in capacity than HBM. So, in typical GPU-algorithm fashion, we'll have to *tile* across $Q, K, V$ to compute small portions of the output at a time. Visualizing:
![img1](/images/posts/fa/3.png)

You might be able to immediately see a potential issue. If we compute $P$ tile-by-tile, how do we compute the row-wise softmax, which requires the global maximum? The paper introduces *online* softmax for this. To understand, consider a simpler case with 2 concatenated vectors: $x = \begin{bmatrix} x^{(1)} & x^{(2)}\end{bmatrix}$.

Suppose $m(x)$ is the max across both vectors, and $m(x^{(1)}), m(x^{(2)})$ are the maximums for $x^{(1)}, x^{(2)}$, respectively. Then the (elementwise) exponentiated numerator for the softmax is:
$$ f(x) = \begin{bmatrix} e^{x^{(1)} - m(x)} & e^{x^{(2)} - m(x)}  \end{bmatrix}$$

We can re-write the vectors in terms of their respective local maxes:
$$ f(x) = \begin{bmatrix} e^{ m(x^{(1)}) - m(x) } (e^{x^{(1)} - m(x^{(1)})}) & e^{ m(x^{(2)}) - m(x) } (e^{x^{(2)} - m(x^{(2)})}) \end{bmatrix}$$

and the same for the denominators $\ell(x^{(n)}) = \sum e^{x^{(n)} - m(x^{(n)})}$ (excuse my loose notation&mdash;this is referring to summation of the exponentiated vector $n$):
$$ \ell(x) = \ell ( \begin{bmatrix} x^{(1)} & x^{(2)} \end{bmatrix} ) = e^{ m(x^{(1)}) - m(x) } \ell (x^{(1)}) + e^{ m(x^{(2)}) - m(x) } \ell (x^{(2)})$$

...which suggests that it's possible do some sort of running renormalization of past vectors with the newly seen max. The algorithm keeps track of the currently seen max and denominator and constantly rescales all past outputs to exponents to be in "units" of the *newly* seen max.

### Algorithm
![img1](/images/posts/fa/4.png "FlashAttention-1 algorithm. Source: Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")

Up until line 10, you can see that the algorithm is just computing tile-local values. Note that $\tilde{P}_{ij}$ is still the unnormalized numerator; division happens later. Pretty standard, nothing new there. Line 11 is where the algorithm updates the running max and rescaled denominator for each row. The line:

$$ \ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \ell_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{\ell}_{ij} $$

...is what we saw in the earlier online softmax example. Line 12 is a bit packed, so I'll go over each term of the sum separately. For the $PV$ matmul:

$$ O_i \leftarrow \dots + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{P}_{ij}V_j $$

Since we just computed $\tilde{P}_{ij}$ in "units" of $\tilde{m}_{ij}$, we need to rescale it using the difference between it and the updated running max&mdash;we saw this earlier with $f(x)$. Also, keep in mind that this $PV$ matmul is a reduction along the sequence length, *not* the head dimension like in $QK^T$. Focusing on just one row $r$ of $\tilde{P}_{ij}$, where $k$ ranges from the tile's starting to ending index:

$$ O_i[r, :] \leftarrow \dots + e^{\tilde{m}_{ij}[r] - m_i^{\text{new}}[r]} \sum_{k} \tilde{P}_{ij}[r, k] \cdot V_{j}[k, :] $$
$$ \leftarrow \dots +  \sum_{k} e^{\tilde{m}_{ij}[r] - m_i^{\text{new}}[r]} (e^{S_{ij}[r, k] - \tilde{m}_{ij}} \cdot V_{j}[k, :]) $$
$$ \leftarrow \dots +  \sum_{k} e^{S_{ij}[r, k] - m_i^{\text{new}}[r]} \cdot V_{j}[k, :] $$

So we've shown that multiplying by the rescale factor after computing $PV$ *with the tile-local max* is equivalent to having just computed $PV$ with the running max.

Moving onto the preceding part of Line 12 now, the algorithm applies the renormalization (denominator), but must do so while also rescaling any previous tile's contributions to the running max.

$$ O_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1}(\text{diag}(\ell_i)e^{m_i - m_i^{\text{new}}}O_i + \dots)$$

$O_i$ at this point is the sum of all previous tiles' contributions, which means it's in units of $m_i$. Thus, we need to rescale it to $m_i^{\text{new}}$, as we may have seen a larger max in the current tile. After rescaling, we then *undo* the normalization done by $\ell_i$, so that we can directly add the unnormalized current tile's contribution. Left multiplying by the diagonalized $\ell_i$ vector here just indicates elementwise multiplying each row of $O_i$ with each entry in $\ell_i$.

Ok great. Now we have all tiles' (including the current one) unnormalized contributions in units of $m_i^{\text{new}}$. We've already computed the new denominator $\ell_i^{\text{new}}$ in Line 11, so all that's left is to take the reciprocal of its values, diagonalize it, and left-multiply it again. 

---

![img1](/images/posts/fa/5.png)

> [!NOTE]- A Simplification
> FlashAttention-1 puts the tiling across $K, V$ as the *outer* loop and across $Q$ as the *inner* loop, which means we technically process all $Q$ row tiles for each $K, V$ column tile before moving onto the next.
> 
> Thus, the visualization above is when the algorithm has already completed the first iteration of the outer loop. I'll focus on this first to highlight the renormalization logic, but revisit the $Q$ loop after.

