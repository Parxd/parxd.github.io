---
title: "a learning journey through ML systems"
date: 2026-06-24
tags: ["personal"]
---

As I approach a little over a year since formally starting my journey of self-learning and stumbling through the world of ML systems, I wanted to look back and recap / document my entire roadmap and the insane rabbit holes I've fallen through during the past year, and also outline what my goals are for the next couple months. 

The real story started in my sophomore year of college, when I started my project GroundUpNeuralNet, a *very* minimal implementation of an MLP library I wrote in C++ after taking a DSA class. At this point, I wasn't very concerned (or knowledgeable) about performance, so I leaned heavily on Eigen for the compute. Nevertheless, I still learned a good deal about multivariate calc, backprop, and just generally how to structure a relatively big C++ project.
You can still play around with it today [here](https://github.com/Parxd/GroundUpNeuralNet), although it only supports linear layers and a handful of activation + loss functions.

During my exploration of how to improve this project, I learned about the concept of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), or autodiff, which greatly opened my eyes as to how actual production ML libraries implement backprop. And of course, I wanted to write a toy autodiff system myself, which led to another project: [accelerate](https://github.com/Parxd/accelerate). Using NumPy + CuPy for the backend linear algebra, I wrote a simple autodiff library centered around a toy tensor implementation. Being based on autodiff instead of individually written complex gradients, this allowed for chaining of more complex ops. However, I was far too focused on designing an interface inspired by PyTorch: with the main `Tensor` abstraction, naturally calling overloaded operators that build the autodiff DAG, and ultimately calling `backward`  on some leaf tensor. 

Disappointingly, when I actually benchmarked my "library's" performance against PyTorch, I found it to be many magnitudes slower on the GPU, even though it used CuPy for the compute. Upon profiling training runs, I found many issues: poor core design choices (i.e. *very* tightly coupled interface and backend, Python-level DAG traversal during backprop) caused frequent device syncing, slow and unnecessary device memory movement, and individual kernel launch overhead to dominate. The code snippet below is directly from the project; you can see how *each* child node recursively backpropagating creates a new `Tensor` object, which is effectively a `cudaMalloc` call&mdash;not optimal.

```python {title="hello", style=onedark}
def backward(self, grad: Optional[Tensor] = None) -> None:
    if self.requires_grad:
        if grad is None:
            if self.ndim != 0:
                raise RuntimeError("grad implicitly created only for scalar outputs")
            else:
                grad = Tensor(1, device='cpu' if self.device is CPU else 'cuda')
        self.grad.data += grad.data
        if self.grad_fn:
            gradients = self.grad_fn(self.grad.data)
            for child, gradient in zip(self._children, gradients):
                # re-wrap incoming gradient array as Tensor
                child.backward(Tensor(gradient, device='cpu' if self.device is CPU else 'cuda'))
```

I thus turned to a lower level to manage memory myself: C++.

I began learning about the C++ CUDA API and common practices for efficiently managing device memory. I attempted to translate `accelerate` over to C++, handing over the compute to be done by cuBLAS and to have finer control over memory; however, I felt that I was missing something; clearly, performant frameworks like PyTorch were not manually allocating separate memory for every individual tensor on-the-fly during backprop. To delve deeper, I created yet another project: a toy clone of PyTorch's CUDA caching allocator ([zdevito's post](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html) was a great resource for this). 

![caching allocator notes](/images/posts/roadmap/notes.jpeg "Some notes... The key idea behind the allocator is maintaining a doubly linked list for fast merging and splitting of memory blocks, and an ordered set for fast retrieval of an appropriately-sized memory block when requested.")

While I learned a great deal, I *still* had major unanswered questions on the compute side.
- How does torch translate complex chained expressions that may not be as simple as GEMMs or some elementwise op. from its (backend-agnostic) interface to fast kernels? How are kernels fused to decrease launch overhead?
- How are fast kernels even written? Especially those that aren't covered by a BLAS library? What makes one implementation magnitudes faster than others?

As I would later learn, these are large and actively researched areas in the ML systems space: ML compilers and kernel engineering. Given the breadth of these fields, I chose to go down the kernel rabbit hole. I'll admit, I'm still stuck in this rabbit hole today and still have much more to learn, especially on kernels for LLMs. But starting from Simon Boehm's legendary GEMM optimization [blogpost](https://siboehm.com/articles/22/CUDA-MMM), I've come a long way in being able to answer how fast kernels are written. I've been trying to track my progress since day 1 the best I can with this [repo](https://github.com/Parxd/kernels). 

If you look at the commit history on it, you'll see a lot of time spent on just matmul kernels. While it really can't get any simpler algorithmically, optimizing GEMMs teaches so many fundamental lessons about GPU architecture and designing hardware-aware kernels. I'd highly recommend anyone interested in ML systems to try going from zero to SOTA on a GEMM kernel as a first step. 

I've also been up to a lot of other stuff&mdash;like this April, when I took a break from NVIDIA to check out the state of AMD accelerators. I participated in the AMD x GPU Mode e2e speedrun competition on the quantize + block-scaled `fp4` matmul track, where I placed in the top 20%, even as someone pretty new to HIP and AMD's ISA. Although it was quite a stressful crunch towards the end, I learned an incredible amount about programming AMD chips in a very short timeframe. I would highly recommend anyone getting into kernels to also participate in these [GPU Mode competitions](https://www.gpumode.com/home) to upskill fast.

Anyway, as for the future, I plan on contributing more to OSS and getting into kernels for LLMs and maybe even inference/model serving optimization. If you've made it this far, I hope this was of some help. The path of learning about these (or any) complex niche topics can be frustratingly unclear, with very little hand-holding, but I believe genuine curiosity and persistence tends to guide you to the right lessons eventually. For learning kernels specifically, the aforementioned GPU Mode Discord has a lot of great resources (shoutout to Mark Saroufim) like reading groups and spaces to just ask questions.