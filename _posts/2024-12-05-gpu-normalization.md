---
layout: post
title: "Image Normalization: Comparing CPU vs GPU Performance in PyTorch"
subtitle: "Understanding the performance differences in Pytorch's image normalization"
author: Miguel Mendez
description: "A profiling comparison between CPU and GPU performance when normalizing images in PyTorch. There are several factors to consider when optimizing preprocessing pipelines, such as data types, data transfer, and parallel processing capabilities. This posts explores these factors and provides insights on how to optimize your data pipeline."
image: "/assets/images/fullsize/posts/2024-12-05-gpu-normalization/thumbnail.jpg"
selected: y
mathjax: n
tags: [PyTorch, GPU, Performance, Computer Vision, Python]
categories: [Deep Learning, Performance, CUDA]
---

This post has been on my to-do list for a long time, and I’m excited to finally have the time to write it. I hope you find it useful because it’s about a topic I find very interesting.  

It all started while I was testing some of our internal data preprocessing pipelines and began thinking about the performance of the normalization step. Image normalization is a straightforward process: subtract the mean and divide by the standard deviation. These operations are well-suited for parallel processing, which is where GPUs excel. However, there are a few other factors to consider, such as data types and data transfer between the CPU and GPU.

In this post, I’ll compare the performance differences between CPU and GPU when normalizing images, and explore the impact of data types and data transfer. Let’s get started!

## Setup

Let's start by creating some test data and importing the necessary libraries:

```python
import torch
from torchvision.transforms import v2
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.utils.data import Dataset, DataLoader

batch_size = 10
height, width = 720, 1280
# Create test images as uint8
img_uint8 = torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.uint8)
# Create float version since torch.Normalize expects float input
img_float = img_uint8.float()

# ImageNet normalization values
mean = torch.tensor([123.675, 116.28, 103.53])
std = torch.tensor([58.395, 57.12, 57.375])
```

To profile our operations, we'll use PyTorch's built-in profiler. This is just because I wanted to play with it and this looked like a good opportunity. We'll define a custom profiling function to simplify the process:

```python
def custom_profile(func, activities, times=10):
    with profile(activities=activities, record_shapes=True) as p:
        for _ in range(times):
            func()
            p.step()
    print(p.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

## Basic CPU Normalization

Let's start by implementing our normalization function and profiling it on the CPU:

```python
def my_normalize(image, mean, std):
    # Reshape mean and std for broadcasting
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return (image - mean) / std

custom_profile(
    lambda: my_normalize(img_uint8, mean, std), 
    activities=[ProfilerActivity.CPU]
)
```

The profiler reveals some interesting insights:

```plaintext
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
              aten::sub        36.69%     643.926ms        67.74%        1.189s     118.902ms            10  
              aten::div        32.25%     565.990ms        32.25%     565.990ms      56.599ms            10  
               aten::to         0.00%      73.817us        31.06%     545.093ms      54.509ms            10  
         aten::_to_copy         0.02%     324.441us        31.05%     545.019ms      54.502ms            10  
            aten::copy_        31.01%     544.341ms        31.01%     544.341ms      54.434ms            10  
    aten::empty_strided         0.02%     354.039us         0.02%     354.039us      35.404us            10  
             aten::view         0.01%     225.351us         0.01%     225.351us      11.268us            20  
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.755s
```

- The basic operations (`aten::sub` and `aten::div`) account for most of the execution time.
- A significant portion of time is also spent on type conversion operations (`aten::to`, `aten::_to_copy`).
- These conversions occur because the input is `uint8`, while the mean and standard deviation are `floats`. So PyTorch converts the input to `float` before performing the operations.

Now, let’s compare the performance when using `float` as input instead:

```python
custom_profile(
    lambda: my_normalize(img_float, mean, std), 
    activities=[ProfilerActivity.CPU]
)
```

```plaintext
--------------  ------------  ------------  ------------  ------------  ------------  ------------  
          Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
--------------  ------------  ------------  ------------  ------------  ------------  ------------  
     aten::sub        50.19%     569.169ms        50.19%     569.169ms      56.917ms            10  
     aten::div        49.79%     564.562ms        49.79%     564.562ms      56.456ms            10  
    aten::view         0.02%     203.964us         0.02%     203.964us      10.198us            20  
--------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.134s
```

This is much faster! Here’s our first takeaway: always use the correct data types to avoid unnecessary conversions. Casting between types is costly and can significantly impact performance.

At this point I became quite interested in knowing how my simple function would compare to PyTorch’s built-in normalization function. So I couldn't resist the temptation to compare them.
Note that I have used the new torchvision transforms v2 API:

```python
torch_normalize = v2.Normalize(mean, std)
custom_profile(lambda: torch_normalize(img_float), activities=[ProfilerActivity.CPU])
```

```plaintext
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    aten::sub        87.41%     785.070ms        87.41%     785.070ms      78.507ms            10  
                   aten::div_        12.21%     109.692ms        12.21%     109.692ms      10.969ms            10  
                   aten::item         0.30%       2.692ms         0.33%       2.921ms      32.458us            90  
             aten::is_nonzero         0.02%     136.106us         0.32%       2.899ms      96.634us            30  
    aten::_local_scalar_dense         0.03%     229.645us         0.03%     229.645us       2.552us            90  
                   aten::view         0.02%     169.587us         0.02%     169.587us       8.479us            20  
                  aten::empty         0.02%     161.253us         0.02%     161.253us       8.063us            20  
                     aten::to         0.00%      14.746us         0.00%      14.746us       0.737us            20  
             aten::lift_fresh         0.00%      12.286us         0.00%      12.286us       0.614us            20  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 898.176ms
```

We can spot a few extra operations because PyTorch checks that the types and dimensions are correct (along with other performance optimizations, as seen in the [source code](https://github.com/pytorch/vision/blob/6279faa88a3fe7de49bf58284d31e3941b768522/torchvision/transforms/v2/functional/_misc.py#L19){:target="_blank"}{:rel="noopener noreferrer"}). 

You might notice a small difference: there’s an `aten::div_` operation instead of `aten::div`. This happens because PyTorch’s `Normalize` function does the division in-place, while our custom function does not. Still, the main operations are the same, and the performance is very similar.  

This means we’ve mostly achieved the same behavior. Great! Now, let’s see how it runs on the GPU.  


## GPU Implementation

We only need a simple adjustment to run our code on the GPU. Specifically, we need to move our tensors to the GPU and ensure the operations execute there. Additionally, I’ve included CUDA in our profiler activity:

```python
img_gpu = img_float.cuda()
mean_gpu = mean.cuda()
std_gpu = std.cuda()

custom_profile(
    lambda: my_normalize(img_gpu, mean_gpu, std_gpu),
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
)
```

The profiler output shows the following:

```plaintext
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                  cudaDeviceSynchronize        85.85%      20.758ms        85.85%      20.758ms      20.758ms       0.000us         0.00%       0.000us       0.000us             1  
                                              aten::sub        10.09%       2.439ms        13.14%       3.177ms     317.717us      10.450ms        48.68%      10.450ms       1.045ms            10  
                                       cudaLaunchKernel         3.31%     799.561us         3.31%     799.561us      39.978us       0.000us         0.00%       0.000us       0.000us            20  
                                              aten::div         0.55%     133.436us         0.81%     195.034us      19.503us      11.019ms        51.32%      11.019ms       1.102ms            10  
                                             aten::view         0.21%      50.311us         0.21%      50.311us       2.516us       0.000us         0.00%       0.000us       0.000us            20  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      10.450ms        48.68%      10.450ms       1.045ms            10  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      11.019ms        51.32%      11.019ms       1.102ms            10  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 24.181ms
Self CUDA time total: 21.469ms
```

There are a few new operations in the trace:  
- `cudaLaunchKernel`: A CUDA runtime API function that launches a CUDA kernel on the GPU. This is the mechanism used to execute GPU code asynchronously.  
- `cudaDeviceSynchronize`: Forces the CPU to wait until all previously launched GPU operations are complete.  

The key observation here is that the `sum` and `div` operations take significantly less time on the GPU compared to the CPU. This is because the GPU can parallelize these operations across many cores, making it much faster than the CPU. This supports our initial hypothesis that the GPU should be faster for this kind of operation.  

But... is there anything else we can improve? Can we make this optimization even better?  

## The importance of data transfer

So this is where things get interesting. We have seen that using the correct data types is crucial for performance. Avoiding unnecessary conversions can save a lot of time. But have you considered the time it takes to move data between the CPU and GPU? Let’s start with a simple exploration:  

```python
def custom_pipeline(img, mean, std):
  img_cuda = img.to("cuda")
  return my_normalize(img_cuda, mean, std)

custom_profile(
    lambda: custom_pipeline(img_float, mean, std), 
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
)
```

```plaintext
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                               aten::to         1.02%       2.999ms        98.06%     288.017ms      28.802ms       0.000us         0.00%     267.229ms      26.723ms            10  
                                         aten::_to_copy         0.07%     213.702us        97.03%     285.018ms      28.502ms       0.000us         0.00%     267.229ms      26.723ms            10  
                                            aten::copy_         0.09%     271.131us        96.91%     284.653ms      28.465ms     267.229ms        92.59%     267.229ms      26.723ms            10  
                                        cudaMemcpyAsync        96.68%     283.985ms        96.68%     283.985ms      28.398ms       0.000us         0.00%       0.000us       0.000us            10  
                                              aten::div         0.09%     278.727us         0.97%       2.859ms     285.949us      10.986ms         3.81%      10.986ms       1.099ms            10  
                                             cudaMalloc         0.83%       2.424ms         0.83%       2.424ms       2.424ms       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaDeviceSynchronize         0.66%       1.952ms         0.66%       1.952ms       1.952ms       0.000us         0.00%       0.000us       0.000us             1  
                                              aten::sub         0.18%     514.698us         0.26%     771.397us      77.140us      10.402ms         3.60%      10.402ms       1.040ms            10  
                                       cudaLaunchKernel         0.14%     410.469us         0.14%     410.469us      20.523us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamSynchronize         0.14%     396.956us         0.14%     396.956us      39.696us       0.000us         0.00%       0.000us       0.000us            10  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 293.730ms
Self CUDA time total: 288.617ms
```

It’s interesting to see that `cudaMemcpyAsync` takes up to **96.68%** of the total CPU time! That’s a lot!  

Can we avoid this data transfer? Unfortunately no, the data must be on the GPU to be processed. But can we optimize it? Yes! We can do this by transferring less data.  

How? By using the correct data types! `uint8` is 1/4 the size of `float32` (1 byte vs. 4 bytes). So, we can move the data to the GPU as `uint8`, convert it to `float32`, and then normalize it. If memory bandwidth is the bottleneck, this should be faster!  


```python
def custom_pipeline_uint8(img, mean, std):
  img_cuda = img.to("cuda").float()
  return my_normalize(img_cuda, mean, std)

custom_profile(
    lambda: custom_pipeline_uint8(img_uint8, mean, std), 
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
)
```

```plaintext
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                               aten::to         1.09%     935.260us        95.88%      82.642ms       4.132ms       0.000us         0.00%      62.728ms       3.136ms            20  
                                         aten::_to_copy         0.31%     264.719us        94.80%      81.707ms       4.085ms       0.000us         0.00%      62.728ms       3.136ms            20  
                                            aten::copy_         0.42%     365.007us        94.21%      81.197ms       4.060ms      62.728ms        74.50%      62.728ms       3.136ms            20  
                                        cudaMemcpyAsync        93.10%      80.242ms        93.10%      80.242ms       8.024ms       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize         3.42%       2.949ms         3.42%       2.949ms       2.949ms       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaStreamSynchronize         0.45%     385.047us         0.45%     385.047us      38.505us       0.000us         0.00%       0.000us       0.000us            10  
                                       cudaLaunchKernel         0.43%     370.317us         0.43%     370.317us      12.344us       0.000us         0.00%       0.000us       0.000us            30  
                                              aten::sub         0.26%     221.024us         0.37%     317.485us      31.748us      10.457ms        12.42%      10.457ms       1.046ms            10  
                                    aten::empty_strided         0.28%     244.689us         0.28%     244.689us      12.234us       0.000us         0.00%       0.000us       0.000us            20  
                                              aten::div         0.17%     142.558us         0.24%     210.994us      21.099us      11.010ms        13.08%      11.010ms       1.101ms            10  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 86.192ms
Self CUDA time total: 84.195ms
```

And it is indeed faster! This is a great optimization! We often focus on how efficient the GPU is at computations (which is true), but we should pay more attention to the data being transferred between the CPU and GPU and look for ways to optimize that!  

Although torch profiler has been very useful here, let's compare things now using a classic timer to see how total time compares:

```python
import time

def timer(func):
    start = time.time()
    func()
    return time.time() - start

gpu_time_from_float = timer(lambda: custom_pipeline(img_float, mean, std))
gpu_time_from_uint = timer(lambda: custom_pipeline_uint8(img_uint8, mean, std))

print(f"Time transferring floats: {gpu_time_from_float:.4f}s")
print(f"Time transferring uint8: {gpu_time_from_uint:.4f}s")
print(f"Speedup: {gpu_time_from_float / gpu_time_from_uint:.2f}x")
```

```plaintext
Time transferring floats: 0.0306s
Time transferring uint8: 0.0083s
Speedup: 3.69x
```

This is something to always keep in mind when optimizing your data pipeline and I am pretty sure this can help many people out there to tune their inference pipelines!

I imagine the avid reader might be wondering: hasn’t PyTorch been designed to handle data loading and preprocessing on the GPU to avoid it sitting idle, waiting for data? That’s true. During training, if we can fully parallelize data preprocessing on the CPU and model inference on the GPU, we can achieve optimal performance. However, when serving a model, we typically only need to load a few images on demand and apply normalization. In such cases, we can leverage the optimizations discussed in this post.  

## Conclusion

In this post, we’ve explored the performance differences between CPU and GPU when normalizing images in PyTorch, and examined how data transfer and data types influence preprocessing efficiency. Key takeaways for optimizing your pipeline include:

1. Always use the correct data types to avoid unnecessary conversions.  
2. Minimize the amount of data transferred between the CPU and GPU.  
3. Optimize your data pipeline to take full advantage of the GPU’s parallel processing capabilities.  