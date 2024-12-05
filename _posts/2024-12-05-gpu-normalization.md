---
layout: post
title: "Image Normalization: CPU vs GPU Performance in PyTorch"
subtitle: "Understanding the performance differences in Pytorch's image normalization"
author: Miguel Mendez
description: "A profiling comparison between CPU and GPU performance when normalizing images in PyTorch. There are several factors to consider when optimizing preprocessing pipelines, such as data types, data transfer, and parallel processing capabilities. This posts explores these factors and provides insights on how to optimize your data pipeline."
image: "/assets/images/fullsize/posts/2024-12-05-gpu-normalization/thumbnail.jpg"
selected: y
mathjax: n
tags: [PyTorch, GPU, Performance, Computer Vision, Python]
categories: [Deep Learning, Performance]
---

This is a post that has been on my to-do list for a long time, and I’m happy to have finally found the time to write it. I hope you find it useful because it’s about a topic I find quite interesting. It all started while I was testing some of our internal data preprocessing pipelines and began thinking about the performance of the normalization step. Image normalization is a straightforward process: subtract the mean and divide by the standard deviation. At first glance, it seems like something where a GPU should be able to help significantly—probably what you’re thinking too! But is that really the case? Let’s find out!

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

TODO: ADD PROFILER OUTPUT

- The basic operations (`aten::sub` and `aten::div`) account for most of the execution time.
- A significant portion of time is also spent on type conversion operations (`aten::to`, `aten::_to_copy`).
- These conversions occur because the input is `uint8`, while the mean and standard deviation are `floats`.

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
     aten::sub        51.70%     788.476ms        51.70%     788.476ms      78.848ms            10  
     aten::div        48.28%     736.341ms        48.28%     736.341ms      73.634ms            10  
    aten::view         0.01%     216.701us         0.01%     216.701us      10.835us            20  
--------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.525s
```

his is much faster! Here’s our first takeaway: always use the correct data types to avoid unnecessary conversions. Casting between types is costly and can significantly impact performance.

Now, what happens if we compare our simple normalization with the official PyTorch implementation?

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

We can spot a few extra operations because PyTorch checks that the types and dimensions are correct (along with other performance optimizations, as seen in the [source code](https://github.com/pytorch/vision/blob/6279faa88a3fe7de49bf58284d31e3941b768522/torchvision/transforms/v2/functional/_misc.py#L19){:target="_blank"}{:rel="noopener noreferrer"}). However, the main operations remain the same, and the performance is very similar. This means we’ve successfully replicated the behavior. Awesome! Now it’s GPU time—let’s see how it performs.

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
                                       cudaLaunchKernel        67.12%      84.013ms        67.12%      84.013ms       4.201ms       0.000us         0.00%       0.000us       0.000us            20  
                                              aten::sub        14.61%      18.287ms        52.47%      65.686ms       6.569ms      10.446ms        48.73%      10.446ms       1.045ms            10  
                                              aten::div         2.29%       2.872ms        32.15%      40.248ms       4.025ms      10.989ms        51.27%      10.989ms       1.099ms            10  
                                  cudaDeviceSynchronize        15.30%      19.147ms        15.30%      19.147ms      19.147ms       0.000us         0.00%       0.000us       0.000us             1  
                                             cudaMalloc         0.60%     755.287us         0.60%     755.287us     377.644us       0.000us         0.00%       0.000us       0.000us             2  
                                             aten::view         0.08%      94.181us         0.08%      94.181us       4.709us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.01%       6.915us         0.01%       6.915us       3.457us       0.000us         0.00%       0.000us       0.000us             2  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      10.446ms        48.73%      10.446ms       1.045ms            10  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      10.989ms        51.27%      10.989ms       1.099ms            10  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 125.175ms
Self CUDA time total: 21.435ms
```

There are a few new operations in the trace:  
- `cudaLaunchKernel`: A CUDA runtime API function that launches a CUDA kernel on the GPU. This is the mechanism used to execute GPU code asynchronously.  
- `cudaDeviceSynchronize`: Forces the CPU to wait until all previously launched GPU operations are complete.  

The key observation here is that the `sum` and `div` operations take significantly less time on the GPU compared to the CPU. This is because the GPU can parallelize these operations across many cores, making it much faster than the CPU. This supports our initial hypothesis that the GPU should be faster for this kind of operation.  

So, is there anything else we can improve? Can we make this optimization even better?  

## The importance of data types

We have seen that using the correct data types is crucial for performance. Avoiding unnecessary conversions can save a lot of time. But have you considered the time it takes to move data between the CPU and GPU? Let’s start with a simple exploration:  

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

It is particularly interesting to see that `cudaMemcpyAsync` accounts for up to `96.68%` of the total CPU time! That is a **LOT** of time!  

This leads us to wonder: can we avoid this data transfer? The answer is no. The data needs to be on the GPU to be processed there. So the next question is, can we optimize this data transfer? The answer is yes! We can optimize it by... transferring less data!  

How? By using the correct data types! `uint8` is 1/4 the size of `float32` (1 byte vs. 4 bytes). So, we can try moving the data to the GPU as `uint8`, converting it to `float32`, and then normalizing it. If memory bandwidth is the bottleneck, this approach should be faster!  


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
                                               aten::to         5.65%       7.563ms        97.20%     130.189ms       6.509ms       0.000us         0.00%      70.131ms       3.507ms            20  
                                         aten::_to_copy         0.26%     349.837us        91.55%     122.626ms       6.131ms       0.000us         0.00%      70.131ms       3.507ms            20  
                                            aten::copy_         0.50%     673.579us        91.01%     121.906ms       6.095ms      70.131ms        76.59%      70.131ms       3.507ms            20  
                                        cudaMemcpyAsync        64.48%      86.361ms        64.48%      86.361ms       8.636ms       0.000us         0.00%       0.000us       0.000us            10  
                                       cudaLaunchKernel        25.99%      34.806ms        25.99%      34.806ms       1.160ms       0.000us         0.00%       0.000us       0.000us            30  
                                  cudaDeviceSynchronize         2.14%       2.869ms         2.14%       2.869ms       2.869ms       0.000us         0.00%       0.000us       0.000us             1  
                                              aten::sub         0.23%     314.127us         0.36%     475.542us      47.554us      10.446ms        11.41%      10.446ms       1.045ms            10  
                                    aten::empty_strided         0.28%     370.266us         0.28%     370.266us      18.513us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamSynchronize         0.25%     338.483us         0.25%     338.483us      33.848us       0.000us         0.00%       0.000us       0.000us            10  
                                              aten::div         0.16%     213.005us         0.24%     324.173us      32.417us      10.992ms        12.00%      10.992ms       1.099ms            10  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 133.943ms
Self CUDA time total: 91.569ms
```

Eureka! This is a great optimization! We often focus on how efficient the GPU is at computations (which is true), but we should pay more attention to the data being transferred between the CPU and GPU and look for ways to optimize that!  

I imagine the avid reader might be wondering: hasn’t PyTorch been designed to handle data loading and preprocessing on the GPU to avoid it sitting idle, waiting for data? That’s true. During training, if we can fully parallelize data preprocessing on the CPU and model inference on the GPU, we can achieve optimal performance. However, when serving a model, we typically only need to load a few images on demand and apply normalization. In such cases, we can leverage the optimizations discussed in this post.  

## Conclusion

In this post, we’ve explored the differences between CPU and GPU performance when normalizing images in PyTorch. We’ve learned that several factors are key to optimizing preprocessing pipelines:  

1. Always use the correct data types to avoid unnecessary conversions.  
2. Minimize the amount of data transferred between the CPU and GPU.  
3. Optimize your data pipeline to take full advantage of the GPU’s parallel processing capabilities.  