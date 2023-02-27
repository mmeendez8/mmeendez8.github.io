---
layout: post
title:  "Torchserve in Computer Vision: REST vs gRPC"
subtitle: "Benchmarking protocols performance for sending images"
description: "This post compares the performance of gRPC and REST communication protocols for serving a computer vision deep learning model using TorchServe. I tested both protocols and looked at the pros and cons of each. The goal is to help practitioners make informed decisions when choosing the right communication protocol for their use case."
image: "/assets/images/fullsize/posts/2021-12-09-cnn-vs-transformers/thumbnail.jpg"
selected: y
mathjax: y
---

I have been using [TorchServe](https://pytorch.org/serve/) for a while now and I am quite happy with it. It provides all the flexibility I need and it is quite simple to set up. Model handlers allow you to customize every detail for your specific model without really worrying about other complex things as batching and queing requests. I do not pretend to give an overview of TorchServe or make a comparison of its advantages compared to other inference servers but you can get some of that information [here](https://hamel.dev/notes/serving/#inference-servers).

During last weeks at my current company we have been considering the advantages of using gRPC for improving performance. However, despite my efforts to research the topic, I have been unable to find relevant information that is suitable for our current use case, sending images to a model server and receiving a response in the most efficient manner possible. It is easy to find benchmarks which show how switching from REST to gRPC using structured data results in huge performance improvements. Nevertheless it was pretty hard to find some similar benchmarking using images... So best thing we could do is to test this!

The benchmark is organised as follows:

1. Replicate benchmarks using simple grpc and rest servers that received structured data and images.
2. See if these results also translate to Torchserve

## Some thoughts on gRPC

When you start reading about gRPC, you soon realize that there are two things that can really help you to speed up your system communications. 

### HTTP2

HTTP/2 protocol was designed to solve some of the latency issues of HTTP/1.1. We need to highlight two features of the newest protocol:

- **Multiplexed streams**: where multiple requests and responses can be sent over a single connection. HTTP/1.1 only allows one request and response at a time which is inefficient and increases latency.
- Use of **binary protocol** instead of being text-based, which reduce message size and improves parsing.

### Protobuf

Protocol Buffers, also known as protobuf, is a language-agnostic binary serialization format developed by Google. It is used for efficient data **serialization of structured data** and communication between applications. It is faster than JSON for two reasons: messages are shorter and serialization is faster.

In [this post](https://nilsmagnus.github.io/post/proto-json-sizes/) you can see a good comparison of protobuf vs json sizes for structured data. TLDR: Protobuf is always smaller than gzipped json but seems to lose its clear advantage when mesage sizes are large.

### How does this apply to images?

Images are a special case of data. Structured data is formatted text that has been predefined and formatted to a set structure. This allows to create efficient as Protobuf that take advantage of the schema definitions of the data to speed up serialization and compression size. This is way harder with images. Basically if you want to convert an image to bytes in an efficient manner and without losing information you have to use specific handcrafted methods that have been carefully designed for this, such as JPEG, PNG ... This means that the size of the message and the serialization time will be identical for gRPC and REST! Let's try to find out if these translates to numbers in our benchmarking!

## 1. Base benchmark 

First thing we wanted to do is to check if we were able to reproduce those benchmarks we found on the web. The idea is simple, create two equivalent REST and GRPC servers and measure the time they take to process and response to different requests.
The grpc server has been implemented using python grpc library and we have used FastAPI for the REST one. All code for this experiments (server code and client code) is located inside the `base_benchmark` folder.

We decided to measure three different request. One very basic that contains only text data, another one that contains images in base64 string format and finally a request that contains a image directly codified in bytes. The response is going to be the same for all the requests. The gRPC `.proto` file for those requests looks like the following:

```python
class BasicRequest(BaseModel):
    field1: str
    field2: str
    field3: int
    field4: Dict[str, int]

class ImageBase64Request(BaseModel):
    image: str

class ImageBinaryRequest(BaseModel):
    image: bytes

class BasicResponse(BaseModel):
    prediction1: List[float]
    prediction2: Dict[str, int]
```

Our client does a very simple thing, it sends concurrent requests to each server and waits for a response. It then computes the average time it took. Pseudocode for the client it is shown below:

```python
times = []
for _ in range(10):
	start= time()
	send_concurrent_request_to_specific_server(n=20)
	times.append(time() - start())

average_time = mean(times)
```

| Image               | Description                                                               |
|---------------------|---------------------------------------------------------------------------|
| prod/torch1.11-cpu  | Running torch based projects in CPU. Poetry is installed and ready to use |
| prod/torch1.11-cu11 | Running torch based projects in GPU. Poetry is installed and ready to use |
| serve/torchserve    | Base torchserve image with poetry preinstalled.                           |
| serve/fastapi       | Base image used for fastapi projects with poetry preinstalled.            |