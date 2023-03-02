---
layout: post
title:  "Torchserve in Computer Vision: REST vs gRPC"
subtitle: "Benchmarking protocols performance for sending images"
description: "This post compares the performance of gRPC and REST communication protocols for serving a computer vision deep learning model using TorchServe. I tested both protocols and looked at the pros and cons of each. The goal is to help practitioners make informed decisions when choosing the right communication protocol for their use case."
image: "/assets/images/fullsize/posts/2021-12-09-cnn-vs-transformers/thumbnail.jpg"
selected: y
mathjax: y
---

For the past few weeks at my current company we have been considering the benefits of using gRPC to improve the performance of some of our services. However, despite my efforts to research the topic, I have not been able to find relevant information that is suitable for our current use case, sending images to a model server and receiving a response in the most efficient way possible. It's easy to find benchmarks that show how switching from REST to gRPC using structured data results in huge performance improvements. However, it was quite difficult to find a similar benchmark using images... And that is the main reason behind these post!

## Some thoughts on gRPC

When you start reading about gRPC, you soon realize that it involves two main things that can really help you to speed up your system communications. 

### HTTP2

gRPC is build on HTTP/2 protocol which was designed to solve some of the latency issues of HTTP/1.1. We need to highlight two features of the newest protocol that will make an impact on our benchmarking:

- **Multiplexed streams**: where multiple requests and responses can be sent over a single connection. Note that HTTP/1.1 can reuse the connections using pooling. Multiplexed streams would only be important in a scenario with a large number of servers that would force HTTP/1.1 to open and maintain a large number of connections.
- Use of **binary protocol** instead of being text-based, which reduces message size and improves parsing.

### Protobuf

Protocol Buffers, also known as protobuf, is a language-agnostic binary serialization format developed by Google. It is used for efficient data **serialization of structured data** and communication between applications. It is faster than JSON for two reasons: messages are shorter and serialization (convert messages to and from bytes) is faster.

In [this post](https://nilsmagnus.github.io/post/proto-json-sizes/) you can see a good comparison of Protobuf vs JSON sizes for structured data. TLDR: Protobuf is always smaller than gzipped json but seems to lose its clear advantage when mesage sizes are large.

### How does this apply to images?

Images are a special case of data. Structured data is text that has been predefined and formatted to a set structure. Protobuf can take advantage of the schema definitions of the data to speed up serialization and compression size. 

Things are different with images. Basically if you want to convert an image to bytes in an efficient manner and without losing information you have to use specific handcrafted methods that have been carefully designed for this, such as JPEG, PNG... In other words, Protobuf is not going to help you here since compression and serialization will depend on your image library. See this example:

```python
# create random 100x100 rgb image
image = numpy.random.rand(100, 100, 3) * 255
# serialize image to jpg using opencv
encoded_image = cv2.imencode(".jpg", image)[1].tobytes()
# fake send with grpc
grpc.send(encoded_image)
```

The point here is that Protobuf is not really helping. Given that it is one of key points of gRPC, differences between REST and gRPC cannot be that high here... Let's check this with real numbers :)

## 1. Base benchmark 

First thing we wanted to do is to check if we were able to reproduce those benchmarks we found on the web. The idea is simple, create two equivalent REST and gRPC servers and measure the time they take to process and respond to different requests.
The gRPC server has been implemented using [python grpc library](https://grpc.io/docs/languages/python/basics/) and we have used [FastAPI](https://fastapi.tiangolo.com/) for the REST one. 

We decided to measure three different requests and using a single response for all of them. The gRPC `.proto` file for those requests looks like the following:

```python
class BasicRequest(BaseModel):
    """
    Structured data request, we expect to match online benchmarks with this
    """
    field1: str
    field2: str
    field3: int
    field4: Dict[str, int]

class ImageBase64Request(BaseModel):
    """
    Encode image as a string using Base64 encoding. 
    This is simple and very bad solution (but simple to do) that should be always avoided
    """
    image: str

class ImageBinaryRequest(BaseModel):
    """
    Contains an image encoded as bytes.
    """
    image: bytes

class BasicResponse(BaseModel):
    prediction1: List[float]
    prediction2: Dict[str, int]
```

Note REST's requests and responses are identical to these so we can make a fair comparison.

Our client does a very simple thing, it sends concurrent requests to each server and waits for a response. It then computes the average time it took. Pseudocode for the client it is shown below:

```python
times = []
for _ in range(10):
	start= time()
	send_concurrent_request_to_specific_server(n=20)
	times.append(time() - start())

average_time = mean(times)
```

We tested this for three different image sizes. Results are collected below:

<div class="table-wrapper" markdown="block">

|      | Basic (0.001 MB) | B64 (0.306 MB) | Binary (0.229 MB)  |
|------|------------------|----------------|--------------------|
| REST | 0.0723           | 0.0943         | 0.0572             |
| gRPC | 0.0093 (x7.7)    | 0.0179 (x5.2)  | 0.0120 (x4.7)      |

<p>Table 1. Results for small images: 360x640</p>

</div>

<div class="table-wrapper" markdown="block">

|      | Basic (0.001 MB) | B64 (1.160 MB) | Binary (0.870 MB)  |
|------|------------------|----------------|--------------------|
| REST | 0.0611           | 0.2350         | 0.0872             |
| gRPC | 0.0090 (x6.7)    | 0.0926 (x2.5)  | 0.0570 (x1.5)      |

<p>Table 2. Results for medium images: 720x1280</p>

</div>


<div class="table-wrapper" markdown="block">

|      | Basic (0.001 MB) | B64 (3.094 MB) | Binary (2.320 MB)  |
|------|------------------|----------------|--------------------|
| REST | 0.0583           | 0.8056         | 0.1909 (x1.03)     |
| gRPC | 0.0097 (x6)      | 0.2793 (x2.9)  | 0.1974             |

<p>Table 3. Results for large images: 1080x1920</p>

</div>


We can extract some conclussion from previous tables:

1. gRPC achieves around a x6 improvement with respect to REST for structured data (Basic column). The results [match online benchmarks](https://medium.com/@EmperorRXF/evaluating-performance-of-rest-vs-grpc-1b8bdf0b22da#:~:text=gRPC%20is%20roughly%207%20times,of%20HTTP%2F2%20by%20gRPC.) and we know this is because we are taking advantage of Protobuf serialization and HTTP2 protocol.
2. For Base64 and Binary we observe a relation between image size and gRPC performance. As the image size increase, the difference between REST and gRPC are smaller.
3. In the Base64 case, gRPC helps to serialize faster and in a more optimal way the string. We know from [this post](https://nilsmagnus.github.io/post/proto-json-sizes/) that Protobuf loses its advantage when message size increases.
4. Binary is a special case as we know we are not getting any advantage from using Protobuf for our serialization and message size (this is determined by the image format we chose). On the contrary it is harming our performance. There is still  some encoding going on in Protobuf, since it needs to format our chunk of image bytes inside the Protobuf message format. This little thing might be making REST as good as gRPC for large images!


## Torchserve benchmark

I have been using [TorchServe](https://pytorch.org/serve/) for a while now and I am quite happy with it. It provides all the flexibility I need and it is quite simple to set up. Model handlers allow you to customize every detail for your specific model without really worrying about other complex things as batching and queing requests. I do not pretend to give an overview of TorchServe or make a comparison of its advantages compared to other inference servers but you can get some of that information [here](https://hamel.dev/notes/serving/#inference-servers).
