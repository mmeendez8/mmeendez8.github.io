---
layout: post
title:  "Image Transmission for Computer Vision: A Comparison of Torchserve's REST and gRPC"
subtitle: "Benchmarking protocols performance for sending images"
author: Miguel Mendez
description: " This post compares the performance of Torchserve's REST and gRPC communication protocols for transmitting images to a computer vision deep learning model. We conducted benchmarks for structured data, Base64 encoded images, and binary image transmission. The goal is to help practitioners make informed decisions when choosing the right communication protocol for their specific use case, taking into account factors such as ease of implementation and familiarity with the technology."
image: "/assets/images/fullsize/posts/2023-02-24-torchserve-grpc/thumbnail.jpg"
selected: y
mathjax: y
---

Special thanks to [Javier Guzman](https://www.linkedin.com/in/jguzmanfd/){:target="_blank"}{:rel="noopener noreferrer"} for working with me in completing the benchmarking discussed in this post.

In the past few weeks, we have been exploring the potential advantages of adopting gRPC to enhance the performance of our services. Although I have conducted extensive research on this topic, I have not been able to find relevant information that specifically addresses our use case, which involves transmitting images to a model server and receiving a response in the most efficient manner. While there are numerous benchmarks that demonstrate significant performance improvements when migrating from REST to gRPC using structured data, it has been challenging to locate a similar benchmark for image transmission... And that is the main reason behind this post!  

All the code for the different benchmarks can be found in [this Github repository](https://github.com/mmeendez8/grpc_vs_rest){:target="_blank"}{:rel="noopener noreferrer"}. You can find instructions in the README file. It's important to note that our primary objective was to conduct this testing on our cloud infrastructure, where both the servers and clients were deployed on the same Kubernetes cluster. This allowed us to replicate a real-world scenario as closely as possible.

## Some thoughts on gRPC

When you start reading about gRPC, you soon realize that it involves two main features that can really help you to speed up your system communications. 

### HTTP2

gRPC is built on the HTTP/2 protocol, which was specifically designed to address the latency issues of its predecessor, HTTP/1.1. There are two key features of HTTP/2 that are particularly relevant to our benchmarking efforts:

- **Multiplexed streams**: With HTTP/2, multiple requests and responses can be transmitted over a single connection. While HTTP/1.1 can also reuse connections through pooling, the ability to multiplex streams becomes more important as the number of servers increases. When multiple HTTP requests are performed in a very short span of time, HTTP/1.1 has no way to share those connections. Therefore, it will create new connections to the content server for each HTTP request (see [here](https://blog.codavel.com/http2-multiplexing){:target="_blank"}{:rel="noopener noreferrer"} for a extended explanation)

- **Binary protocol**: Unlike HTTP/1.1, which is text-based, HTTP/2 uses a binary protocol which facilitates more efficient parsing. This can have a significant impact on performance, particularly when dealing with large datasets such as images.

### Protobuf

Protocol Buffers, also known as Protobuf, is a language-agnostic binary serialization format developed by Google. It is used for efficient data **serialization of structured data** and communication between applications. It is faster than JSON for two reasons: 

- **Messages are shorter**. In Protobuf messages do not contain any metadata or extra information such as field names and data types. This is not needed since the data schema has been strictly predefined in the `.proto` file. It also uses a compact binary representation, variable-length encoding, which means that the number of bytes required to represent a value depends on its size.
- **Serialization is faster**. Converting messages to and from bytes is faster than in JSON because of its binary format and predefined schema. Decoding can be optimized and parallelized.

In [this post](https://nilsmagnus.github.io/post/proto-json-sizes/){:target="_blank"}{:rel="noopener noreferrer"} you can see a good comparison of Protobuf vs JSON sizes for structured data. TLDR: Protobuf is always smaller than gzipped json but seems to lose its clear advantage when mesage sizes are large.

### How does this apply to images?

Structured data is text that has been predefined and formatted to a set structure. Protobuf can take advantage of the schema definitions of the data to speed up serialization and compression size. However, images do not fall under the category of structured text. Basically if you want to convert an image to bytes in an efficient manner and without losing information you have to use specific handcrafted methods that have been carefully designed for this, such as JPEG, PNG... In other words, Protobuf is not going to help you here since compression and serialization will depend on your image library. See this example:

```python
# create random 100x100 rgb image
image = numpy.random.rand(100, 100, 3) * 255
# serialize image to jpg using opencv
encoded_image = cv2.imencode(".jpg", image)[1].tobytes()
# fake send with grpc
grpc.send(encoded_image)
```

The key feature here is that Protobuf is not really helping. Given that it is one of key points of gRPC, differences between REST and gRPC cannot be that high here... Let's check this with real numbers:

## 1. Base benchmark 

First thing we wanted to do is check if we were able to reproduce those benchmarks we found on the web. The idea is simple, create two equivalent REST and gRPC servers and measure the time they take to process and respond to different requests.
The gRPC server has been implemented using [python grpc library](https://grpc.io/docs/languages/python/basics/){:target="_blank"}{:rel="noopener noreferrer"} and we have used [FastAPI](https://fastapi.tiangolo.com/){:target="_blank"}{:rel="noopener noreferrer"} for the REST one. 

<div class="post-center-image">
{% picture pimage /assets/images/fullsize/posts/2023-02-24-torchserve-grpc/cat_bytes.png --alt Cat being compressed to bytes  %}
</div>

{:refdef: class="image-caption"}
*This is what Stable Diffusion creates with the prompt "an image of a cat is being encoded into a chunk of bytes"*
{: refdef}

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
    This is a very bad solution (but simple to do) that should always be avoided
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

Our client does a very simple thing, sends twenty concurrent requests to each server and waits for a response. It repeats this ten times for then computing the average time it took. Pseudocode for the client it is shown below:

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

1. gRPC achieves around a x6 improvement with respect to REST for structured data (Basic column). The results [match online benchmarks](https://medium.com/@EmperorRXF/evaluating-performance-of-rest-vs-grpc-1b8bdf0b22da#:~:text=gRPC%20is%20roughly%207%20times,of%20HTTP%2F2%20by%20gRPC.){:target="_blank"}{:rel="noopener noreferrer"} and we know this is because we are taking advantage of Protobuf serialization and HTTP2 protocol.
2. For Base64 and Binary we observe a relation between image size and gRPC performance. As the image size increase, the difference between REST and gRPC are smaller.
3. In the Base64 case, gRPC helps to serialize faster and in a more optimal way the string. We know from [this post](https://nilsmagnus.github.io/post/proto-json-sizes/){:target="_blank"}{:rel="noopener noreferrer"} that Protobuf loses its advantage when message size increases.
4. Binary is a special case as we know we are not getting any advantage from using Protobuf for our serialization and message size (this is determined by the image format we chose). On the contrary it is harming our performance. There is still  some encoding going on in Protobuf, since it needs to format our chunk of image bytes inside the Protobuf message format. This little thing might be making REST as good as gRPC for large images!


## Torchserve benchmark

I have been using [TorchServe](https://pytorch.org/serve/){:target="_blank"}{:rel="noopener noreferrer"} for a while now and I am quite happy with it. It provides all the flexibility I need and it is quite simple to set up. Model handlers allow you to customize every detail for your specific model without really worrying about other complex things such as batching and queing requests. 
I do not intend to give an overview of TorchServe or make a comparison of its advantages compared to other inference servers, I will leave that for a plausible future post.

The documentation for Torchserve's [gRPC API](https://pytorch.org/serve/grpc_api.html){:target="_blank"}{:rel="noopener noreferrer"} could be improved, as it currently requires users to download the official repository to generate a Python gRPC client stub from the proto files. However, I have attached these files to the repository, so you can easily run the benchmark without having to worry about this step.

The experiment is very similar to the previous one, sending 20 concurrent request and repeating that 10 times to measure the average time. I am going to use one of the pytorch vision model examples, [densenet161](https://pytorch.org/hub/pytorch_vision_densenet/){:target="_blank"}{:rel="noopener noreferrer"}. The model is not important here since we do not really care about inference results. Let's see some results:


<div class="table-wrapper" markdown="block">

|      | B64 (0.306 MB) | Binary (0.229) |
|------|----------------|----------------|
| REST | 0.884          | 0.628          |
| gRPC |       X        | 0.645          |

<p>Table 4. Results for small images: 360x640</p>

</div>

<div class="table-wrapper" markdown="block">

|      | B64 (0.306 MB) | Binary (0.229) |
|------|----------------|----------------|
| REST | 1.262          | 0.946          |
| gRPC |        X       | 0.927          |

<p>Table 5. Results for medium images: 720x1280</p>

</div>

<div class="table-wrapper" markdown="block">

|      | B64 (0.306 MB) | Binary (0.229) |
|------|----------------|----------------|
| REST | 2.188          | 1.384          |
| gRPC |         X      | 1.422          |

<p>Table 6. Results for large images: 1080x1920</p>

</div>

Note there are not results for B64 gRPC since this is not allowed by Torchserve schema definition. 

Translating the insights gained from benchmarking with the base servers can be challenging. The tables indicate that Base64 encoding should be avoided and that there are no significant performance differences between using gRPC and REST.

Two factors contribute to the similar performance results for gRPC and REST. Firstly, the model's inference time is considerably longer than the networking time, making it difficult to discern the small gains obtained by changing the transmission protocol. For example, sending 20 large images concurrently in the simple base case (Table 3) took roughly 0.19s, whereas we are now spending approximately 1.4 seconds (Table 6), highlighting the significant impact of model inference time on the comparison.

Secondly, the Torchserve implementation plays a role in these results. It has been observed that Torchserve's `.proto` definition for [prediction response](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/inference.proto#L20-L23){:target="_blank"}{:rel="noopener noreferrer"} is too generic and it cannot be personalized with your model specifics.

```json
message PredictionResponse {
    // Response content for prediction
    bytes prediction = 1;
}
``` 

This means that your response will be converted to a chunk of bytes so you would not be getting any advantage from Protobuf serialization (similar to what happens with images). For example if our model returns three lists of bounding boxes, class and scores, the `.proto` file for our response could be something like:

```json
message PredictionResponse {
    repeated float scores = 1;
    repeated int32 scores = 2;
    repeated repeated int32 bboxes = 3;
}
``` 

The differences between this response and the one provided by Torchserve are clear. You do not get any of the Protobuf advantage since the Torchserve schema definition is too general. A better or more customizable definition such as the one provided by Tfserving, of the `.proto` file could help boost performance. 

## Conclusion

If you're using Torchserve to serve computer vision models, it's recommended to steer clear of gRPC. Our findings show that there are no performance benefits to using gRPC. Moreover, it adds code complexity while hindering debugging due to its non-human-readable messages. Since REST is more commonly used, most developers are already familiar with it. Switching to gRPC in this scenario comes with a learning curve that doesn't offer any significant advantages.


*Any ideas for future posts or is there something you would like to comment? Please feel free to reach out via [Twitter](https://twitter.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"} or [Github](https://github.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"}*


