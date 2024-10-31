---
layout: post
title: "Concurrent Requests in Python: httpx vs aiohttp"
subtitle: "How switching from httpx to aiohttp resolved my networking errors"
author: Miguel Mendez
description: "In high-concurrency networking situations, handling a large number of requests can lead to different behaviors between httpx and aiohttp. While httpx may fail under heavy load, switching to aiohttp offers a more reliable solution for managing high traffic in asynchronous Python applications. This post explores how httpx struggles with concurrency and how aiohttp outperforms it in such scenarios"
image: "/assets/images/fullsize/posts/2024-10-20-aiohttp-vs-httpx/thumbnail.jpg"
selected: y
mathjax: n
tags: [Python, aiohttp, httpx, concurrency, networking]
categories: [Programming, Python, Networking]
---

Over the past couple of weeks, I encountered a tricky bug while working on a computer vision application. After spending time troubleshooting, I decide to write this post since it might help other people to avoid the same issue. 

## The Setup

I have a computer vision REST API that receives images and returns predictions. The setup includes several replicas running behind a load balancer, and each server batches requests to maximize GPU usage and minimize latency—a fairly common setup for computer vision applications. Here’s a simple diagram showing the flow of requests from the client to the server:

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-10-20-aiohttp-vs-httpx/architecture.jpg --alt Application Diagram %}
</div>

I was using the `httpx` library from my Python client to send requests asynchronously to this service. Initially, things worked fine, but as the traffic increased, random errors started appearing. These errors were intermittent, making it even harder to trace the root cause and putting my patience to the limit. 

Note that these are the library versions I am using across this post:

```bash	
python                            3.11.3
aiohttp                           3.10.10
httpx                             0.27.2
```

## The Problem

I initially chose `httpx` for my client due to its modern async/await support and its recommendation in FastAPI’s documentation. It seemed like a solid choice for handling concurrent requests. However, as I ramped up the number of requests, I began encountering random crashes that were difficult to debug. The errors weren’t immediately obvious, and it took time to isolate the issue as being related to `httpx`. Here's a minimal example of the client-server setup I was working with.


### Server Code (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import asyncio
import uvicorn

app = FastAPI()

class ImagePayload(BaseModel):
    image: str = Field(..., description="Base64 encoded image")

@app.post("/process_image")
async def process_image(payload: ImagePayload):
    await asyncio.sleep(3)

    return {
        "message": "Image processed successfully after a delay"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
``` 

### Httpx Client Code

This is the client code that sends a lot of requests concurrently to the server. Note I have added a few variables here:

- `CONCURRENT_REQUESTS`: The number of concurrent requests that will be sent to the server
- `TOTAL_REQUESTS`: The total number of requests that will be sent to the server
- `TIMEOUT`: The timeout in seconds for each individual request

```python
import asyncio
import httpx
import base64
import numpy as np

SERVER_URL = "http://localhost:8000/process_image"

CONCURRENT_REQUESTS = 300
TOTAL_REQUESTS = 1000
TIMEOUT = 30

async def send_request(client, image_data):
    response = await client.post(SERVER_URL, json={"image": image_data})
    return response.json()

async def main():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_data = base64.b64encode(image).decode()
    
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONCURRENT_REQUESTS), verify=False, timeout=TIMEOUT) as client:
        tasks = [send_request(client, image_data) for _ in range(TOTAL_REQUESTS)]
        results = await asyncio.gather(*tasks)

    for i, result in enumerate(results, 1):
        print(f"Response {i}:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

When I run this code I see that a lot of requests succeed but suddenly the whole program crashes with the following error:

```bash
(the exception is too long to be shown here)
  File "/home/mmendez/pypoetry/virtualenvs/example//lib/python3.11/site-packages/httpx/_client.py", line 1776, in _send_single_request
    response = await transport.handle_async_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mmendez/pypoetry/virtualenvs/example//lib/python3.11/site-packages/httpx/_transports/default.py", line 376, in handle_async_request
    with map_httpcore_exceptions():
  File "/home/mmendez/.pyenv/versions/3.11.3/lib/python3.11/contextlib.py", line 155, in __exit__
    self.gen.throw(typ, value, traceback)
  File "/home/mmendez/pypoetry/virtualenvs/example//lib/python3.11/site-packages/httpx/_transports/default.py", line 89, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ReadError
```

This is quite frustrating because `httpx.ReadError` is a generic error indicating that something went wrong while reading the response from the server, but it doesn’t give much detail. What made this especially challenging is that on my application the error wasn’t consistent. It only occurred under heavy traffic, and even then, it didn’t happen every time.



## The Fix: Switch to aiohttp

After searching through the `httpx` GitHub repository, I found upon this [issue](https://github.com/encode/httpx/issues/3215){:target="_blank"}{:rel="noopener noreferrer"}, which provided insight into the performance limitations of `httpx` when compared to `aiohttp`. This is another popular library for handling asynchronous HTTP requests in Python. The solution became clear: switch to `aiohttp` and see if the problem persists.

Here’s how the client code looks after switching to aiohttp:

```python
import asyncio
import aiohttp
import base64
import numpy as np

SERVER_URL = "http://localhost:8000/process_image"
CONCURRENT_REQUESTS = 300
TOTAL_REQUESTS = 1000
TIMEOUT = 30

async def send_request(session, image_data):
    async with session.post(SERVER_URL, json={"image": image_data}) as response:
        return await response.json()

async def main():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_data = base64.b64encode(image.tobytes()).decode()
    
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [send_request(session, image_data) for _ in range(TOTAL_REQUESTS)]
        
        results = await asyncio.gather(*tasks)

    for i, result in enumerate(results, 1):
        print(f"Response {i}:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

This simple change resolved all the random errors and even improved the overall performance.

## Conclusion

If you're building an application that needs to handle a large number of concurrent requests, especially in a production environment, I recommend switching to `aiohttp`. While `httpx` is a great library, it may not be the best choice for high-concurrency use cases just yet. Hopefully, these issues will be resolved soon, but until then, `aiohttp` has proven to be a more reliable choice.