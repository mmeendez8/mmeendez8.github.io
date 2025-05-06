---
layout: post
title: "Deepstream Tutorial"
subtitle: ""
author: Miguel Mendez
description: ""
image: "/assets/images/fullsize/posts/2025-02-19-configure-multiple-ssh-keys-git/thumbnail.jpg"
selected: y
mathjax: y
tags: []  
categories: []  
---

Deepstream is a powerful video streaming and processing library that allows you to build computer vision applications that run at full speed. It is really great but it is also very hard to start with it. I am creating this blog post with the aim of helping other people to get started with Deepstream, this is the classic "I would like to have had this when I started" post. Let's get started with some basics.

## Gstreamer basics

[NVIDIA’s DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) is a complete streaming analytics toolkit based on GStreamer for AI-based multi-sensor processing, video, audio, and image understanding. Ok, nice, so we probably need to understand GStreamer first. Let's check the definition of GStreamer in the [GStreamer website](https://gstreamer.freedesktop.org/):
> GStreamer is a library for constructing graphs of media-handling components

That's a very good definition in my opinion. It is all about graphs, you have a source, you have a sink, and you have some processing in between. You just need to add components to your graph, link them, and run it. These components are generally called elements. I see

- **Source**: This is the element that generates the data. It can be a file, a camera, or any other source of data.
- **Sink**: This is the element that consumes the data. It can be a file, a display, or any other sink of data.
- **Filter**: This is the element that processes the data. It can be a decoder, an encoder, or any other filter of data.

A simple pipeline that combines all these elements would look like this:

<div class="post-center-image" style="max-width: 600px; margin: 0 auto;">
    {% picture pimage /assets/images/fullsize/posts/2025-04-22-deepstream-tutorial/image.png --alt Example gstreamer pipeline %}
</div>

{:refdef: class="image-caption"}
*Figure 1. Example Gstreamer pipeline extracted from [Gstreamer docs](https://gstreamer.freedesktop.org/documentation/tutorials/basic/concepts.html?gi-language=c)*
{: refdef}

The ports through which GStreamer elements communicate are called pads (GstPad). Source pads output data, sink pads receive data. Source elements have only source pads, sink elements have only sink pads, and filter elements have both.