---
layout: post
title: "Video Segmentation Post"
subtitle: ""
description: ""
image: "/assets/images/fullsize/posts/2024-02-07-nfl-field-mapping/thumbnail.jpg"
selected: y
mathjax: n
---

I found myself visiting the most recent literature on Video Segmentation and I found surpised about the publication rate in this field. If you are related with this topic you probably are familiar with terms such as SVOS, UVOS, VIS, Zero-Shot, One-Shot, etc. If you are not then you will probably find yourself as lost as I was when I started reading about this topic.

My intention in here is to focus on a single topic which was the one I was interested in. This is Video instance Segmentation (VIS) which extends the image instance segmentation task from the image domain to the video domain. Goal is to segment object instances from a predefined category set in videos and associate the instance identities across frames. It can be seen as a combination of instance segmentation and object tracking in videos.

I will be updating this post with the most recent papers and code implementations I find interesting. 


## Video Segmentation

Video segmentation is a topic that condenses a lot of different tasks which can have multiple names. I personaly like the taxonomy used in the [Youtube-VOS Dataset](https://youtube-vos.org/dataset/) which is one of the main benchmarks in this field so I will stick with it through this post. The different tasks are:

- **Video Object Segmentation (VOS)**: Semi-supervised video object segmentation targets at segmenting a particular object instance throughout the entire video sequence given only the object mask of the first frame.
- **Video Instance Segmentation (VOS)**: Video Instance Segmentation (VIS) extends image instance segmentation to videos, aiming to segment and track object instances across frames.
- **Referring Video Object Segmentation (RVOS)**: Referring Video Object Segmentation (RVOS) is a task that requires to segment a particular object instance in a video given a natural language expression that refers to the object instance.

[maybe add figure explaining the difference between them]

## Dataset section? 
hello

## Key concepts

### Input sequence length

The input sequence length or clip(aka the number of frames a model can process in parallel) is key in video instance segmentation. This is a very similar issue to the one faced by LLMs. Longer sequences offer more temporal context, helping in accurately segmenting objects across frames, especially in scenarios of occlusion and appearance variation.  However, they also demand higher computational resources and extend training and inference times. Think that transformers have a quadratic complexity with respect to the sequence length. Thus, balancing sequence length with computational feasibility and model capability is essential for optimizing performance and efficiency in video instance segmentation tasks.

### Offline vs Onlne

Many video instance segmentation approaches are categorized as offline, where the entire video is analyzed as a single input. This method is particularly feasible for datasets composed of short clips, with the primary requirement being the model's capacity to handle the longest clip in the dataset. On the other hand, online methods handle video frames as they arrive, requiring immediate predictions without seeing future frames. This can complicate maintaining consistency over time. They typically use shorter input sequences to balance accuracy with the demands of instant processing.


### Stride

The input sequence length defines the number of frames processed in parallel. The stride, on the other hand, determines the temporal distance between frames in the input sequence. A stride of 1 means that the input sequence is a continuous sequence of frames, while a stride of 2 means that every other frame is skipped. By increasing the stride - that is, skipping more frames - the system can work faster because it's looking at fewer frames.

## Video Instance Segmentation

From now on I will exclusively focus on Video Instance Segmentation (VIS) since it is the topic I was interested in. Most of papers before 2020 were based on either:

- **Top-down approach**: following tracking-by-detection methods (you can check [this other post]({% post_url 2023-11-08-tracking-by-detection-overview %}) for more information on this topic)
- **Bottom-up approach**: clustering embedding of pixels into objects

This methods suffered of different issues and around 2020 transformer-based methods started to appear and most of research focused on how to throw a transformer into this problem that could equal the state-of-the-art. 


### VisTR (2021)

VisTR stands simply for VIS Transformer. It is a leading VIS framework based on Transformers, boasting the highest speed and accuracy on the YouTube-VIS dataset through a significant adaptation of DETR [2] for segmentation. It processes a fixed sequence of video frames with a ResNet backbone to extract features of each individual image independently. These are then concatenated and combined with a 3D positional encoding and fed into an encoder-decoder transformer that outputs a sequence of object prediction in order.

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-02-08-video-segmentation/vistr.png --alt VisTR architecture diagram  %}
</div>

{:refdef: class="image-caption"}
*VisTR architecture diagram*
{: refdef}

Things we need to highlight about this method:

1. Instance queries are fixed, learnable parameters that determine the number of instances that can be predicted.
2. Training involves instance sequence prediction over $N$ frames that requires ground truth matching to compute the loss.
3. The Hungarian algorithm is used to find a bipartite matching between ground-truth and prediction (as DETR [2]).
4. Solving 3 using masks is very expensive. A module is added to obtain bounding boxes from instance predictions and solve the matching efficiently.
5. The loss consists on a combination of the bounding box, the mask and the class prediction errors.
6. A 3D conv module fuses temporal information before masks are obtained.

Drawbacks: The models are trained on 8 V100 GPUs of 32G RAM, with 1 video clip per GPU. Either low resolution or short clips are used to fit in memory. Completely offline! VisTR remains as a complete offline strategy because it takes the entire video as an input (from ifc paper)

### IFC

Inter-frame Communication Transformers (IFC) is based on the following assumption: humans can summarize scenes with only a few words. Also, frames from a same video share a lot of commonalities, the difference between them can be communicated even with a small bandwidth. In order to solve computational overhead, only a few tokens, called memory tokens, are used to exchange information between frames. This minimizes space-time attention complexity. 

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-02-08-video-segmentation/ifc.png --alt IFC architecture diagram  %}
</div>

{:refdef: class="image-caption"}
*IFC architecture diagram*
{: refdef}

The architecture integrates Transformer encoders with ResNet feature maps and learnable memory tokens. Encoder blocks are composed of:
- Encode-Receive ($\xi$): fuses frame features and memory tokens, blending frame and temporal data.
- Gather-Communicate ($\zeta$): processes memory tokens across frames for inter-frame communication.

The decoder used a fixed number of object queries ($N_q$) that is indepentent on the number of input frames. It features two heads:

- A class head for class probability distribution of instances $p(c) \in \mathbb{R}^{N_q \times \|\mathbb{C}\|}$.
- A segmentation head producing $N_q$ conditional convolutional weights $w \in \mathbb{R}^{N_q \times C}$, convolved with the output of the spatial decoder (reshaped encoder output and upscaled à la fpn).

Loss calculation also follows DETR incorporating the Hungarian algorithm, applying it directly to the masks.

## References

- [[1](https://arxiv.org/pdf/1905.04804.pdf)] Yang, L., Fan, Y., & Xu, N. (2019). Video instance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 5188-5197).
- [[2](https://arxiv.org/pdf/2011.14503.pdf)] Wang, Y., Xu, Z., Wang, X., Shen, C., Cheng, B., Shen, H., & Xia, H. (2021). End-to-end video instance segmentation with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 8741-8750).
- [[3](https://arxiv.org/pdf/2005.12872.pdf)] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020, August). End-to-end object detection with transformers. In European conference on computer vision (pp. 213-229). Cham: Springer International Publishing.
- [[4](https://arxiv.org/pdf/2106.03299.pdf)] Hwang, S., Heo, M., Oh, S. W., & Kim, S. J. (2021). Video instance segmentation using inter-frame communication transformers. Advances in Neural Information Processing Systems, 34, 13352-13363.