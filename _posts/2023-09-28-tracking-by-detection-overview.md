---
layout: post
title: "An overview of tracking by detection"
subtitle: "A look at the evolution and precision of Multi-Object Tracking"
description: "Object tracking is a challenging problem in computer vision. It involves identifying and tracking the movement of objects in a video or image sequence. One common approach to object tracking is tracking by detection. This approach first uses an object detector to identify objects in each frame of the video. Then, a tracker is used to associate the detected objects across frames. This blog post provides an overview of tracking by detection. We will discuss the basics of the approach, and we will introduce some of the most popular tracking algorithms: SORT, DeepSORT, ByteTrack... We will also discuss the strengths and weaknesses of each algorithm, and we will provide some recommendations on which algorithm to choose for your application."
image: "/assets/images/fullsize/posts/2023-09-28-tracking-by-detection-overview/thumbnail.png"
selected: n
mathjax: y
---


## Table of Contents

<div class="table-wrapper" markdown="block">

|                           | Year | Appearence Features | Camera Compensation | HOTA MOT20 | Extra data |
|---------------------------|------|---------------------|---------------------|------------|------------|
| [SORT](#sort)             | 2016 | ❌                   | ❌                   |            | ❌          |
| [DeepSORT](#deepsort)     | 2017 | ✅                   | ❌                   |            | ✅          |
| [ByteTrack](#bytetrack)   | 2021 | ✅                   | ❌                   | 61.3       | ✅          |
| [BoT-SORT](#bot-sort)     | 2022 | ✅                   | ✅                   | 63.3       | ✅          |
| [SMILEtrack](#smiletrack) | 2022 | ✅                   | ✅  (?)              | 63.4       | ✅          |

</div>

## Introduction

Tracking by detection is an object tracking approach that first detects objects in each frame of a video and then associates the detections across frames. This is done by matching the detections based on their location, appearance, or motion. Tracking by detection has become the most popular method for addressing object tracking due to the rapid development of reliable object detectors.

The intention of this blog is to keep myself updated with the bibliography of tracking by detection methods. My intention is to regularly update this blog with new information and resources I find interesting.
I have included the SORT and DeepSORT papers in the list, despite being older methods, as they laid the groundwork for many of the techniques covered here.

## SORT 

It is a very good and simple work from 2016 that quickly became a standard in the field. Authors main goal was to create the fastest possible tracker relying on the quality of the object detector predictions. Appearance features of the objects are not used, only bounding box position and size. 

They employ two classical methods:

- **Kalman Filter:** is in charge of handling motion prediction, this is, figuring out where a track is going to move in the next frame given previous states. Track states are modeled with six different variables:
    
    $$
    \mathbf{x} = [u,v,s,r,\dot{u},\dot{v},\dot{s}]^T,
    $$
    
    These are the center of the target bounding box ($u, v$), the scales and aspect ratio of it ($s, r$) and their velocity components ($\dot{u},\dot{v},\dot{s}$).
    
- **Hungarian method:** used in the data association step to match new predictions with tracks based on IOU metric.

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2023-09-28-tracking-by-detection-overview/sort.jpg --alt SORT architecture diagram %}
</div>

{:refdef: class="image-caption"}
*SORT architecture diagram*
{: refdef}


1. An object detector returns bounding boxes for frame 0.
2. In T=0, a new track is created for each of the predicted bounding boxes
3. KF will predict a new position for each of the tracks
4. Object detector returns bounding boxes for frame 1
5. These bounding boxes are associated with tracks positions predicted by KF
6. New tracks are created for unmatched bounding boxes
7. Unmatched tracks can be terminated if they are not matched to any detection for $T_{Lost}$ frames. 
8. Matched tracks and new tracks are passed to the next time step
9. Back to 3

## DeepSORT

DeepSORT is an extension of SORT that uses appearance features. It adds a simple CNN extension that extracts appearance features from bounding boxes, improving object tracking, especially during occlusions. An object can be re-identified using appearance similarity after being occluded for a long period of time

Each track maintains a gallery of the last $$n$$ appearance descriptors, enabling cosine distance calculations between new detections and descriptors. Track age, determined by frames since the last association, plays a crucial role in the association process. DeepSORT adopts a cascade approach, prioritizing tracks with lower ages over a single-step association between predicted Kalman states and new measurements.

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2023-09-28-tracking-by-detection-overview/deepsort.jpg --alt DeepSORT architecture diagram  %}
</div>

{:refdef: class="image-caption"}
*DeepSORT architecture diagram*
{: refdef}

There is a small modification on the Kalman Filter prediction step that is included in the [code](https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py#L108){:target="_blank"}{:rel="noopener noreferrer"} but not mentioned in the original paper. The matrices $$Q$$, $$R$$ of the Kalman Filter were chosen in SORT to be time indepent, however in DeepSORT it was suggested to choose $$Q%$$, $$R$$ as functions of the scale of the bounding box. This can be due to the scale is less likely to change over time than other features and it can be also be used to compensate for changes in camera's viewpoint.

The cascade association step would look like this:

```python
for track_age in range(1, maximum_age):
    tracks_to_associate = get_tracks_with_age(tracks, track_age)
    associate(tracks_to_associate, detections)
    remove_associated_detections(detections)
```


## ByteTrack

ByteTrack is a recent object tracking algorithm that proposes a simple but effective optimization for the data association step. Most methods filter out detections with low confidence scores. This is because low-confidence detections are more likely to be false positives, or to correspond to objects that are not present in the scene. However, this can lead to problems when tracking objects that are partially occluded or that undergo significant appearance changes.

ByteTrack addresses this problem by using all detections, regardless of their confidence score. The algorithm works in two steps:

1. **High-confidence detections**: High-confidence detections are associated with tracks using intersection-over-union (IoU) or appearance features. Both approaches are evaluated in the results section of the paper.
2. **Low-confidence detections**: Low-confidence detections are associated with tracks using only IoU. This is because low-confidence detections are more likely to be spurious or inaccurate, so it is important to be more conservative when associating them with tracks.

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2023-09-28-tracking-by-detection-overview/bytetrack.jpg --alt ByteTrack architecture diagram  %}
</div>

{:refdef: class="image-caption"}
*ByteTrack architecture diagram*
{: refdef}

The ByteTrack algorithm has been shown to be very effective and it is currently among the top-performing methods on the [MOT Challenge leaderboard](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1){:target="_blank"}{:rel="noopener noreferrer"}.


## BoT-SORT

I personally love the BoT-SORT paper. It is build upon ByteTrack and it combines three different ideas that work very well together. These are:

1. **Kalman Filter update**: SORT introduced a way of modelling the track state vector using a seven-tuple $$\mathbf{x} = [x_c,y_c,a,h,\dot{x_c},\dot{y_c},\dot{s}]^T$$. BoT-SORT proposes to replace the scale and aspect ratio of the bounding box  ($$s$$, $$a$$) with the widht and height ($$w$$, $$h$$) to create an eight-tuple:

    $$
    \mathbf{x} = [x_c,y_c,w,h,\dot{x_c},\dot{y_c},\dot{w}, \dot{h}]^T
    $$

    They also choose Q, R matrices from the Kalman Filter as functions of the bounding box width and height. Recall that in DeepSORT, only the scale of the bounding box influenced on the Q, R matrices (see section 3.1 of [BoT-SORT paper](https://arxiv.org/pdf/2206.14651v2.pdf){:target="_blank"}{:rel="noopener noreferrer"} for more details).

2. **Camera Motion Compensation**: In dynamic camera situations, objects that are static can appear to move, and objects that are moving can appear to be static. The Kalman Filter does not take camera motion into account for its predictions, so BoT-SORT proposes to incorporate this knowledge. To do this, they use the global motion compensation technique (GMC) from the OpenCV Video Stabilization module. This technique extracts keypoints from consecutive frames and computes the homography matrix between the matching pairs. This matrix can then be used to transform the prediction bounding box from the coordinate system of frame $$k − 1$$ to the coordinates of the next frame $$k$$ (see section 3.2 of [BoT-SORT paper](https://arxiv.org/pdf/2206.14651v2.pdf){:target="_blank"}{:rel="noopener noreferrer"} to a full formulation on how incorporate the homography matrix in the prediction step).

    <div class="post-center-image">
        {% picture pimage /assets/images/fullsize/posts/2023-09-28-tracking-by-detection-overview/cmc.png --alt Camera movement example  %}
    </div>

    {:refdef: class="image-caption"}
    *Player is static on the pitch while throwing the ball but location on the image changes due to camera movement.*
    {: refdef}

3. 3. **IoU - ReID Fusion**:  BoT-SORT proposes a new way of solving the association step by combining motion and appearance information. The cost matrix elements are computed as follows:

    $$
    \hat{d}^{cos}_{i,j} = 
    \begin{equation}
    \begin{cases}
    0.5 \cdot {d}^{cos}_{i,j}, ({d}^{cos}_{i,j} < \theta_{emb}) \hat{} ({d}^{iou}_{i,j} < \theta_{iou})\\
    1, \text{otherwise}
    \end{cases}
    \end{equation}
    $$

    $$
    C_{i,j} = min(d^{iou}_{i,j}, \hat{d}^{cos}_{i,j})
    $$

    The appearence distance is recomputed as shown in the first equation. The idea is to filter out pairs with large iou or large appearance distance (two different thresholds are used here).  Then, the cost matrix element is updated as the minimum between the IoU and the new appearance distance. This method seems to be handcrafted, and the authors likely spent a significant amount of time evaluating different thresholds on the MOT17 dataset to arrive at this formulation. Note thresholds are callibrated using MOT17 validation set. 

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2023-09-28-tracking-by-detection-overview/botsort.jpg --alt BoT-SORT architecture diagram  %}
</div>

{:refdef: class="image-caption"}
*ByteTrack architecture diagram*
{: refdef}


## SMILEtrack

This method currently holds the title of being the State-of-the-Art (SOTA) in the MOT17 and MOT20 datasets. It builds upon ByteTrack but throws in a handful of fresh ideas designed to give appearance features more importance.

I spent a couple hours trying to understand the paper but I have to admit it felt very confusing to me, so I went straight to the [code](https://github.com/pingyang1117/SMILEtrack_Official){:target="_blank"}{:rel="noopener noreferrer"}. Things got even trickier there; I noticed quite a few things that didn't align with what was mentioned in the paper. As a results, so I opened an [issue](https://github.com/pingyang1117/SMILEtrack_Official/issues/3){:target="_blank"}{:rel="noopener noreferrer"} on the project's GitHub repository. I'll update this section once I hear back from the authors.

<!-- Let's see what they are:

1. **Similarity Learning Module (SLM)**: It is a Siamese network that computes appearence similarity between two objects using a Patch Self-Attention (PSA) block. Think about it as a boosted feature descriptor that incorporates attention mechanism following ViT style.

2. **Similarity Matching Cascade (SMC)**: Very similar to ByteTrack, it splits the data association step in two parts depending on detection scores. First, high confidence detections are tried to be matched with the tracks, for then proceeding with the low confidence ones. In both cases, IoU and appearence features are used to compute the cost matrix. The key addition is a new **GATE function** that is used right after the high confidence association. Unmatched objects with high scores might find matches in subsequent frames due to occlusions or lighting changes. When an object passes this GATE function, a new track is created for it.

Idea is, if iou is high but they don't look alike, probably occlusion. If they don't match but they have some past track that looks alike, create a new track for it (this i don't understand very well)
There is camera motion correction in the code but not in the paper!!!!
 -->


## References

- [[1](https://arxiv.org/pdf/1602.00763.pdf)] Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016, September). Simple online and realtime tracking. In 2016 IEEE international conference on image processing (ICIP) (pp. 3464-3468). IEEE.
- [[2](https://arxiv.org/pdf/1703.07402.pdf)] Wojke, N., Bewley, A., & Paulus, D. (2017, September). Simple online and realtime tracking with a deep association metric. In 2017 IEEE international conference on image processing (ICIP) (pp. 3645-3649). IEEE.
- [[3](https://arxiv.org/pdf/2110.06864.pdf)] Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., ... & Wang, X. (2022, October). Bytetrack: Multi-object tracking by associating every detection box. In European Conference on Computer Vision (pp. 1-21). Cham: Springer Nature Switzerland.
- [[4](https://arxiv.org/pdf/2206.14651.pdf)] Aharon, N., Orfaig, R., & Bobrovsky, B. Z. (2022). BoT-SORT: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651.
- [[5](https://arxiv.org/pdf/2211.08824.pdf)] Wang, Y. H. (2022). SMILEtrack: SiMIlarity LEarning for Multiple Object Tracking. arXiv preprint arXiv:2211.08824.