---
layout: post
title: "Understanding Object Tracking Metrics"
subtitle: "Comparing MOTA, IDF1, and HOTA for Multi-Object Tracking Evaluation"
author: Miguel Mendez
description: "Explore the essential Object Tracking metrics with this comprehensive guide. We break down and compare key metrics like MOTA, IDF1, and HOTA, while also covering DetA and AssA. Understand how these metrics assess spatial accuracy and temporal consistency in object tracking."
image: "/assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/thumbnail.jpg"
selected: y
mathjax: y
tags: [MOT, tracking, metrics, MOTA, IDF1, HOTA]
categories: [Computer Vision, Machine Learning, Tracking]
---

After a long time, I have finally sat down to write this blog post on tracking metrics. It builds on my last post about [tracking by detection]({% post_url 2023-11-08-tracking-by-detection-overview %}) and explores how we measure tracking performance.

In this article, I'll provide an introduction to tracking metrics, starting from the basic principles and breaking down the key differences between various metrics. I'll focus on three popular metrics: MOTA, IDF1, and HOTA, which are widely used in the [Multi-Object Tracking (MOT) community](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1){:target="_blank"}{:rel="noopener noreferrer"}. Understanding these is crucial because the choice of metric can significantly impact how we interpret a tracker's performance.

Let's get started!

## The basics


### Hungarian algorithm 

The Hungarian algorithm plays a crucial role for tracking metrics, primarily used to:

1. Optimize bipartite matching between detections and ground truth objects per frame
2. Assign tracks to ground truth trajectories across the entire sequence

Let's for now focus on the first point. The algorithm matches predicted tracks to ground truth objects in each frame, maximizing overall IoU scores. This results in:

- **TP**: True Positives (matches with IoU above threshold)
- **FP**: False Positives (unmatched predictions)
- **FN**: False Negatives (unmatched ground truth objects)

While a detailed explanation of the algorithm is beyond the scope of this post, understanding its basic function helps in grasping how these metrics work.
For a more in-depth explanation of the Hungarian algorithm, check out [this excellent tutorial](https://www.thinkautonomous.ai/blog/hungarian-algorithm/){:target="_blank"}{:rel="noopener noreferrer"}.

### DetA

The Detection Accuracy (DetA) measures how well a tracker localizes objects in each frame, typically using Intersection over Union (IoU) thresholds. It essentially quantifies the spatial accuracy of detections. 

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/iou.png --alt IoU diagram %}
</div>

{:refdef: class="image-caption"}
*Figure 1. IoU diagram [from jonathanluiten](https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1){:target="_blank"}{:rel="noopener noreferrer"}*
{: refdef}

So once we have the TP, FP, and FN, we can compute the DetA as:

$$ \text{DetA} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} $$


### AssA

The Association Accuracy (AssA), on the other hand, evaluates how accurately a tracker maintains object identities across frames. It focuses on the temporal consistency of ID assignments, measuring how well the tracker links detections of the same object over time. See, for example, the image below, extracted from HOTA [[1]](#references): 

<div class="post-center-image" style="max-width: 300px; margin: 0 auto;">
    {% picture pimage /assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/hota_assa_example.png --alt AssA example from HOTA paper %}
</div>

{:refdef: class="image-caption"}
<a id="figure-2"></a>
*Figure 2. Different association results example (from HOTA [[1]](#references))*
{: refdef}

We can observe different tracking results (A, B, C) for a single ground truth object (GT):

- **A**: Detects the object 50% of the time with consistent identity
- **B**: Detects the object 70% of the time, but assigns two different identities
- **C**: Detects the object 100% of the time, but assigns up to four different identities

Which result is best? This is what the Association Accuracy (AssA) metric aims to determine. Different tracking metrics like MOTA, IDF1, and HOTA approach this question in various ways, each with its own methodology and emphasis on detection accuracy versus identity consistency.

## MOTA (Multiple Object Tracking Accuracy)

MOTA introduces the concept of identity tracking to object detection metrics. It incorporates identity switches (IDSW), which occur when a single ground truth (GT) object is assigned to different track predictions over time.

The computation of MOTA involves temporal dependency, penalizing track assignment changes between consecutive frames. An IDSW is counted when a GT target $i$ matches track $j$ in the current frame but was matched to a different track $k$ ($k ≠ j$) in the previous frame.

In practice, the Hungarian matching algorithm is modified to minimize identity switches from the previous frame. In [TrackEval code](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py#L81){:target="_blank"}{:rel="noopener noreferrer"} this is done using a simple gating trick:

```python
score = IoU(GT, pred)
if pred == previous_assigned_id(GT):
    score = score * 1000
```

The MOTA metric is computed across all frames as:

$$
\text{MOTA} = 1 - \frac{\sum_t (FN_t + FP_t + IDSW_t)}{\sum_t GT_t}
$$

Where $t$ is the frame index, FN are False Negatives, FP are False Positives, IDSW are Identity Switches, and GT is the number of ground truth objects.

While MOTA's simplicity is appealing, it has some limitations:

1. It only considers the **previous frame** for IDSW, so each switch is penalized **only once**, regardless of how long the incorrect assignment persists.
2. It can be **dominated by FP and FN** in crowded scenes, making IDSW less impactful.
3. **IoU threshold is fixed** so more or less detection accuracy is not reflected on the metric

## IDF1

IDF1 addresses some of MOTA's limitations by focusing on how long the tracker correctly identifies an object, rather than just counting errors. It's based on the concept of Identification Precision (IDP) and Identification Recall (IDR).

It computes the assignment between prediction and ground truth objects across the entire video, rather than frame by frame. 

The metric is simple:

$$
\text{IDF1} = \frac{2 * \text{IDTP}}{2 * \text{IDTP} + \text{IDFP} + \text{IDFN}}
$$

Where:

- **IDTP** (ID True Positive): The number of correctly identified detections
- **IDFP** (ID False Positive): Tracker predictions that don't match any ground truth
- **IDFN** (ID False Negative): Ground truth trajectories that aren't tracked

The global assignment is computed using the Hungarian algorithm. It picks the best combination between prediction and ground truth that maximizes IDF1 for the whole video. It is easier to understand this by observing the image introduced in HOTA paper:

<div class="post-center-image" style="max-width: 800px; margin: 0 auto;">
    {% picture pimage /assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/idf1.png --alt IDF1 metric diagram %}
</div>

{:refdef: class="image-caption"}
*Figure 3. IDF1 metric diagram*
{: refdef}

The main problem I see with IDF1 is finding the best one-to-one matching between predicted and ground truth trajectories for the entire sequence since it can oversimplify complex tracking scenarios:

Imagine a corner kick in football. A tracker might correctly follow Player A running into the box, lose them in a cluster, and then mistakenly pick up Player B after the ball is cleared. IDF1 might treat this as one partially correct track for either Player A or B, ignoring that it's correct for different players at different times.

This simplification can misrepresent a tracker's performance in complex situations like crowded football plays, where player interactions and occlusions are frequent.

Key advantages of IDF1:

1. It's more sensitive to **long-term tracking consistency**.
2. It balances **precision and recall** of identity predictions.
3. It's less affected by the number of objects in the scene than MOTA.

However, IDF1 also has limitations:

1. **IDF1 can decrease when improving detection**. Just avoiding FP can result in a better metric (A vs C in [Figure 2](#figure-2))
2. **IoU threshold is fixed** so more or less detection accuracy is not reflected on the metric

More limitations are presented in the HOTA paper. I recommend you to have a read because it is very well explained and intuitive.

## HOTA (Higher Order Tracking Accuracy)

HOTA is a more recent metric designed to address the limitations of both MOTA and IDF1. It aims to provide a balanced assessment of detection and association performance. HOTA can be broken down into DetA (Detection Accuracy) and AssA (Association Accuracy), allowing separate analyses of these aspects.

The core HOTA formula is:

$$
 \text{HOTA}_{\alpha} = \sqrt{\text{DetA}_{\alpha} \cdot \text{AssA}_{\alpha}} 
$$

In this formula, the $\alpha$ term represents the different Intersection over Union (IoU) thresholds used to compute the metric. A True Positive (TP) is only considered when the match IoU score is above the given $\alpha$ threshold. The metric uses 19 different $\alpha$ values, ranging from 0.05 to 0.95 in increments of 0.05.

HOTA uses global alignment between predicted and ground truth tracks across the entire video, similar to IDF1. Ideally, we would evaluate every possible association like this:

```python
for each frame:
    for each α:
        matching between gt and preds (Hungarian algorithm)
        obtain TP, FP and FN from previous matching
        compute AssA across the entire video for each TP.
```

But this would be computationally expensive. Instead, it uses a more efficient approach that approximates a similar result. [TrackEval implementation](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py#L53) uses a two-pass approach to balance computational efficiency with accuracy:

```python
# Pass 1: Build global ID relationships
for each frame:
    potential_matches += normalize_similarity(frame_similarity)

# Jaccard-like aligment between GT and predictions
global_alignment = potential_matches / (gt_count + pred_count - potential_matches)

# Pass 2: Optimal matching per frame
for each frame:
    frame_score = global_alignment[frame_gt_ids, frame_pred_ids] * frame_similarity
    matching = hungarian_algorithm(frame_score)

    for each α:
        valid_matches = frame_similarity[matching] >= α
        TP[α] += count(valid_matches)
        FN[α] += count(unmatched_gt)
        FP[α] += count(unmatched_pred)

# Final compute aggregated metrics
for each α:
    AssA[α] = compute_AssA(TP[α], FP[α], FN[α])
```

The first pass builds global ID relationships between ground truth and predictions across all frames. The second pass uses these global relationships, weighted by frame-specific similarities, to find optimal matchings using the Hungarian algorithm.

The key insight is that HOTA weighs each potential match by both:
- How often these IDs appear together **globally**  
- How well they match in this **specific frame** 

This approximates evaluating all possible associations without the computational cost.

In the original paper, Ass-IoU is referred to as the metric obtained by computing DetA across the entire sequence for a single true positive (TP) match in the current frame. The AssA metric can then be defined as follows:

$$\text{AssA} = \frac{1}{|\text{TP}|} \sum_{c \in \text{TP}} \text{Ass-IoU}(c) $$

HOTA drawbacks:

- **Not Ideal for Online Tracking**: HOTA's association score depends on future associations across the entire video, making it less suitable for evaluating online tracking where future data isn't available.
- **Doesn't Account for Fragmentation**: HOTA does not penalize fragmented tracking results, as it is designed to focus on long-term global tracking, which may not align with all application needs.

If you want to learn more about HOTA, I recommend reading the blog post by [Jonathon Luiten](https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1){:target="_blank"}{:rel="noopener noreferrer"}{:target="_blank" rel="noopener noreferrer"}. He is one of the authors of the HOTA paper, and his post is an excellent resource for learning how to use the metric to compare different trackers.

## How do these metrics compare to each other?

We have examined how MOTA, IDF1, and HOTA function. Each metric has its own strengths and limitations. While HOTA is generally recommended for most applications, the choice of metric ultimately depends on your specific tracking scenario. The HOTA paper provides an excellent comparison that effectively captures the differences between these metrics:

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/metrics.png --alt Metric comparison %}
</div>

{:refdef: class="image-caption"}
*Figure 5. Metric comparison*
{: refdef}

Having already introduced the left side of the image, let's now focus on the right side, which displays metrics for each tracker output. The leftmost metric, DetA, exclusively evaluates detection quality. It yields the best results when the tracker accurately detects objects, regardless of their track ID.
On the opposite end, we have AssA (derived from the HOTA definition). This metric prioritizes track ID consistency, which is why output A performs best in this category.
The authors demonstrate how HOTA positions itself in the middle, striking a balance between detection quality and association accuracy.

The most suitable metric depends on your specific application. For instance:
1. If you're developing a simple camera system to count people in a room, you might prioritize detection quality (DetA).
2. In a criminal tracking system where maintaining consistent track IDs is crucial, you should focus on AssA.
3. For most applications, such as sports tracking systems, you'll need to balance both aspects. In these scenarios, HOTA emerges as the optimal choice, providing a comprehensive evaluation of tracker performance.

## Conclusion

In this post, we've explored three key metrics used in Multi-Object Tracking: MOTA, IDF1, and HOTA. Each metric offers unique insights into tracking performance, with its own strengths and limitations. MOTA provides a straightforward measure but may be oversimplistic in complex scenarios. IDF1 focuses on long-term consistency but may not fully capture detection improvements. HOTA, which attempts to balance detection and association accuracy, has emerged as the standard metric used today for benchmarking tracking algorithms.


## References

- [[1](https://arxiv.org/pdf/1603.00831){:target="_blank"}{:rel="noopener noreferrer"}] Milan, A., Leal-Taixé, L., Reid, I., Roth, S., & Schindler, K. (2016). MOT16: A benchmark for multi-object tracking. arXiv preprint arXiv:1603.00831.
- [[2](https://arxiv.org/pdf/1609.01775){:target="_blank"}{:rel="noopener noreferrer"}] Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016, October). Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision (pp. 17-35). Cham: Springer International Publishing.
- [[3](https://arxiv.org/pdf/2009.07736){:target="_blank"}{:rel="noopener noreferrer"}] Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2021). Hota: A higher order metric for evaluating multi-object tracking. International journal of computer vision, 129, 548-578.