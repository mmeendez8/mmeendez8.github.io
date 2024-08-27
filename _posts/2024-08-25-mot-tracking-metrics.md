---
layout: post
title: "Understanding MOT Metrics"
subtitle: "Comparing MOTA, IDF1, and HOTA for Multi-Object Tracking Evaluation"
description: "Explore the essential metrics in Multi-Object Tracking (MOT) with this comprehensive guide. We break down and compare key metrics like MOTA, IDF1, and HOTA, while also covering DetA and AssA. Understand how these metrics assess spatial accuracy and temporal consistency in object tracking."
image: "/assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/thumbnail.jpg"
selected: y
mathjax: y
---

After a long time, I have finally sat down to write this blog post on tracking metrics. It builds on my last post about [tracking by detection]({% post_url 2023-11-08-tracking-by-detection-overview %}) and delves deeper into how we measure tracking performance.

In this article, I'll provide an introduction to tracking metrics, starting from the basic principles and breaking down the key differences between various metrics. I'll focus on three popular metrics: MOTA, IDF1, and HOTA, which are widely used in the Multi-Object Tracking (MOT) community. Understanding these is crucial because the choice of metric can significantly impact how we interpret a tracker's performance.

Let's get started!

## The basics

**DetA** (Detection Accuracy) measures how well a tracker localizes objects in each frame, typically using Intersection over Union (IoU) thresholds. It essentially quantifies the spatial accuracy of detections. 

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/iou.png --alt IoU diagram %}
</div>

{:refdef: class="image-caption"}
*Figure 1. IoU diagram [from jonathanluiten](https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1){:target="_blank"}{:rel="noopener noreferrer"}*
{: refdef}

**AssA** (Association Accuracy), on the other hand, evaluates how accurately a tracker maintains object identities across frames. It focuses on the temporal consistency of ID assignments, measuring how well the tracker links detections of the same object over time. See, for example, the image below, extracted from the HOTA paper, where different tracking associations (A, B, C) are shown for the ground truth track (GT).

<div class="post-center-image" style="max-width: 300px; margin: 0 auto;">
    {% picture pimage /assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/hota_assa_example.png --alt AssA example from HOTA paper %}
</div>

{:refdef: class="image-caption"}
*Figure 2. Different association results example (from HOTA paper)*
{: refdef}

The **Hungarian algorithm** plays a crucial role in many tracking metrics, including IDF1 and HOTA. This algorithm solves the assignment problem, which in tracking means matching predicted objects to ground truth objects in a way that maximizes overall matching quality.
In tracking metrics, the Hungarian algorithm is typically used to:

1. Find the optimal bipartite matching between detections and ground truth objects in each frame
2. Determine the best global assignment of tracks to ground truth trajectories across the entire sequence

While a detailed explanation of the algorithm is beyond the scope of this post, understanding its basic function helps in grasping how these metrics work.
For a more in-depth explanation of the Hungarian algorithm, check out [this excellent tutorial](https://www.thinkautonomous.ai/blog/hungarian-algorithm/){:target="_blank"}{:rel="noopener noreferrer"}.

## MOTA (Multiple Object Tracking Accuracy)

MOTA introduces the concept of identity tracking to object detection metrics. It's a straightforward metric that incorporates identity switches (IDSW) - cases where a single ground truth (GT) object is assigned to different track predictions over time.

The computation of MOTA involves a temporal dependency. It penalizes cases where track assignments change between consecutive frames. An identity switch is counted if a ground truth target i is matched to track j in the current frame but was matched to a different track k (where k ≠ j) in the previous frame.

 In [TrackEval code](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py#L81){:target="_blank"}{:rel="noopener noreferrer"} this is done using a simple gating trick:

```yaml
score = IoU(GT, pred)
if pred != previous_assigned_id(GT):
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
3. **IoU is fixed** so more or less detection accuracy is not reflected on the metric

## IDF1

IDF1 addresses some of MOTA's limitations by focusing on how long the tracker correctly identifies an object, rather than just counting errors. It's based on the concept of Identification Precision (IDP) and Identification Recall (IDR).

It computes the assignment between prediction and ground truth objects across the entire video, rather than frame by frame. 

The metric is simple:

$$
\text{IDF1} = \frac{2 * \text{IDTP}}{2 * \text{IDTP} + \text{IDFP} + \text{IDFN}}
$$

Where:

- **IDTP** (ID True Positive): The number of correctly identified detections
- **IDFP** (ID False Positive): Tracker hypotheses that don't match any ground truth
- **IDFN** (ID False Negative): Ground truth trajectories that aren't tracked

The global assignment is computed using the Hungarian algorithm. It pick the best combination between prediction and ground truth for the whole video. It is easier to understand by observing the image introduced in HOTA paper:


<div class="post-center-image" style="max-width: 800px; margin: 0 auto;">
    {% picture pimage /assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/idf1.png --alt IDF1 metric diagram %}
</div>

{:refdef: class="image-caption"}
*Figure 3. IDF1 metric diagram*
{: refdef}

The main problem I see with IDF1 is finding the best one-to-one matching between predicted and ground truth trajectories for the entire sequence. However, this approach can sometimes oversimplify complex tracking scenarios:

Imagine a corner kick in football. A tracker might correctly follow Player A running into the box, lose them in a cluster, and then mistakenly pick up Player B after the ball is cleared. IDF1 might treat this as one partially correct track for either Player A or B, ignoring that it's correct for different players at different times.

This simplification can misrepresent a tracker's performance in complex situations like crowded football plays, where player interactions and occlusions are frequent.

Key advantages of IDF1:

1. It's more sensitive to **long-term tracking consistency**.
2. It balances **precision and recall** of identity predictions.
3. It's less affected by the number of objects in the scene than MOTA.

However, IDF1 also has limitations:

1. **IDF1 can decrease when improving detection**. Just avoiding FP can result in a better metric (A vs C)
2. **IoU is fixed** so more or less detection accuracy is not reflected on the metric

More limitations are presented in the HOTA paper. I recommend you to have a read because it is very well explained and intuitive.

## HOTA (Higher Order Tracking Accuracy)

HOTA is a more recent metric designed to address the limitations of both MOTA and IDF1. It aims to provide a balanced assessment of detection and association performance. HOTA can be broken down into DetA (Detection Accuracy) and AssA (Association Accuracy), allowing separate analyses of these aspects.

The core HOTA formula is:

$$
 \text{HOTA}_{\alpha} = \sqrt{\text{DetA}_{\alpha} \cdot \text{AssA}_{\alpha}} 
$$

In this formula, the alpha term represents the different Intersection over Union (IoU) thresholds used to compute the metric. A True Positive (TP) is only considered when the match IoU score is above the given alpha threshold. The metric uses 19 different alpha values, ranging from 0.05 to 0.95 in increments of 0.05.

HOTA uses global alignment (high-order association) between predicted and ground truth detections, similar to IDF1, but also incorporates localization accuracy. This means that HOTA evaluates both the ability to detect objects accurately and to maintain correct associations over time.

The HOTA algorithm can be summarized in the following steps:

```yaml
For each frame
	For each alpha value
		Perform matching between gt and preds (Hungarian algorithm)
		Obtain TP, FP and FN from previous matching
		Compute association accuracy across the entire video for each TP.
```

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-08-25-mot-tracking-metrics/hota.jpg --alt HOTA metric diagram %}
</div>

{:refdef: class="image-caption"}
*Figure 4. HOTA metric diagram*
{: refdef}

HOTA drawbacks:

- **Not Ideal for Online Tracking**: HOTA's association score depends on future associations across the entire video, making it less suitable for evaluating online tracking where future data isn't available.
- **Doesn't Account for Fragmentation**: HOTA does not penalize fragmented tracking results, as it is designed to focus on long-term global tracking, which may not align with all application needs.

If you want to learn more about HOTA, I recommend reading the blog post by [Jonathon Luiten](https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1){:target="_blank" rel="noopener noreferrer"}. Jonathon Luiten is one of the authors of the HOTA paper, and his post is an excellent resource for learning how to use the metric to compare different trackers.

## Conclusion

In this post, we've explored three key metrics used in Multi-Object Tracking: MOTA, IDF1, and HOTA. Each metric offers unique insights into tracking performance, with its own strengths and limitations. MOTA provides a straightforward measure but can oversimplify in complex scenarios. IDF1 focuses on long-term consistency but may not fully capture detection improvements. HOTA, which attempts to balance detection and association accuracy, has emerged as the standard metric used today for benchmarking tracking algorithms.


## References

- [[1](https://arxiv.org/pdf/1603.00831){:target="_blank"}{:rel="noopener noreferrer"}] Milan, A., Leal-Taixé, L., Reid, I., Roth, S., & Schindler, K. (2016). MOT16: A benchmark for multi-object tracking. arXiv preprint arXiv:1603.00831.
- [[2](https://arxiv.org/pdf/1609.01775){:target="_blank"}{:rel="noopener noreferrer"}] Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016, October). Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision (pp. 17-35). Cham: Springer International Publishing.
- [[3](https://arxiv.org/pdf/2009.07736){:target="_blank"}{:rel="noopener noreferrer"}] Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2021). Hota: A higher order metric for evaluating multi-object tracking. International journal of computer vision, 129, 548-578.