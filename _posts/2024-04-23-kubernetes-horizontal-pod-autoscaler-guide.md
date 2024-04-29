---
layout: post
title: "A Guide to Horizontal Pod Autoscaler"
subtitle: "Understand and visualize how Kubernetes HPA works with a real world example"
description: "Discover how Kubernetes' Horizontal Pod Autoscaler (HPA) functions using a real use case. Learn to manage and observe HPA in action with practical examples. This post will show you how to optimize resource usage, streamline pod scaling, and enhance application performance using a simple visualization tool."
image: "/assets/images/fullsize/posts/2024-04-23-kubernetes-horizontal-pod-autoscaler-guide/test.jpg"
selected: y
mathjax: y
---

A few weeks ago, while reviewing service metrics in Grafana, I noticed some unexpected behaviour in one of our services—there were more pods than necessary given the current traffic load. This led me to uncover that the extra pods were spawned by the Horizontal Pod Autoscaler (HPA) based on the metrics we had configured (a while ago). Understanding HPA took me a few hours. This is a task typically handled by specialized teams in larger companies, but working at a startup forces you to wear many hats and I often find myself analyzing how models perform in production. In this post, I'll discuss the issues I encountered with HPA and demonstrate how a simple [visualization tool](#visualization-tool)    can help anticipate the number of replicas needed.

## What is HPA?

The [Horizontal Pod Autoscaler (HPA)](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/) in Kubernetes automatically adjusts the number of pod replicas in a deployment, replicaset, or statefulset based on observed CPU utilization or other select metrics. This feature is very useful for managing application scalability and resource efficiency, particularly in environments with variable workloads.

For example Statsbomb can use HPA to handle increased traffic during a weekend when there are more games being played. The HPA can automatically scale up the number of web server pods to maintain performance, and scale down during off-peak hours to reduce costs. This dynamic adjustment helps ensure that the application consistently meets performance targets without manual intervention.

In next sections I will briefly explain how HPA works and how to use [this simple tool](#visualization-tool) to ease your scaling decisions.

## How does HPA work?

First of all we need to make sure we understand the concepts of `requests` and `limits` since they are fundamental to how resources are allocated and managed across the pods in a cluster.

- **Requests**: This value specifies the amount of CPU or memory that Kubernetes guarantees to a pod. When a pod is scheduled, the Kubernetes scheduler uses this request value to decide on which node the pod can fit. So this number ensures the pod has the resources it needs to run.
- **Limits**: This value specifies the maximum amount of CPU or memory that a pod can use. If a pod exceeds this limit, Kubernetes will throttle the pod or kill it. This is how k8s ensures that a single pod does not consume all the resources in a node.

Imagine our deployment has the following setup:

```yaml
# deployment.yaml
resources:
  limits:
    memory: 2000Mi
    cpu: 1500m
  requests:
    memory: 1350Mi
    cpu: 500m
    
```

And our HPA is configured in the following manner:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: autoscaler-name
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autoscaler-name
  minReplicas: 1
  maxReplicas: 3
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 90
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 90
```

So what does this mean? Well if you are a proper engineer what you would do is check the [official docs](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/#algorithm-details) and try to carefully understand this. But if you are like me... you probably would make some assumptions and hope for the best (only to end up having to read the docs 😅). 

My first guess was that this would configure HPA to scale up the number of pods when memory or CPU usage exceeded 90%. However, I overlooked a crucial detail: the calculation also needs to include the current number of replicas. Here’s how HPA actually works:

```
desiredReplicas = ceil[currentReplicas * ( currentMetricValue / desiredMetricValue )]
```

Where:
- `currentReplicas` is the number of replicas the deployment is currently running.
- `currentMetricValue` is the current value of the metric we are monitoring (e.g. CPU usage).
- `desiredMetricValue` is the target value for the metric we are monitoring (e.g. 90% requested CPU usage).

So let's see what I observed in Grafana that day.

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-04-23-kubernetes-horizontal-pod-autoscaler-guide/memory_cpu.png --alt Grafana plot showing memory and CPU usages %}
</div>

{:refdef: class="image-caption"}
*Figure 1. Grafana plot showing memory and CPU usages*
{: refdef}

What I observed was that at 13:10, we had three pods running with memory usage around `1100 MB` and CPU usage less than `100m`. Both metrics appeared to be below the target values: `1215 MB (=0.9*1350 MB)` for memory and `450m (=0.9*500m)` for CPU. So, why were there three pods running?


## Visualization tool 

Before moving forward with the debbuging I would like to introduce the visualization tool I have built. It allows you to enter the specific details of your current/target metrics, as well as the current number of replicas. Based on those inputs, it computes and displays the desired number replicas using the scaling formula above.
  
{% include hpa.html %}

## Debugging HPA

Let's start by inspecting what has happened to our application step by step using our visualization tool using previous metrics. First, let's check the memory usage:

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-04-23-kubernetes-horizontal-pod-autoscaler-guide/memory_init.png --alt HPA Memory expected replicas %}
</div>

{:refdef: class="image-caption"}
*Figure 2. HPA memory expected replica*
{: refdef}

The memory usage seems to be below the 90% so the number of replicas would be set to 1. Let's check the CPU usage now:

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-04-23-kubernetes-horizontal-pod-autoscaler-guide/cpu_init.png --alt HPA CPU expected replicas %}
</div>

{:refdef: class="image-caption"}
*Figure 3. HPA CPU expected replica*
{: refdef}


At startup, the CPU usage exceeded the target value of `450m`. This means that the 'currentMetricValue / desiredMetricValue' ratio was greater than one, indicating that the autoscaler needed to scale up the replicas. But by how much? Let’s adjust the x-axis of the plot to display the number of replicas:

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-04-23-kubernetes-horizontal-pod-autoscaler-guide/cpu_stairs.png --alt HPA CPU expected replicas showing stairs pattern %}
</div>

{:refdef: class="image-caption"}
*Figure 4. HPA CPU current replicas vs expected replicas*
{: refdef}

There you go! We can clearly see the staircase pattern, similar to $f(x) = x + 1$. This occurs because the 'currentMetricValue / desiredMetricValue' ratio is greater than 1, prompting the autoscaler to continuously increase the number of replicas until it reaches the maximum allowed. In this instance, maxReplicas was set to 3. Thus, we have identified the root of the problem!

## Why is HPA not scaling down?

Although CPU usage spiked at startup, it quickly returned to low levels. So why isn't the HPA scaling down the number of replicas? It appears that the CPU requirement is well below the target value of `450m`, as illustrated in Figure 1. According to the official HPA documentation:

> "If multiple metrics are specified in a HorizontalPodAutoscaler, this calculation is done for each metric, and then the largest of the desired replica counts is chosen."

This indicates that the issue now lies with memory usage. Grafana shows us that memory usage has remained constant after the scaling. According to Figure 2, the expected number of replicas should be just 1. However, since the HPA previously increased our replicas to 3, when we view the same plot with the number of replicas on the x-axis, it reveals the following:

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-04-23-kubernetes-horizontal-pod-autoscaler-guide/memory_stairs.png --alt HPA memory current replicas vs expected replicas %}
</div>

{:refdef: class="image-caption"}
*Figure 5. HPA memory current replicas vs expected replicas*
{: refdef}

With the current memory usage, the HPA behaves like the function $f(x) = x$, preventing the number of replicas from scaling down. This is why we continuously see three pods running, even though the pods are not receiving much traffic.

## What can we do?

We have a couple of options to fix this problem. For instance, we could change the memory and CPU targets in the HPA settings. But this isn't a lasting solution because if our application's memory use changes, we could face the same issue again. Instead, we should look at the main cause: the constant memory usage.

Our application's memory use stays the same no matter how many pods are running or how much traffic we have. Because of this, the HPA acts like a function where $f(x) = x$. This means adjusting the number of pods based on memory doesn't help since the memory doesn’t change with the traffic. The best approach is to stop using the memory metric in the HPA settings and rely only on the CPU metric.

## Conclusion

In this post, we looked at how the Horizontal Pod Autoscaler (HPA) in Kubernetes helps manage the number of pods based on CPU and memory usage. This is very important for keeping applications running smoothly as demands change over time. However, we also learned that it's crucial to keep an eye on the HPA settings and adjust according to your real-world scenario.

We have learnt how different metric patterns can affect the HPA's behaviour and on deciding on which metrics to use. We should only use HPA with those metrics which values fluctuate with the traffic or with the number of pods running.

Through an example, we've seen how crucial it is to choose the right metrics for HPA. Effective HPA deployment relies on using metrics that genuinely reflect changes in workload and resource needs. Metrics that do not vary with traffic levels or pod count might not be suitable for making scaling decisions.

For anyone using Kubernetes, understanding how to manage these settings is key, whether you're in a small startup or a large company. It's all about understanding your systems well and making the right adjustments.