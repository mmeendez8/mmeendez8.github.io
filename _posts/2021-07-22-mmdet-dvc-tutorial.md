---
layout: post
title:  "Train a mmdetection model and monitor its progress with DVC"
subtitle: "Train a reproducible model extracting the best of mmdetection and DVC frameworks"
description: ""
image: "/assets/posts/2021-07-19-new-docker-cache-is-out/thumbnail.jpg"
selected: y
---

I recently [published a post]({% post_url 2021-07-01-dvc-tutorial %})  where I show how we use DVC to maintain versions of our datasets so we reduce data reproducibility problems to a minimum. This post is intended to be a second part of this tutorial, my intention here is to show how we can combine the power of mmdetection framework and its huge [model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) with DVC for designing ML pipelines, versioning our models and monitor training progress.

It is quite a lot of content to cover, so I will be going through it step by step and trying to keep things as simple as possible. You can find all the code for this tutorial in my [Github](). So let's start with it!

### 1. Setup the environment

We are gonna need a few packages to get our up and running. I have created a [conda.yaml]() that you can find in the root of the repository, this is going to install pytorch and cudatoolkit since we are going to train our models using a GPU. You can create the environment by:

```bash
conda env create -f conda.yaml
conda activate mmdetection_dvc
```

Note that this might take a while, since it will install tens of packages.

### 2. Import our dataset

In the previous post we used a subset of the COCO dataset created by fast.ai. We push all data to a Google Drive remote storage using DVC and keep all metada files in a Github repository. It's time to import the dataset in our repository and for this we can use [dvc import](https://dvc.org/doc/command-reference/import) that automatically deals with all this logic.

```bash
dvc init
dvc import "git@github.com:mmeendez8/coco_sample.git" "data/" -o "data/"
```

Note that we are importing the dataset into the `data/` directory, this is where all our data will be stored. This may take a while since we need to download all images and annotations from the remote gdrive storage. Let's now publish the changes on git:

```bash
git add . 
git commit -m "Import coco_sample dataset" 
git push
```

Once it is downloaded, we can move between [versions of the dataset](https://github.com/mmeendez8/coco_sample/releases) with the [dvc update](https://dvc.org/doc/command-reference/update) command. So let's get back to v1.0 of our dataset and push our changes to git:

```bash
dvc update --rev v1.0 data.dvc
git add data.dvc
git commit -m "Get version 1.0 of coco_sample"
git push
```

That's it! We have imported our dataset and we know how to move between different versions so let's create a script that will train our model!


### 3. Train our model

#### 3.1. Mmdetection basics

[MMDetection](https://github.com/open-mmlab/mmdetection) is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project and it is one of the most popular computer vision frameworks. I love it and have been using it and collaborating since a year and a half. Check tweet below to get a glimpse of how big this community grow:

<center>
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">【Today Dispatch】OpenMMLab came to the World Artificial Intelligence Conference 2021 <br><br>We appreciate the trust and support of OpenMMLab users from 109 countries/regions.<br>15+ Research Areas, 160+ Algorithm, 1300+ Checkpoints, help you to easily build your project with OpenMMLab! <a href="https://t.co/HTFLn3vUQp">pic.twitter.com/HTFLn3vUQp</a></p>&mdash; OpenMMLab (@OpenMMLab) <a href="https://twitter.com/OpenMMLab/status/1413049602396147712?ref_src=twsrc%5Etfw">July 8, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
</center>

They have an extense documentation that makes thing really easy for first time users. In this post I will not cover the very basics, my intention is to show how easily can we train a RetinaNet object detector on our coco_sample dataset. First thing we need to do is to find the config file for our model, so let's explore [mmdet model zoo] and more specifically [RetinaNet section](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet). There's a bunch of models there but let's stick with the base config from the [original paper](https://arxiv.org/pdf/1708.02002.pdf), you can find model configuration file under `configs/retinanet_r50_fpn.py` in our source code. We first have the backbone definition, which in our case is RetinaNet50, which weights come from some torchvision checkpoint `init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))`. I check out [official torchvision documentation](https://pytorch.org/vision/stable/models.html) and it seems this network was trained with some dataset that is currently lost so there is no chance to reproduce this results!

<center>
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Just found this checking <a href="https://twitter.com/hashtag/torchvision?src=hash&amp;ref_src=twsrc%5Etfw">#torchvision</a> stable models, it seems they were trained on some volatile dataset 😅<br>cc <a href="https://twitter.com/DVCorg?ref_src=twsrc%5Etfw">@DVCorg</a> <a href="https://t.co/rR8ANSmucI">pic.twitter.com/rR8ANSmucI</a></p>&mdash; Miguel Mendez (@mmeendez8) <a href="https://twitter.com/mmeendez8/status/1418507102465765376?ref_src=twsrc%5Etfw">July 23, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
</center>

We then have the [Feature Pyramid Network (FPN)](https://arxiv.org/abs/1612.03144) configuration, which is a nice way to find features at different scale levels. This net has two different head, the classification head and the regression head. The first of them is a simple convolutional head that outputs the classification scores for each anchor box. The regression head is a convolutional head that outputs the regression values for each anchor box. I cannot really go deep into what anchors are and how this model work but you should check our repo [pyodi](https://github.com/Gradiant/pyodi) if you really want to understand all the details.

