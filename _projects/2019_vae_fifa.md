---
layout: project
title:  "Generating FIFA 19 players with VAEs and Tensorflow"
subtitle: "Learn to generate your own footbal players"
---


This is my third post dealing with Variational Autoencoders. If you want to catch up with the math I recommend you to check my first post. If you prefer to skip that part and go directly to some simple experiments with VAEs then move to my second post, where I showed how useful these networks can be. If you just want to see how a neural network can create fake faces of football players then you are in the right place! Just keep reading!

I must admit that I would like to dedicate this post…

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/0*4xBerr9GTrKG0k75.jpg)
    {: refdef}

*Note: All code in here can be found on my [Github](https://github.com/mmeendez8) account*

### 1. Introduction

I spent a few hours thinking about which dataset could I use to apply all the knowledge and lines of code I acquire learning about VAEs.

I had some constraints that limit my work but the main one were the limited availability of resources and computational power. The dataset I was looking for, must had small images which will allow me to get some results in a rational amount of time. Also I would like to deal with colored images, since [my previous model](https://towardsdatascience.com/vaes-generating-images-with-tensorflow-61de08e82f1f) was designed for black and white images and I wanted to evolve it to deal with more complex images.

Finally I found a dataset that fitted all my requirements and at the same time was interesting and funny for me, so I could not resist.

### 2. Data Collection

The dataset I found was upload to Kaggle (more [here](https://www.kaggle.com/karangadiya/fifa19)). It consists on detailed attributes for every player registered in the latest edition of FIFA 19 database. I downloaded the csv file and open a Jupyter notebook to have a look at it.

{:refdef: style="text-align: center;"}
![Sample from csv file](https://cdn-images-1.medium.com/max/2000/1*qqMNpYth27mUJz4ulyJQpA.png)
*Sample from csv file*
{: refdef}

This file contains URLs to the images of each football player. I started by downloading a couple of images to see if links were working well and to check if image sizes were constant. I got good news, links were fine and images seem to be PNG files of 48x48 pixels with 4 different channels (Red, Green, Blue, Alpha).

After this I coded a Python script that could download as fast as possible this dataset. For this **I used threads to avoid our CPU to be idle** waiting for the IO tasks. You can find the script on [Github](https://github.com/mmeendez8/Fifa/blob/master/downloader.py) or **implement it by yourself**.

I was able to collect a **total of 15216 different images** since some of the URLs in the csv file were not valid.


### 3. Data Processing

Yeah! I was able to download our images in a fast way but… when I plotted these images I got the following results:

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*a79H-6aOrWS2OitftEnjGw.png) 
![Downloaded images](https://cdn-images-1.medium.com/max/2000/1*rcHUjgC8uJLEefIBOz0QXA.png)
{: refdef}

{:refdef: style="text-align: center;"}
*Downloaded images*
{: refdef}

I deduced that someone had applied some preprocessing technique to the edges between the player and the background. So I had to revert this process.

One of the constraints I had in mind was that I had to be able of solving this problem using Tensorflow and nothing else (after all I am trying to improve my skills with it). Alright, so I must implement something that is relatively easy and works like a charm…

In the Jupyter notebook you can see how I elaborated this method. I basically used the alpha channel (as a boolean mask) and I convert it to a binary image using a certain threshold. After this step I filtered all the pixels in the image that were not present in the mask. This can be easily done in a few lines of code and results are really great! I encourage you to check the notebook since is the easiest way to understand it.

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*RfMAnXBo3fs-tn2fXV71lQ.png)![](https://cdn-images-1.medium.com/max/2000/1*BGKp7AMpM6cX31IAVfUYIQ.png)![](https://cdn-images-1.medium.com/max/2000/1*JQhWpY-MHWSte1fk3PgHUw.png)
{: refdef}


In Tensorflow you can create a simple function that takes a RGBA image as input and returns the reconstructed one (without the alpha channel).

<script src="https://gist.github.com/mmeendez8/874ea37859b7b93e22ba95dd787335b2.js" charset="utf-8"></script>

In here I convert the alpha channel into a 48x48 boolean matrix. After this I convert the matrix to *uint8 *and I add a third dimension to the data so I can apply a wise multiplication with all the channels of the original image. This multiplication will set to zero all those pixels that have a zero value in the mask and I return only the RGB channels. Isn’t it easy?

## 3. Data Pipe

Now it’s time to create a pipeline for our data. We need to:

* Read bytes from image files

* Decode those bytes into tensors

* Remove noise

* Convert to tf.float32

* Shuffle and prefetch data

Tensorflow’s Dataset API allow us to do all these steps in a very simple way.

<script src="https://gist.github.com/mmeendez8/44712f11376486c2d8feb8c4c63b5493.js" charset="utf-8"></script>

### 4. Define the network

In my [previous post](https://towardsdatascience.com/vaes-generating-images-with-tensorflow-61de08e82f1f) I show how to define the Autoencoder network and its cost function. Nevertheless, I was not happy with that implementation since the definition of the network was in the main file, very far away from what I would consider a modular code.

So now, I have created a new class which will allow me to create the network in a very simple and intuitive manner. I must thank to [this amazing post](https://danijar.com/structuring-your-tensorflow-models/) from [Danijar Hafner](https://danijar.com/) where I found a way to combine decorators with Tensorflow’s Graph intuition. I won’t explain this code in here, otherwise the post would be too long but feel free to ask below about anything you want.

The main advantage of using this decorator is that ensures that all nodes of the model are only created once in our Tensorflow graph (and we save a lot of lines of code).

<script src="https://gist.github.com/mmeendez8/257c469341d1fe3b920a4bc15b601981.js" charset="utf-8"></script>

### 5. Put the pieces together

The program will read data from a list of filenames which I locate into a placeholder, since I want to be able to feed it with new data in a near future (after training) to evaluate its performance.

<script src="https://gist.github.com/mmeendez8/3b6bb54507b77cde55bf84031168795f.js" charset="utf-8"></script>

As you can see above, the network class constructor receives the *input batch* tensor, an iterator that moves through our bank of images. Also, *filenames *variable* *corresponds with the previously mentioned placeholder. It has a default value returned by the *get_files *function which simply lists all filenames that are contained in the directory *data_path.*

How does this look in TensorBoard? Pretty cool!

{:refdef: style="text-align: center;"}
![Graph structure (TensorBoard)](https://cdn-images-1.medium.com/max/2000/1*9C_-T1qQ9ej5w58pjgFBbw.png)
{: refdef}
{:refdef: style="text-align: center;"}
*Graph structure (TensorBoard)*
{: refdef}

The save block, it’s created to save the network state (weights, biases) and the graph structure (see [GitHub](https://github.com/mmeendez8/Fifa) for more)

### 5. Experiments

The training phase is computationally expensive and I am working on my personal laptop so I could not get very far with the number of epochs. I encourage the readers who have more resources to try new experiments with longer workouts and to share their results.

#### 2D Latent Space

One of the coolest things with VAEs, is that if you reduce the latent space to just two dimensions, you will be able to deal with data in a 2 dimensional space. So you move your data from a 48x48x3 dimensional space to just 2! This allows you to get visual insights about how is the model distributing the data around the space. I did this for a total of 500 epochs and I force the net to return both input images and the reconstructed ones, in order to see, if the network was able to reconstruct the original inputs.

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*BfqHoDNPUvbjjXRSgNotcw.png)
![](https://cdn-images-1.medium.com/max/2000/1*tjKY6ko7p5_r-BNWAy8A1A.png)
![Reconstruction at epochs 2, 150 and 500 (left to right)](https://cdn-images-1.medium.com/max/2000/1*wq2N6TeCk-rpowsSwzt0Lg.png)
{: refdef}

{:refdef: style="text-align: center;"}
*Reconstruction at epochs 2, 150 and 500 (left to right)*
{: refdef}


As we can see, at epoch 2, the reconstructed images are almost all the same. It seems the networks learns the pattern of a ‘general’ face and it simply modifies its skin color. At epoch 150 we can see greater differences, the reconstruction is better but still far from reality, though we start to see different hair styles. At epoch 500, the reconstruction did not succeed, but it learn to differentiate the most general aspects of each face and also some expressions.

Why are these results so poor compared with our previous work with MNIST datasets? Well, our model is too simple. We are not able to encode all the information in a 2 dimensional space, so a lot of it is being lost.

But we still can have a look to our **2 dimensional grid space **and force the network to create some fake players for us! Let’s have a look below:
{:refdef: style="text-align: center;"}
![Football players face distribution in two dimensional space](https://cdn-images-1.medium.com/max/2000/1*MqjnuvwdBL0hc7z0gUDJGA.png)*Football players face distribution in two dimensional space*
{: refdef}


This is what we got! Interesting result… but we already knew this was going to happen isn’t it?. The network is too simple, so this ‘standard faces’ that we see are made of similar football players profiles. This means that our encoder is doing a good job, placing similar faces in close regions in the space BUT the decoder is not able to reconstruct a 48x48x3 image from just 2 points (which is, in fact, a super hard task). This is why the decoder is always outputting similar countenances.

What could we do? Well, we can increase the complexity of our model and add some extra dimensions to our latent space!

#### 15D Latent Space

Let’s repeat the training with a total 15 dimensions in our latent space and check how our face reconstructions are evolving with the training. For this I have trained the network during 1000 epochs.

I discovered [Floydhub](https://www.floydhub.com) a Deep Learning platform that allows me to use their GPUs for free (during 2 hours). This fact, allowed me to perform this long training phase. I really recommend to try Floydhub, since it is really easy to deploy your code and run your training there.

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*91bs-t90abbEHQwVLXAd3A.png)
![](https://cdn-images-1.medium.com/max/2000/1*T9WpZl_dJg4CDJqvzJ3Baw.png)
![](https://cdn-images-1.medium.com/max/2000/1*9lHNkYvZTQJu6b8pjy5Paw.png)
![Reconstruction at epochs 150, 500, 750, 1000 (left to right and up down)](https://cdn-images-1.medium.com/max/2000/1*oYVwzzxmZQWHVAixfMfKeQ.png)
{: refdef}

{:refdef: style="text-align: center;"}
*Reconstruction at epochs 150, 500, 750, 1000 (left to right and up down)*
{: refdef}

It seems our network is doing a better job now! The reconstructions have improved and they really look like the original input. Let’s note the jersey colors, the haircuts and the face positions! But is this good enough? Can you guess the original input from these reconstructions?

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*Xp_3-VkhdN2bqeFrZ-Y3jA.png)
{: refdef}

Well, it’s pretty hard to imagine the original input for these reconstructed faces… From left to right these are Messi, Ronaldo, Neymar, De Gea and De Bruyne. A** longer training** or another **complexity increasement** would produce better results so I encourage you to get my code on Github and give it a try!

#### Interpolate points

We know that our encoder will place each image in a specific point of our 15 dimensional space. So if get two images we will have two different points in that latent space. We can apply a linear interpolation between them, in order to extract other points.

{:refdef: style="text-align: center;"}
![Modric — Kante transformation](https://cdn-images-1.medium.com/max/3358/1*5zx0GtXBe9h67AO_ZFCw0w.png)*Modric — Kante transformation*
{: refdef}

This is the result when we get Modric and Kante’s latent vector and we interpolate a total of 5 points between them. We can observe how moving around our latent space make us obtain different facial features. You can try as many combinations as you want!

{:refdef: style="text-align: center;"}
![Marcelo — Ramos transformation](https://cdn-images-1.medium.com/max/3358/1*HmM42jVImDW2OiFoex2Axg.png)*Marcelo — Ramos transformation*
{: refdef}

#### Average player by country

If you are used to work with machine learning, you have heard about unsupervised learning and the importance of centroids on this field. A centroid can be defined as the mean point of all those that belong to the same cluster or set.

Our data is categorized by skills, football team, age, nationality… So imagine the following, we could get all the players for a certain country, let’s say Spain, and obtain the latent vector that our encoder produces for each one of them. After this we can compute an average of these latent vector and encode it, in a way, that we are obtaining an image of a fake player that has the most common facial features for a certain country!! How cool sounds that?

You can observe the most common attributes for each country collected in a single image.

{:refdef: style="text-align: center;"}
![Centroid fake player for each country](https://cdn-images-1.medium.com/max/3350/1*81LDkhE9VaOZ31H3n-_FHA.png)*Centroid fake player for each country*
{: refdef}

### Conclusion

In this post we have learnt how to apply VAEs to a real dataset of colored images. I have shown how to preprocess the data and how to create our network in a structured way. The experiments I have collected are just a reduced set of the huge amount of possibilities that one can obtain after dealing with image embeddings (our latent space vectors).

I would like to encourage you to try the code, create new experiments, use different dimensional spaces or longer trainings!

*Any ideas for future posts or is there something you would like to comment? Please feel free to reach out on [Github](https://github.com/mmeendez8) , [Linkedin](https://www.linkedin.com/in/miguel-mendez/) or [my personal web](https://mmeendez8.github.io/).*
