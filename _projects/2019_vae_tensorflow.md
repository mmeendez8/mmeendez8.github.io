---
layout: project
title:  "VAEs! Generating images with Tensorflow"
---

In my previous post I covered the theory behind Variational Autoencoders. It’s time now to get our hands dirty and develop some code that can lead us to a better comprehension of this technique. I decided to use Tensorflow since I want to improve my skills with it and adapt to the last changes that are being pushed towards the 2.0 version. Let’s code!

*Note: All code in here can be found on my [Github](https://github.com/mmeendez8/Autoencoder) account*


## Get the data

Tensorflow (with the recently incorporated Keras API) provides a reasonable amount of [image datasets](https://keras.io/datasets/) that we can use to test the performance of our network. It is super simple to import them without loosing time on data preprocessing.

Let’s start with the classics and import the MNIST dataset. For this, I will use another recently added API, the [*tf.dataset*](https://www.tensorflow.org/guide/datasets), which allows you to build complex input pipelines from simple, reusable pieces.

<script src="https://gist.github.com/mmeendez8/8949b080739804b8703feb9aff72bf7d.js"></script>

In here we download the dataset using Keras. After this we create tensors that will be filled with the Numpy vectors of the training set. The map function will convert the image from *uint8* to *floats. *The dataset API allows us to define a batch size and to prefetch this batches, it uses a* *thread and an internal buffer to prefetch elements from the input dataset ahead of the time they are requested. Last three lines are used to create an iterator, so we can move through the different batches of our data and reshape them. Pretty simple!

If this is the first time you use this API I recommend you to test how is this working with this simple example that shows the first image of each batch.

<script src="https://gist.github.com/mmeendez8/6d6adddf91922f7613cfcbb2e7479b4a.js"></script>

Easy isn’t it? Alright, let’s keep moving.

## Encoder

We must now “code our encoder”. Since we are dealing with images, we are going to use some convolutional layers which will help us to maintain the spatial relations between pixels. I got the some of the ideas of how to structure the network from this [great Felix Mohr’s post](https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776)

My intention is to gradually reduce the dimensionality of our input. The images are 28x28 pixels, so if we add a convolutional layer with a stride of 2 and some extra padding too, we can reduce the image dimension to the half ([review some CNN theory here](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)). With two of these layers concatenated the final dimensions will be 7x7 (x64 if have into account the number of filters I am applying). After this I use another convolutional layer with stride 1 and ‘same’ padding which will maintain the vector size through the conv operations.

The **mean** and **standard deviation** of our Gaussian will be computed through two dense layers. We must note that the standard deviation of a Normal distribution is always positive, so I added a softplus function that will take care of this restriction (other works also apply a diagonal constraint on these weights). The epsilon value will be sampled from an unit normal distribution and then we will obtain our *z *value using the *reparametrization trick* ([see my previous post](https://medium.com/@miguelmndez_30551/vaes-i-generating-images-with-tensorflow-f81b2f1c63b0))

<script src="https://gist.github.com/mmeendez8/6d6adddf91922f7613cfcbb2e7479b4a.js"></script>
*Tip: we can see how our network looks using TensorBoard! ([full code on Github](https://github.com/mmeendez8/Autoencoder))*

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*MX_QFCU-sL03uXt_zMaw6Q.png)
{: refdef}

## Decoder

When we speak about decoding images and neural networks, we must have a word in our mind, **transpose convolutions!** They work as an **upsampling method with learning parameters**, so they will be in charge of recovering the original image dimension from the latent variables one. It’s common to apply some non linear transformations using dense layers before the transposed ones. Below we can observe how the decoder is defined.

<script src="https://gist.github.com/mmeendez8/ea9e71ad05d2f79f448b34f1a8b5be6c.js"></script>

Note that *FLAGS.inputs_decoder *references the number of neurons in our dense layers which will determine the size of the image before the transpose convolution layers. In this case I will use the same structure as in the encoder, so the number of neurons will be 49 and the input image size to the first tranposed conv layer will be 7x7. It’s also interesting to point that the last layer uses a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation, **so we force our outputs to be between 0 and 1.**

{:refdef: style="text-align: center;"}
![Sigmoid function plot (obtained from Wikipedia)](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)
{: refdef}

## Cost function

We have defined out encoder and decoder network but we still need to go trough a final step which consists on joining them and defining the cost function that will be optimized during the training phase.

<script src="https://gist.github.com/mmeendez8/8257986911cc4a0cdffb0e6936422898.js"></script>

I first connected the two networks and then (for simplicity sake) I have reshaped input and output batches of images to a flat vector. After this the image loss is computed using the **binary cross entropy** formula, which can be interpreted as the minus log likelihood of our data ([more in here](https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks)). We move on and compute the latent loss using the KL divergence formula (for a Gaussian and a unit Gaussian) and finally, we get the mean of all the image losses.

That’s all! We have created a complete network with its corresponding data input pipeline with just a few lines of code. We can move now, test our network and check if it can properly learn and create new amazing images!

## **Experiments**
<br>
### 1. Are our networks learning?

Let’s start with a simple example and check that the network is working as it should. For this we will use the MNIST and fashion MNIST datasets and see how is the network reconstructing our input images after a few epochs. For this I will set the **number of latent dimensions equal to 5**.

Below, we can see how our network learns to transform the input image in the latent space of 5 dimensions, to recover them then to the original space. Note how as more epochs have passed, we obtain better results and a better loss.

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*e8znvwIOgqXpejItbLIx4A.png)
{: refdef}

{:refdef: style="text-align: center;"}
![Image recovering vs epoch number for MNIST and Fashion MNIST datasets](https://cdn-images-1.medium.com/max/2000/1*RMzRYaDD1sBTWSW1f9zaYg.png)
{: refdef}

{:refdef: style="text-align: center;"}
*Image recovering vs epoch number for MNIST and Fashion MNIST datasets*
{: refdef}

We see how our network keeps improving the quality of the recovered images. It’s interesting to see how at the initial epochs numbers 6, 3, 4 are converted into something that we can considered as ‘similar’ to a 9. But, after a few iterations the original input shape is conserved. For the fashion dataset we get a similar behavior with the bag pictures that are initially recovered as a boot!

It’s also curious, to observe how the images change to get an intuition of what is happening during the learning. But what is going on with the image encodings?

### 2. How does our latent space look?

In the previous section we used a 5 dimensional latent space but we can reduce this number to a two dimensional space which we can plot. In this way the complete images will be encoded in a 2D vector. Using a scatter plot, we can see how this dimensional space evolves with the number of epochs.

In a first moment all images are close to the prior (all point are located around 0). But during training the encoder learns to approximate the posterior distribution, so it will locate the latent variables in different parts of the space having into account their labels (equal label -> close region in the space). Let’s have a look first and then discuss more about this!

{:refdef: style="text-align: center;"}
![MNIST latent space evolution during 20 epochs](https://cdn-images-1.medium.com/max/2000/1*43HzYOZqJ_psdJkKMIgDqw.gif)
{: refdef}

{:refdef: style="text-align: center;"}
*MNIST latent space evolution during 20 epochs*
{: refdef}

Isn’t this cool? Numbers that are similar are placed in a similar region of the space. For example we can see that zeros (red) and ones (orange) are easily recognized and located in a specific region of the space. Nevertheless it seems the network can’t do the same thing with eights (purple) and threes (green). Both of them occupy a very similar region.

Same thing happens with clothes. Similar garments as shirts, t-shirt and even dresses (which can be seen as an intersection between shirts and trousers) will be located in similar regions and the same thing happens with boots, sneakers and sandals!

{:refdef: style="text-align: center;"}
![Fashion MNIST latent space evolution during 20 epochs](https://cdn-images-1.medium.com/max/2000/1*E_2Opsz0ntqv-sN9q4jLjQ.gif)
*Fashion MNIST latent space evolution during 20 epochs*
{: refdef}
{:refdef: style="text-align: center;"}
*Fashion MNIST latent space evolution during 20 epochs*
{: refdef}

These two dimensional plots can also help us to understand the KL divergence term. In my previous post I explained where does it come from and also that acts as a penalization (or regularization term) over the encoder when this one outputs probabilities that are not following an unit norm. Without this term the encoder can use the whole space and place equal labels in very different regions. Imagine for example two images of a number 1. Instead of being close one to each other in the space, the encoder could place them far away. This would result in a problem to generate unseen samples, since the space will be large and it will have ‘holes’ of information, empty areas that do not correspond to any number and are related with noise.

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*R6qo_u2u8zAZIwr3e9cgtw.png)
{: refdef}

{:refdef: style="text-align: center;"}
![Left: Latent space without KL reg — Right: Latent space with KL reg](https://cdn-images-1.medium.com/max/2000/1*26o7FlZBf4rTGrSZ0FGHKQ.png)
{: refdef}

{:refdef: style="text-align: center;"}
*Left: Latent space without KL reg — Right: Latent space with KL reg*
{: refdef}


Look the x and y axis. On the left plot, there is not regularization, so points embrace a much larger region of the space, while as in the right image they are more concentrated, so this produces a dense space.

### 3. Generating samples

We can generate random samples that belong to our latent space. These points have not been used during training (they would correspond with a white space in previous plots). Our network decoder though, has learnt to reconstruct valid images that are related with those points without seem them. So let’s create a grid of points as the following one:

{:refdef: style="text-align: center;"}
![Two dimensional grid of points](https://cdn-images-1.medium.com/max/3722/1*YbXSWp38bADTmyR7rLRybg.png){: width="650px"}
{: refdef}

{:refdef: style="text-align: center;"}
*Two dimensional grid of points*
{: refdef}


Each of this points can be passed to the decoder which will return us a valid images. With a few lines we can check how is our encoder doing through the training and evaluate the quality of the results. Ideally, all labels would be represented in our grid.

<script src="https://gist.github.com/mmeendez8/aa5b73d6d64cf5e1ceba31472a1ebf64.js"></script>

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*dS1wAqJFmB_ESAJ6QasBnw.gif)
{: refdef}

{:refdef: style="text-align: center;"}
![Comparsion between grid space generated images and latent space distribution](https://cdn-images-1.medium.com/max/2000/1*GyWuuCIj8wNqObfYz3-ViQ.gif)
{: refdef}
{:refdef: style="text-align: center;"}
*Comparsion between grid space generated images and latent space distribution*
{: refdef}

What about the fashion dataset? Results are even more fun! Look how the different garment are positioned by ‘similarity’ in the space. Also the grid generated images look super real!


{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*E0QWlZlIls2JTQDpR5XTyw.gif)
{: refdef}

{:refdef: style="text-align: center;"}
![](https://cdn-images-1.medium.com/max/2000/1*Bh2JAS-FoLGNcwuVTsQxIw.gif)
{: refdef}

After 50 epochs of training and using the grid technique and the fashion MNIST dataset we achieve these results:

{:refdef: style="text-align: center;"}
![Fake images generated using mesh grid points](https://cdn-images-1.medium.com/max/2000/1*j-pxK39k7TLYx7n5h_mKdQ.png)*Fake images generated using mesh grid points*
{: refdef}

All this images here are fake. We can finally see how our encoder works and how our latent space has been able to properly encode 2D image representations. Observe how you can start with a sandal and interpolate points until you get a sneaker or even a boot!

## Conclusion

We have learnt about Variational Autoencoders. We started with the theory and main assumptions that lie behind them and finally we implement this network using Google’s Tensorflow.

* We have used the dataset API, reading and transforming the data to train our model using an iterator.

* We have also implemented our encoder and decoder networks in a few lines of code.

* The cost function, made up of two different parts, the log likelihood and the KL divergence term (which acts as a regularization term), should be also clear now.

* The experimental section was designed to support all the facts mentioned above so we could see with real images what is going on under all the computations.

It could be interesting to adapt our network to work with larger and colored images (3 channels) and observe how our network does with a dataset like that. I might implement some of these features but this is all for now!


*Any ideas for future posts or is there something you would like to comment? Please feel free to reach out on [Github](https://github.com/mmeendez8) , [Linkedin](https://www.linkedin.com/in/miguel-mendez/) or [my personal web](https://mmeendez8.github.io/).*