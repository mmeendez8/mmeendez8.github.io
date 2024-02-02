---
layout: post
title: "A demo on sport field registration"
subtitle: "Extract the homography that relates the sport field to the image"
description: ""
image: "/assets/images/fullsize/posts/2024-02-02-amf-pitch-calibration/thumbnail.jpg"
selected: n
mathjax: n
---

I have been working in sports analytics for 2 years now. I am mainly focusing on the computer vision side of things but saying "sports analytics" is a good way to make it sound more interesting. My team goal is simple: extract as much information as possible from sports event video feeds, ensuring the data is high-quality. In order to achieve this, the starting point is to be able to know what are you seeing in the video. This is, to be able to relate the pixels in the video feed to the real world coordinates. That relation is called the homography. I have already written a post in the company blog about this, you if you are interested in why this is so important and all applications that it has, you can check it out [here](https://statsbomb.com/articles/football/creating-better-data-ai-homography-estimation/).

In this post, I decided to challenge myself withto write simple JavaScript code with the assistance of Copilot. So, be kind and withhold judgment on any of the code you see. After all, I'm no JavaScript developer, and I'm essentially in GPT-4's hands here 😎.

## todo:

goals of post are use gpt, be sure i can do this simple thing and I understand the whole process end to end

## The goal

The goal is straightforward: for any given American football NFL event video feed, I want to map the pixels in the video to their corresponding real-world coordinates. Essentially, we want to pinpoint the location of the ball, the players, the goal, etc. The simplest approach to start tackling this problem is to work at the frame level and figure out how to match each image to a predefined pitch template.    

My plan was to develop a web app that allows users to upload an image and then find the correspondence between that image and the pitch template. The pitch template is a basic image of the pitch, including lines and goals.

## The result

Here’s the result for you to explore directly. Make sure to read the following sections for a deeper understanding of how it all works (or just in case you need to understand what to do with these buttons).

{% include homography.html %}


## The details

### Pitch template

The pitch template is a controlled image we use to model the real-world pitch. By creating a template that accurately represents the real-world pitch, we can directly measure distances and angles within that image. This is crucial for extracting meaningful information later, such as ball position, player location, and player speed.

The starting point is to understand how are the dimensions of an American football pitch and [this](https://turftank.com/us/academy/how-big-is-a-football-field/) page is an excellent resource. It's worth noting that we focused solely on NFL dimensions, as NCAA fields differ slightly.

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-02-02-amf-pitch-calibration/pitchdims.png --alt AMF Pitch dimensions example %}
</div>

{:refdef: class="image-caption"}
*NFL Pitch dimensions obtained from [https://turftank.com/us/academy/how-big-is-a-football-field/](https://turftank.com/us/academy/how-big-is-a-football-field/)*
{: refdef}

What I did is to create a simple image which resolution is the same as the NFL pitch, *120 x 53.3*. So as you can imagine by now, one pixel in this image is equivalent to 1 yard in the real world. Then we need to add the endzones, hashmarks, yards numbers and all the elements to that image located at their expected position. I have to reckon that even this is a quite simple and mechanical task, it took me a while until I got to a decent result. Be sure to check the real code but this is a small example where I (and copilot) show how to create the starting pitch with endzones and sidelines:

I created a simple image with the same resolution as the NFL pitch, *120 x 53.3 px*. This means one pixel in the image equals one yard in the real world. Next, I added endzones, hash marks, yard numbers, and all necessary elements, each positioned accurately. I have to reckon that even this is a quite simple and mechanical task, it took me a while until I got to a decent result. Be sure to check the real code but this is a small example where I (and copilot) show how to create the starting pitch with endzones and sidelines:

```javascript
    // Define the pitch dimensions and other constants
    const pitchHeight = 53 + 1/3;
    const pitchWidth = 120;
    const lineWidth = 2; 
    const ctxTemplate = templateCanvas.getContext('2d');

    // Set size and color of template canvas
    ctxTemplate.strokeStyle = 'red';
    ctxTemplate.fillStyle = "black";
    ctxTemplate.fillRect(0, 0, templateCanvas.width, templateCanvas.height);
    
    // Translate to avoid cropping the sidelines
    ctxTemplate.lineWidth = lineWidth;
    ctxTemplate.translate(lineWidth, lineWidth);

    // Side lines
    addLine({ x: 0, y: 0 }, { x: 0, y: pitchHeight });
    addLine({ x: pitchWidth, y: 0 }, { x: pitchWidth, y: pitchHeight });

    // End lines
    addLine({ x: 0, y: 0 }, { x: pitchWidth, y: 0 });
    addLine({ x: 0, y: pitchHeight }, { x: pitchWidth, y: pitchHeight });
```

I repeated this process for all pitch elements, resulting in the image below:

<div class="post-center-image">
    {% picture pimage /assets/images/fullsize/posts/2024-02-02-amf-pitch-calibration/pitchtemplate.png --alt NFL resulting pitch template image %}
</div>

{:refdef: class="image-caption"}
*NFL resulting pitch template image*
{: refdef}

In reality, you see a scaled-up version of a *120 x 53.3 px* image in the web app, adjusted to match your screen size.

### Recovering the homography

The homography is just a mapping that relates the pitch template with the image we upload. This is, that given a point in one of the images, we can find the corresponding point in the other image. There is a lot of cool theory behind this that I do not think I can cover in this post and I also could not improve the work the my colleague Iñaki Rabanillo has done in its own blog. So I will just refer you to his [post](https://iraban.github.io/2021/12/03/homography.html), be sure to check it out since it is a briliant piece of work.

Homography maps the pitch template to the uploaded image, allowing for corresponding points between the two. The theory behind this is extensive and beyond this post's scope. To be fair I can't do better than what my colleague Iñaki Rabanillo has done in its own blog. So I will just refer you to his [post](https://iraban.github.io/2021/12/03/homography.html), be sure to check it out since it is a briliant piece of work.

Summarizing a lot the problem we need to solve, we need to find an homography transformation that is represented by a 3x3 matrix. This will allow us to go from pixel coordinates of real image $p_i$, multiply them by the homography matrix $H$ and obtain the coordinates of point in the template image $p_t$.

$$p_iH=p_t$$

The homography matrix $H$ is a 3x3 matrix that has 8 degrees of freedom. This means that we need at least 4 points in the real image and their corresponding points in the template image to solve for the homography. This is a simple problem to solve and there are a lot of libraries that can do it for you. I used OpenCV library because it is the one I am most familiar with in python and to be honest I do not know if there is any other library that can do it in javascript. 

The work I need to add in the webapp is the following:

1. Enable users to mark points on both images.
2. Display corresponding points in both images using the same color scheme for easy identification.
3. Provide a simple (very simple) method for removing points in case of errors.
4. Include a button to compute the homography using the listed points and the `cv2.findHomography` function.
5. Display the warped template image overlaid on the uploaded image using the calculated homography.

This is the code that retrieves the homography matrix from the list of points and applies it to the template image:

```javascript
    function computeHomography() {
        // Check that both lists of points have the same size and that their size is at least 4
        if (pointsImage.length !== pointsTemplate.length || pointsImage.length < 4) {
            alert('Both lists of points must have the same size and contain at least 4 points');
            return;
        }

        // Convert points to cv.Mat format
        let imagePoints = cv.matFromArray(pointsImage.length * 2, 1, cv.CV_32FC2, pointsImage.flatMap(point => [Math.round(point.x), Math.round(point.y)]));
        let templatePoints = cv.matFromArray(pointsTemplate.length * 2, 1, cv.CV_32FC2, pointsTemplate.flatMap(point => [Math.round(point.x), Math.round(point.y)]));

        // Compute homography
        let homography = cv.findHomography(templatePoints, imagePoints);

        // Check if homography is none because of colinear points
        if (homography.empty())
        {
            alert("Could not found any homography for these sets of points. Be sure they are not colinear.");
            return;
        }

        // Warp the template image using the homography
        let warpedTemplate = new cv.Mat();
        cv.warpPerspective(templateImage, warpedTemplate, homography, sourceImage.size());

        // Add the warped template to the source image
        let resultWeighted = new cv.Mat();
        cv.addWeighted(sourceImage, 1, warpedTemplate, 0.5, 0, resultWeighted);
        cv.imshow('imageCanvas', resultWeighted);

        // Clean up memory
        imagePoints.delete();
        templatePoints.delete();
        warpedTemplate.delete();

        return homography;
    }

```