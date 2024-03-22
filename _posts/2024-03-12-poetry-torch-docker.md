---
layout: post
title: "Building an Efficient Docker Image with Poetry and PyTorch"
subtitle: "A guide for your Deep Learning environment setup"
description: "Learn to create a Docker image for your Pytorch projects. Discover how to manage dependencies with Poetry and Python 3.11. We'll walk you through using Docker Buildx, handling Torch versions, and optimizing your build. Ideal for developers ready to quickly start their deep learning projects."
image: "/assets/images/fullsize/posts/2024-03-12-poetry-torch-docker/thumbnail.jpg"
selected: y
mathjax: n
---

The goal of this post is straightforward: to guide you through the creation of a Docker image equipped with Poetry for dependency management and Torch for running deep learning models, specifically utilizing Python 3.11. While the task may seem simple at first glance, it involves several tricks that I believe can be very useful to share. This guide will utilize Docker Buildx, a powerful feature that might be unfamiliar to some, yet it is enabled by default in newer Docker releases.

## Docker, Ubuntu, and Python 3.11

I was surprised by the scarcity of information available on creating a Docker image with Python 3.11 as the system's default Python version. While one could use the official Python images, I personally prefer to have complete control over and understanding of my Docker images. This practice helps avoid unnecessary dependencies and can be particularly beneficial when addressing issues as unknown environment variables or extra packages that may be set in the base images.

Note that for building our images we will use Docker Buildx, which is a Docker CLI plugin that extends the capabilities of the Docker CLI. There are two different ways of building images with Docker Buildx: using the `docker buildx build` command or setting up the `DOCKER_BUILDKIT` environment variable.

```bash	
# Using the docker buildx build command
docker buildx build -t my-image:latest -f my-file .

# Setting up the DOCKER_BUILDKIT environment variable
export DOCKER_BUILDKIT=1
docker build -t my-image:latest -f my-file .
```

Let's begin by configuring our Dockerfile:

```Dockerfile
FROM ubuntu:22.04

# Set non-interactive mode to avoid prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Install system tools and libraries.
# Utilize --mount flag of Docker Buildx to cache downloaded packages, avoiding repeated downloads
RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    apt-get update && \ 
    apt-get install -y software-properties-common && \
    # Add the Deadsnakes PPA for Python 3.11
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        python3.11 \
        python3-pip \
        python3.11-venv \
        python3.11-dev && \
    # Clean up to keep the image size small
    rm -rf /var/lib/apt/lists/*  && \
    # Set Python 3.11 as the default Python version
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --set python /usr/bin/python3.11

ENTRYPOINT [ "/bin/bash" ]
```

This Dockerfile is relatively standard. We are utilizing Ubuntu 22.04 as our base image and installing Python 3.11 from the Deadsnakes PPA. Moreover, we're setting Python 3.11 as our default Python version using update-alternatives.

For those curious about the `--mount` flag, it's a Docker Buildx feature that caches downloaded packages, preventing them from being downloaded again when adding new packages. This feature can significantly reduce the time required to build your images.


## Poetry

Next, let's install Poetry using the official installer, which I like because it's simple and straightforward. What I usually do is prevent Poetry from creating a new virtual environment. Instead, I manage the environment manually, giving me greater control over my project's configuration.

If you're curious about the need for a virtual environment in our Docker image, there are a couple of reasons. Firstly, isolating your project's dependencies from the system Python ensures a clean, conflict-free environment. Secondly, you might consider using a multi-stage build. This means first installing everything needed in an initial stage and then copying the final virtual environment to the second clean stage. This would make our image smaller and faster to build. However, this is not the focus of this post and I will not cover it here.

```Dockerfile
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VENV_PATH="/app/.venv" # Use custom venv, avoid auto-creation by Poetry

# Install system tools and libraries.
# Utilize --mount flag of Docker Buildx to cache downloaded packages, avoiding repeated downloads
RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    apt-get update && \ 
    apt-get install -y software-properties-common && \
    # Add the Deadsnakes PPA for Python 3.11
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        python3.11 \
        python3-pip \
        python3.11-venv \
        python3.11-dev && \
    # Clean up to keep the image size small
    rm -rf /var/lib/apt/lists/*  && \
    # Set Python 3.11 as the default Python version
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --set python /usr/bin/python3.11

# Set PATH to include Poetry and custom venv
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION

# Create and prepare the virtual environment
RUN python -m venv $VENV_PATH && \
    python -m pip install --upgrade pip && \
    pip cache purge && rm -Rf /root/.cache/pip/http
    
WORKDIR /app

ENTRYPOINT [ "/bin/bash" ]
```

I've set `VENV_PATH` to `{project-dir}/.venv`. This is because Poetry might not follow the environment variable we set earlier, as mentioned in the [official docs](https://python-poetry.org/docs/configuration/#virtualenvscreate){:target="_blank"}{:rel="noopener noreferrer"}, and instead creates its own environment. Everything else in the setup is quite standard. If you have any questions, please feel free to ask.

## Pytorch

Installing Torch with Poetry can be tricky because Torch can be installed with or without GPU support, making it challenging to support both CPU and GPU versions in your pyproject.toml file. For example, you might use the CPU version for Continuous Integration (CI) and the GPU version for running your models in production. Many issues related to this are discussed on the [Poetry GitHub](https://github.com/python-poetry/poetry/issues/6409){:target="_blank"}{:rel="noopener noreferrer"}. After trying different methods, my preferred solution is to install dependencies using Poetry and then install Torch using pip. We need to ensure Torch is installed inside our virtual environment (venv), which requires setting the correct paths in advance (apologies if you were expecting a more complex solution 😑). Here’s what you need to add:


```Dockerfile
# Copy dependency files to the app directory
COPY poetry.lock pyproject.toml /app

# Install dependencies with Poetry and Torch with pip, caching downloaded packages
RUN --mount=type=cache,target=/root/.cache \
    poetry install --without dev && \
    # Install torch GPU
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html 

# Copy the entire project code to the container
COPY ./ /source/
```

Note that we first copy the `poetry.lock` and `pyproject.toml` to the image and then we run `poetry install --without dev`. Only after deps have been installed we copy our code to the image. This is a good practice to avoid installing the dependencies every time we change our code. Once again we are using the `--mount` flag to cache the downloaded packages.

Although we install Torch with GPU support, we do not install the CUDA toolkit separately. This is because all necessary CUDA binaries are included in the Torch wheel; hence, we specify `cu118` in the installation command to ensure compatibility. This is what makes the torch wheel huge, because it includes code for multiple CUDA architectures so the same binary can be used on different GPUs. If you want to obtain a smaller image, you can build torch from source and only specify the architecture you need for your GPU. [This](https://github.com/pytorch/pytorch/issues/17621){:target="_blank"}{:rel="noopener noreferrer"} is a good thread about this topic.


## Final solution

Here's the final Dockerfile that puts everything we've talked about into action, hope this can help you to build your own image.

```Dockerfile
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VENV_PATH="/app/.venv" # Use custom venv, avoid auto-creation by Poetry

# Install system tools and libraries.
# Utilize --mount flag of Docker Buildx to cache downloaded packages, avoiding repeated downloads
RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    apt-get update && \ 
    apt-get install -y software-properties-common && \
    # Add the Deadsnakes PPA for Python 3.11
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        python3.11 \
        python3-pip \
        python3.11-venv \
        python3.11-dev && \
    # Clean up to keep the image size small
    rm -rf /var/lib/apt/lists/*  && \
    # Set Python 3.11 as the default Python version
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --set python /usr/bin/python3.11

# Set PATH to include Poetry and custom venv
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION

# Create and prepare the virtual environment
RUN python -m venv $VENV_PATH && \
    python -m pip install --upgrade pip && \
    pip cache purge && rm -Rf /root/.cache/pip/http
    
WORKDIR /app

# Copy dependency files to the app directory
COPY poetry.lock pyproject.toml /app

# Install dependencies with Poetry and Torch with pip, caching downloaded packages
RUN --mount=type=cache,target=/root/.cache \
    poetry install --without dev && \
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html 

# Copy the entire project code to the container
COPY ./ /source/

ENTRYPOINT [ "/bin/bash" ]
```


*Any ideas for future posts or is there something you would like to comment? Please feel free to reach out via [Twitter](https://twitter.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"} or [Github](https://github.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"}*
