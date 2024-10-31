---
layout: post
title:  "Export to Github Cache with Docker Buildx"
subtitle: "We can finally use Docker buildx cache-to gha with build-push action and it is blazingly fast!"
author: Miguel Mendez
description: "Github actions cache is integrated with Docker buildx. Learn how to create a simple pipeline using build-push action and Github Cache. Test the new buildx cache-to exporter!"
image: "/assets/images/fullsize/posts/2021-07-19-new-docker-cache-is-out/thumbnail.jpg"
selected: y
---

I have recently [uploaded a post]({% post_url 2021-04-23-cache-docker %}){:target="_blank"}{:rel="noopener noreferrer"} with some tricks for reducing the time you spend when building Docker images on Github Actions. That did indeed work pretty well for me until now, but it was a naive solution while waiting for [Docker BuildX](https://docs.docker.com/buildx/working-with-buildx/){:target="_blank"}{:rel="noopener noreferrer"} integration with Github cache. The wait is over and we do not need to manually cache files since Docker BuildX will do everything as we expected!

## 1. Get the basics

You can read [my previous post]({% post_url 2021-04-23-cache-docker %}){:target="_blank"}{:rel="noopener noreferrer"} to get the whole picture but I also recommend you to visit the official pull requests that lead to this new feature:

* [This](https://github.com/docker/buildx/pull/535){:target="_blank"}{:rel="noopener noreferrer"} is the buildx code that has been merged for allowing the use of github internal cache
* [This](https://github.com/docker/build-push-action/pull/406#issuecomment-879184394){:target="_blank"}{:rel="noopener noreferrer"} draft PR contains an example of how to use it.

If you read through all of those you probably realize that we have been waiting for new buildx 0.6 version and buildkit 0.9 to be generally available... but that happened just a few days ago!

<center>
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New BuildKit v0.9.0 and Docker Buildx v0.6.0 releases are out with Github cache backend, OpenTelemetry support, Dockerfile Here-docs, better errors, variable support in mount flags etc. <a href="https://t.co/uo89yvSo5j">https://t.co/uo89yvSo5j</a> <a href="https://t.co/L0QM7stmC5">https://t.co/L0QM7stmC5</a></p>&mdash; Tõnis Tiigi (@tonistiigi) <a href="https://twitter.com/tonistiigi/status/1416161830469201920?ref_src=twsrc%5Etfw">July 16, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
</center>

At this moment we are waiting for Github virtual environments to have new buildx 0.6.0 in Ubuntu base images, but they are generated on weekends and deployed during the week so we might have to wait a week or two before that happens. Anyway we can already test the new feature and add it to our pipelines! 

### Edit 9/11/21: 

Github virtual environments have been updated so we can use build-push action without extra configurations.

## 2. Simple example

I updated my CI pipeline to support the new feature. I can now remove all conditionals that I was using before to reduce building time when Dockerfile or conda.yaml were not modified. The simplified pipeline would look like this:

```yaml
name: Continuous Integration new cache

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build_docker:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v1

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v1
        with:
          registry: ghcr.io
          username: mmeendez8
          password: ${{ secrets.MY_TOKEN }}

      - name: Build production image
        uses: docker/build-push-action@v2
        with:
          context: .
          file: Dockerfile
          push: true
          tags: ghcr.io/mmeendez8/cache_docker/ci_dlc:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  lint_and_test:
    needs: build_docker
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/mmeendez8/cache_docker/ci_dlc:latest
      credentials:
        username: mmeendez8
        password: ${{ secrets.MY_TOKEN }}

    steps:
      - uses: actions/checkout@v2
      - name: Lint code
        run: |
          pre-commit install
          pre-commit run
      - name: Run tests
        run: pytest tests
```

Note how `cache-from` and `cache-to` type is set now to `gha` (github action). The first time the action is triggered the cache is empty so Docker will need to build all layers from scratch:

<div class="post-center-image">
{% picture pimage /assets/images/fullsize/posts/2021-07-19-new-docker-cache-is-out/empty_cache.jpg --alt Image showing empty cache  %}
</div>

But after this cache is full, so we can reuse all our layers in next builds, if the images was not modified, or just some of them when we apply changes to our Dockerfile. Let's trigger a new build with an empty commit and check the time it needs now:

```bash
git commit --allow-empty -m "Test build"
git push
```

<div class="post-center-image">
{% picture pimage /assets/images/fullsize/posts/2021-07-19-new-docker-cache-is-out/full_cache.jpg --alt Image showing full cache  %}
</div>

That's it! It only took 22 seconds to build our image.

## Conclusion

* Docker Buildx is a powerful enhancement and we should try to take full advantage of it

* It is very simple to use Github Cache with build-push-action now

*Any ideas for future posts or is there something you would like to comment? Please feel free to reach out via [Twitter](https://twitter.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"} or [Github](https://github.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"}*