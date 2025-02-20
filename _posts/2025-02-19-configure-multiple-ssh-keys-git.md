---
layout: post
title: "Managing Multiple GitHub Accounts with SSH"
subtitle: "Easily switch between GitHub accounts using different SSH keys"
author: Miguel Mendez
description: "Learn how to manage multiple GitHub accounts on the same computer using SSH keys and Git configuration. This step-by-step guide covers SSH key setup, Git configuration, and automating repository access, making it easy to switch between work and personal accounts without conflicts."
image: "/assets/images/fullsize/posts/2025-02-19-configure-multiple-ssh-keys-git/thumbnail.jpg"
selected: y
mathjax: y
tags: [GitHub, SSH, Git, Git-Configuration, Multiple-GitHub-Accounts, SSH-Keys, Version-Control]  
categories: [Software Development, DevOps, Git, Programming, Productivity, Tech Guides]  
---

Each time I change my laptop, I spend at least an hour configuring permissions for my GitHub accounts. Most people probably don’t have this problem, but I need a separate account for work and another for my personal projects. So this is a quick guide that I hope will at least help me in the future to do this without spending so much time searching for how to do it.

This solution assumes the following file structure:

```bash
.
├── work/
│   ├── work-repo1
│   ├── work-repo2
│   └── ...
└── personal/
    ├── personal-repo1
    ├── personal-repo2
    └── ...
```

Where `work` and `personal` are the folders assigned to each account. When I want to work with a specific account I need to clone the repository in the corresponding folder.

## Step 1: Generate SSH keys

First, you need to [generate SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent){:target="_blank"}{:rel="noopener noreferrer"} for each account. You can do this by running the following command:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

When you run this command, you will be prompted to enter a file in which to save the key. You can specify a different file for each account. For example, you could save the key for your work account as `id_rsa_work` and the key for your personal account as `id_rsa_personal`.

## Step 2: Add SSH keys to the SSH agent

Next, you need to add the SSH keys to the SSH agent. You can do this by running the following commands:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa_work
ssh-add ~/.ssh/id_rsa_personal
```

## Step 3: Add SSH keys to your Github accounts

Finally, you need to add the SSH keys to your Github accounts. You can do this by copying the contents of the public key file (e.g., `id_rsa_work.pub` and `id_rsa_personal.pub`) and pasting them into the SSH keys section of your Github account settings.

**Note**: If you belong to an organization, you may need to configure SSO and authorize the key. This can be done from the key settings page.

## Step 4: Configuring SSH

Now we need to configure SSH to use the correct key for each repository. For this we are going to first modify the `~/.ssh/config` file to include the following:

```bash
Host github.com
  AddKeysToAgent yes
  IdentitiesOnly yes  
```

The `IdentityOnly` option is important because it tells SSH to only use the key specified in the configuration file.

## Step 5: Configuring Git

Now we need to configure git to use the correct key for each repository. For this we are going to first modify the `~/.gitconfig` file to include the following:

```bash
[includeIf "gitdir:~/personal/"]
  path = ~/.gituser-personal
[includeIf "gitdir:~/work/"]
  path = ~/.gituser-work
```

This just tells git to include the corresponding file when we are in the `personal` or `work` folder. Now let's create the files `~/.gituser-personal` and `~/.gituser-work` with the following content:

```bash
# ~/.gituser-personal
[user]
    email = personal@mail.com # Replace with you personal email
    name = Miguel Mendez

[core]
    sshCommand = ssh -i ~/.ssh/id_rsa_personal
```

```bash
# ~/.gituser-work
[user]
    email = work@mail.com  # Replace with your work email
    name = Miguel Mendez

[core]
    sshCommand = ssh -i ~/.ssh/id_rsa_work
```

This will tell git to use the correct key for each repository. 

## Step 6: Clone the repository

You are almost all set. There is just a hacky part that we are forced to do with this solution. All the previous git configurations will only work if we are under a git repository. So we need to basically run `git init` in both `work` and `personal` folders. This is a bit annoying but it is the only way I found to make this work.

```bash
cd ~/work
git init

cd ~/personal
git init
```

That's all! You can now clone your repositories in the corresponding folder and git will use the correct key for each one.