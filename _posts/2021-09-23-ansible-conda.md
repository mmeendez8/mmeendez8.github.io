---
layout: post
title:  "Using Ansible and Python to monitor my servers"
subtitle: "Code your own scripts in Python, deploy them using Ansible & Conda and get real time reports from your server"
author: Miguel Mendez
description: "The best way for monitoring your server is through your own code,"
image: "/assets/images/fullsize/posts/2021-09-23-ansible-conda/thumbnail.jpg"
selected: y
---

I have been in charge of my team's computational servers for about a year. I am far from being an expert in this field and most of the things I know are related to having been using Linux for so many years. I needed to automate some things that were wasting me a lot of time, so I decided that instead of learning and configuring a new monitoring tool, I would use this opportunity to create my own. This may not be the best decision for you, but in my case I had a clear vision of what I needed and the amount of work it would take to do it (and I also wanted to play a little bit with all this stuff).

Anyway, the best advice I can give and the one I learned by heart is: when it comes to a shared infrastructure, better make sure you test everything thoroughly before touching things there or you will end up with a high rate of mail blaming you.

## Why Ansible?

Well, Ansible's moto is: Ansible is Simple IT Automation. That's what I was looking for and after checking their documentations I realized that I wouldn't need much more than 15 minutes to set it up and that's pretty awesome. I read a few posts such [this one](https://mtyurt.net/post/2020/good-bad-parts-of-ansible-after-two-years.html){:target="_blank"}{:rel="noopener noreferrer"} that basically supported all my initial intuitions. I was also influenced by some of my devops friends who used to say good thing about it. Summarizing:

### Advantages

- Open source
- Written in Python!
- YAML based = Low barrier of entry
- Agentless = You don’t need to install any other software on the client systems (this is super)
- Many integrated modules (it hardly takes time to automate classic operations such as user creation, database operations ...)

### Drawbacks

In fact, I still didn't find any drawbacks, although I did seek other people's opinion, but things like lack of user interface or lack of Windows support don't really matter to me.

## My Python monitoring library

My idea was to create a simple python library that could execute commands and communicate with our internal messaging application [Rocket.chat](https://rocket.chat/){:target="_blank"}{:rel="noopener noreferrer"}. In this way, you could receive private messages when something does not work as expected and also automate tasks such as sending welcome messages on user creation, cleaning system caches, updating packages...

Since we use conda in all our servers, the intention was to automatically create a new environment to install there my monitoring application and all its dependencies. I cannot share this application but it is quite a simple thing, I followed a [register pattern](https://charlesreid1.github.io/python-patterns-the-registry.html){:target="_blank"}{:rel="noopener noreferrer"} since I wanted to be able to sequentially add new functionalities that could get automatically registered in my application. I recommend you to use the [catalogue](https://github.com/explosion/catalogue){:target="_blank"}{:rel="noopener noreferrer"} library from explosion guys. The provide a super simple way to register your functions or classes using a decorator. What I did was to use my library main's `__init__.py` file to declare the register:

```python
import catalogue

SERVICES = catalogue.create("my_services_app", "services")

```

And the simply register my 'services' with:

```python
from my_services_app import SERVICES
from my_services_app.services.base import ServerStatus


@SERVICES.register("PingService")
class PingService(BaseService):
    ...
```

That's all, you now would be able to instantiate your `PingService` by:

```python
from my_services_app import SERVICES
service = SERVICES.get("PingService")()
```

So simple and so powerful! Also note that I have created a `BaseService` class, which basically has all the necessary funcionality to send messages to Rocket.chat.

## Conda & Ansible

I created a super simple playbook that helps to clone the latest version of my code from Github, create a new conda environment if does not exists, install my Python monitoring library and also set up some of those services as cron jobs!

But first we need to create a Github Personal Access Token (PAT) to let ansible to login and clone the source code. You can check the official documentation [here](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token){:target="_blank"}{:rel="noopener noreferrer"}. Once we have our token we can encrypt it with ansible add an extra security layer to our deployment. You can simply do something like:

```bash
echo "gh_tokeen: [YOUR-TOKEN]" > gh_token.enc
ansible-vault encrypt gh_token.enc
```

You should now have a `gh_token.enc` encrypted file and Ansible will be in charge of asking you for the password when running the playbook so it can decrypt it. Note that this is very handy since you can place encrypted content under source control and share it more safely.

Let's now create our Ansible playbook to deploy our monitoring library and also deploy one of our services as a cron tass than runs once hourly:

```yaml

---
- name: Create remote services
  hosts: all
  vars:
    conda_path: [Conda installation path]
    conda_env: [Conda environment name]
    repo_path: [Path where clone the repo]

  tasks:
  - name: Clone and update repo
    git:
      repo: "https://{{ gh_token }}@github.com/mmeendez8/myrepo.git"
      dest: "{{ repo_path }}"
      clone: yes
      update: yes
  
  - name: Check conda environment exists
    command:
      cmd: "{{conda_path}}/bin/activate {{conda_env}}"
    register: env_output
    ignore_errors: True

  - name: Install conda environment if not exists
    command: 
      cmd: "{{conda_path}}/bin/conda env create -f {{ repo_path }}/conda.yaml"
    when: env_output.failed == True
  
  - name: Install server_status package
    shell: 
      cmd: "{{conda_path}}/envs/{{conda_env}}/bin setup.py install"
      chdir: "{{repo_path}}"

  - name: Check status
    ansible.builtin.cron:
    name: "My Ping Service"
    special_time: hourly
    job: >
        {{conda_path}}/envs/{{conda_env}}/bin/server_status --service_name=PingService
```

This code is basically plain english! There are a few things we can highlight. First, see how we use `{{ gh_token }}` variable inside the github repository url. We need to tell Ansible that this variable is inside our encrypted file so we should run our playbook with:

```bash
ansible-playbook create_services.yml -e @gh_token.enc --ask-vault-pass
```

Also see how the 'check conda environmet' task registers in a variable the return of activate command so the conda installation task runs only when the environment does not exist.
You can execute this playbook as many times as you want since Ansible is pretty smart and will not create new cron tasks, it will identify this one was already created and update its values if necessary!

## Conclusion

There is a very simple way to create and deploy Python code on your computational servers without spending effort and time in complex configurations. We have learned to:

- Setup Ansible to deploy our library in all our servers

- Use conda to encapsulate our library dependencies

- Add a pattern registry to dinamically add modules to our Python library

- Hide our secrets with Ansible built-in encription

- Set cron jobs that will run automatically

*Any ideas for future posts or is there something you would like to comment? Please feel free to reach out via [Twitter](https://twitter.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"} or [Github](https://github.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"}*
