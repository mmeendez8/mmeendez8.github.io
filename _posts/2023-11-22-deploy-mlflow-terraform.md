---
layout: post
title: "MLFlow Deployment on AWS with Terraform"
subtitle: "A Simple Guide to MLFlow Deployment on Kubernetes"
author: Miguel Mendez
description: "Follow this step-by-step guide on deploying MLFlow in AWS using Terraform. Learn how to effectively manage your machine learning lifecycle, set up a Postgres database, create a secure S3 bucket, and customize a MLFlow Docker image. Improve you Machine Learning experiment tracking and model management in the cloud."
image: "/assets/images/fullsize/posts/2023-11-22-deploy-mlflow-terraform/thumbnail.jpg"
selected: y
mathjax: n
tags: [MLFlow, AWS, Terraform, Machine Learning, DevOps]
categories: [Machine Learning, DevOps, Cloud]
---

Training and deploying machine learning models is a complex process. There are lots of steps involved – think data prep, model training, evaluating how good your model is, and then deploying it. Especially when it comes to training and evaluation, it's super important to have a tool that makes life easier, something that lets us compare different experiments and track their performance (like losses, hyperparameters, metrics, etc.).

There are a bunch of tools out there for this, but we’re going to focus on one: [MLFlow](https://mlflow.org/){:target="_blank"}{:rel="noopener noreferrer"}. The goal of this post is to learn how to set up MLFlow in AWS, and we're going to use Terraform for that. [Terraform](https://www.terraform.io/){:target="_blank"}{:rel="noopener noreferrer"} is awesome because (disclaimer: I hate its syntax) it lets you define all your infrastructure as code so you will never ever forget what button did you click to set up that AWS instance. 


## Why MLFlow?

MLFlow is an open-source platform designed for managing the entire machine learning lifecycle. It's great for tracking experiments, managing models, and even deploying them into production. Personally, I have primarily used MLFlow for tracking experiments. I became quite familiar with it in my previous job and really appreciated its features, especially the ability to create custom visualizations and share experiment links with colleagues. These links can also be embedded in reports, which is super handy.

I really missed using it and was eager to set it up in my current job. But, as often happens, more pressing tasks always seemed to take priority. To be fair, for basic tracking needs, you can get by with TensorBoard. It's a straightforward tool that provides all the essentials. However, it starts to feel a bit overwhelming when you're juggling a large number of experiments. Also, I did not have much experience with Terraform, just added a few lines here and there to existing configurations to set up some permissions and such. So, I decided to kill two birds with one stone and get into this project.

## What do we need?

Before we dive in, let's make sure we have got everything we need. We are going to use Terraform for setting up our infrastructure, so you'll need that installed. This post assumes you've got Terraform ready and configured for your AWS account, so we'll skip that part and focus on what we need to add to our Terraform configuration. If you haven't set up Terraform yet, no problem – just follow [this guide](https://learn.hashicorp.com/tutorials/terraform/aws-build?in=terraform/aws-get-started){:target="_blank"}{:rel="noopener noreferrer"} for the setup. Also, I have deployed the MLFlow server in AWS EKS using [Flux](https://fluxcd.io/){:target="_blank"}{:rel="noopener noreferrer"}, a great tool for deploying services in Kubernetes. It's not essential for this tutorial, though. Feel free to deploy the MLFlow server manually or with any other tool that you're comfortable with.

Now, the MLFlow tracking server needs a few key components:

- A **database** to store all your data
- A **storage bucket** for artifacts, like model checkpoints.
- A **server** to run the tracking server, which is essentially a Docker image with MLFlow installed.

Let’s take it step by step and see how we can set up each of these components.


## Database

MLFlow supports a bunch of different databases, but we are going to use Postgres – it's popular, and I'm quite familiar with it.We will use AWS RDS to set up our database and its security group. Here's what to add to your Terraform configuration:

First, we create a random password without special characters:

```hcl
# Creates a random password for the database
resource "random_password" "mlflow_db" {
  length  = 16
  special = false
}
```

Next, we set up a security group for the database that allows TCP traffic on port 5432 (typical for PostgreSQL databases). We are limiting incoming traffic to only our VPN. If you are not looking to restrict traffic, you can skip this step.

```hcl
# This specifies the security group for the database
module "mlflow_db_security_group" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 4"

  name        = "mlflow-db-sg"
  description = "RDS Aurora ingress security group"
  vpc_id      = aws_vpc.main.id # You need to use you own vpc id here

  ingress_with_cidr_blocks = [
    {
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      description = "RDS Aurora access from within VPC"
      cidr_blocks = [var.your_vpn]
    },
  ]
}
```

Finally, we create a RDS Aurora cluster for the MLFlow database. The setup is straightforward, but you can always refer to the [official documentation](https://registry.terraform.io/modules/terraform-aws-modules/rds-aurora/aws/latest){:target="_blank"}{:rel="noopener noreferrer"} for more details. Notice how we use the earlier created password and security group to restrict access. We are creating a single writer instance, assuming limited traffic. If you expect more, consider adding a read replica in the instances map.

```hcl
# Database configuration
module "mlflow_cluster_db" {
  source         = "terraform-aws-modules/rds-aurora/aws"
  version        = "6.2.0"

  name           = "mlflow-db"
  engine         = "aurora-postgresql"
  engine_version = "14.5"
  instance_class = "db.r5.large"
  instances = {
    one = {}
  }

  database_name          = "mlflow"
  master_username        = "mlflow"
  create_random_password = false
  master_password        = random_password.mlflow_db.result

  create_security_group  = false
  subnets                = local.subnets_ids_database
  vpc_security_group_ids = [module.mlflow_db_security_group.security_group_id]

  storage_encrypted   = true
  apply_immediately   = true
  monitoring_interval = 10

  enabled_cloudwatch_logs_exports = ["postgresql"]

  tags = {
    Environment = "dev"
    Terraform   = "true"
  }
}
```

That is about it for the database setup. Lastly, we need to store the DB user, password, and endpoint for later connection. I will use AWS Secrets Manager for this. How you manage secrets may vary; the only thing you really want to avoid is to store them in source code. You could also use AWS Parameter Store or Hashicorp Vault, for example. Here's how to store the secrets in AWS Secrets Manager:

```hcl
resource "aws_secretsmanager_secret" "mlflow_db_master_username_id" {
  name  = "mlflow-username"
}

resource "aws_secretsmanager_secret_version" "mlflow_db_master_username" {
  secret_id     = aws_secretsmanager_secret.mlflow_db_master_username_id[0].id
  secret_string = module.mlflow_cluster_db.cluster_master_username
}

resource "aws_secretsmanager_secret" "mlflow_db_master_password_id" {
  name  = "mlflow-password"
}

resource "aws_secretsmanager_secret_version" "mlflow_db_master_password" {
  secret_id     = aws_secretsmanager_secret.mlflow_db_master_password_id[0].id
  secret_string = module.mlflow_cluster_db.cluster_master_password
}

resource "aws_secretsmanager_secret" "mlflow_db_endpoint_id" {
  name  = "mlflow-db-writer-endpoint"
}

resource "aws_secretsmanager_secret_version" "mlflow_db_endpoint" {
  secret_id     = aws_secretsmanager_secret.mlflow_db_endpoint_id[0].id
  secret_string = module.mlflow_cluster_db.cluster_endpoint
}
```

## Storage bucket and IAM Role

Next up is setting up a storage bucket for all the artifacts, and we're going to use AWS S3 for this. Here's the Terraform configuration needed:

```hcl
resource "aws_s3_bucket" "mlflow_artifacts_bucket" {
  bucket = "mlflow-artifacts-bucket"
}

resource "aws_s3_bucket_ownership_controls" "mlflow_bucket_ownership" {
  bucket = aws_s3_bucket.mlflow_artifacts_bucket[0].id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "mlflow_bucket_acl" {
  depends_on = [aws_s3_bucket_ownership_controls.mlflow_bucket_ownership]

  bucket = aws_s3_bucket.mlflow_artifacts_bucket[0].id
  acl    = "private"
}
```

This configuration creates an S3 bucket named 'mlflow-artifacts-bucket'. It also sets up ownership controls, ensuring new objects uploaded without an ACL are owned by the bucket owner. Additionally, it enforces a private ACL for the bucket, securing the stored data.

Now, MLFlow needs to access this bucket from EKS, so we'll create an IAM Role for Service Accounts. Here’s how to do it:

```hcl
resource "aws_iam_role" "mlflow-role" {
  name = "mlflow-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com" 
        },
      },
    ],
  })
}

# Custom Policy for Specific S3 Bucket Access
resource "aws_iam_policy" "mlflow_s3_policy" {
  name   = "mlflow_s3_policy"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:GetObject*",
          "s3:ListBucket",
          "s3:PutObject",
          "s3:DeleteObject"
        ],
        Effect   = "Allow",
        Resource = [
          "arn:aws:s3:::mlflow-artifacts-bucket",
          "arn:aws:s3:::mlflow-artifacts-bucket/*"
        ]
      }
    ]
  })
}

# Attach Custom Policy to the Role
resource "aws_iam_role_policy_attachment" "s3_policy_attachment" {
  role       = aws_iam_role.mlflow-role.name
  policy_arn = aws_iam_policy.mlflow_s3_policy.arn
  depends_on = [aws_iam_policy.mlflow_s3_policy]
}
```

This code sets up a new role, `mlflow-role`, and attaches a custom policy, `mlflow_s3_policy`, to it. This policy grants specific permissions to access the S3 bucket we created earlier.


## Docker

Before we proceed, it's important to note that the official MLFlow Docker image doesn't include the necessary libraries for connecting with AWS S3 and Postgres. To address this, we'll need to create a custom image that includes these libraries. This is done by crafting a Dockerfile like the one below:

```dockerfile
FROM ghcr.io/mlflow/mlflow:v2.7.1

RUN apt-get -y update && \
    apt-get -y install python3-dev build-essential pkg-config && \
    pip install --upgrade pip && \
    pip install psycopg2-binary boto3

CMD ["bash"]
```

This Dockerfile starts with the official MLFlow image and adds the required libraries. After crafting this file, you can build the image and push it to your ECR repository, or any other repository you prefer to use.

## Deployment

Finally, we'll set up the server to run the MLflow tracking server on AWS EKS. We typically use Flux for deploying services in Kubernetes, so we'll create a `kustomization.yaml` file containing all the necessary resources. Alternatively, you could also deploy using Terraform, although I am less familiar with this method as we primarily utilize Flux for our deployments.

We will begin with defining secrets. Our secrets are stored in AWS Secrets Manager and accessed from Kubernetes. We use the [External Secrets](https://github.com/external-secrets/kubernetes-external-secrets){:target="_blank"}{:rel="noopener noreferrer"} package for simplicity. Here is what our `secrets.yaml` file looks like:

```yaml
apiVersion: "kubernetes-client.io/v1"
kind: ExternalSecret
metadata:
  name: mlflow-secrets
  namespace: monitoring
spec:
  backendType: secretsManager
  data:
    - key:  mlflow-username
      name: AWS_SECRET_MLFLOW_USERNAME
    - key: mlflow-password
      name: AWS_SECRET_MLFLOW_PASSWORD
    - key: mlflow-db-writer-endpoint
      name: AWS_SECRET_MLFLOW_HOST
```

Next, we need a service account for our deployment, defined in `service-account.yaml`:


```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mlflow-sa
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::941819254007:role/mlflow-role
```

This configuration uses the earlier created IAM role for accessing the S3 bucket.

We also need a service for our deployment, specified in `service.yaml` file:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mlflow
spec:
  ports:
    - port: 8080
      name: http
      targetPort: http
  selector:
    app.kubernetes.io/name: mlflow
```

This is a very simple service definition that just exposes port 8080 and targets all pods with the label `app.kubernetes.io/name: mlflow`.

Next, we need an ingress to expose our service to the outside world. We use AWS ALB Ingress Controller for this. Here's what our `ingress.yaml` file looks like:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlflow
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internal
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/ssl-redirect: "443"
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS": 443}]'
    alb.ingress.kubernetes.io/healthcheck-path: /

spec:
  rules:
  - host: host-url
    http:
      paths:
        - path: /
          pathType: Prefix
          backend:
            service:
              name: mlflow
              port:
                name: http
```

This Ingress configuration sets up access for our MLFlow tracking server. It is very simple, the configuration ensures SSL redirection to HTTPS on port 443 and that is is only accessible from within the VPC (internal). The rule specified routes traffic for the `host-url` to the MLflow service on the HTTP port. 

Lastly, the deployment itself, outlined in `deployment.yaml` file:


```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: mlflow
  replicas: 1
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      serviceAccount: mlflow-sa
      containers:
      - name: app
        image: add-your-image-here
        command:
        - "mlflow"
        - "server"
        - "--host=0.0.0.0"
        - "--backend-store-uri=postgresql://$(AWS_SECRET_MLFLOW_USERNAME):$(AWS_SECRET_MLFLOW_PASSWORD)@$(AWS_SECRET_MLFLOW_HOST):5432/mlflow"
        - "--default-artifact-root=s3://mlflow-artifacts-bucket"
        - "--port=8080"
        ports:
        - name: http
          containerPort: 8080
        envFrom:
        - secretRef:
            name: mlflow-secrets
        readinessProbe:
          httpGet:
            path: /
            port: 8080
        startupProbe:
          initialDelaySeconds: 10
          httpGet:
            path: /
            port: 8080
        resources:
          requests:
            memory: 1G
            cpu: 500m
          limits:
            memory: 1G
            cpu: "1"
```

Key takeaways from this setup:

1. We deploy a single replica of the MLFlow pod.
2. Replace `add-your-image-here` with the image we created earlier. 
3. Secrets created earlier are passed as environment variables in the `envFrom` section.
4. Entrypoint is set to `mlflow server` and we pass the required arguments to connect to the database and the S3 bucket.
5. Readiness and startup probes ensure the pod is fully operational before receiving requests.

And that's it! With these steps, our MLFlow server is ready to go and can start tracking experiments.

## Conclusion

In this post we have seen how to set up MLFlow in AWS using Terraform. We have seen how to set up a Postgres database, an S3 bucket, and a Kubernetes deployment for the MLFlow server. We have also seen how to create a custom Docker image that includes the required libraries to connect to the database and the S3 bucket.



*Any ideas for future posts or is there something you would like to comment? Please feel free to reach out via [Twitter](https://twitter.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"} or [Github](https://github.com/mmeendez8){:target="_blank"}{:rel="noopener noreferrer"}*


