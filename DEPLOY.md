# Deployment

The manifests in `k8s/` are cluster-agnostic. What follows is the path used for
AWS: build, push to ECR, apply to EKS.

Nothing here is claimed as running until it has actually been applied to a live
cluster. If you are reading this to check whether the README's deployment claims
are true, the test is `kubectl get hpa` against a real cluster, not the presence
of these files.

## 1. Train, so the serving image has a model

The serving image loads the model from `artifacts/`, and `artifacts/` is
gitignored. Train before building, or the container starts and immediately fails
its readiness probe.

```bash
python main.py
ls artifacts/            # summary.json, preprocessor.joblib, model_*.joblib
```

## 2. Build and test locally

```bash
docker build -f Dockerfile.serve -t salescast-serve:latest .
docker run --rm -p 8000:8000 salescast-serve:latest
curl localhost:8000/health
```

## 3. Push to ECR

```bash
export AWS_REGION=eu-west-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/salescast-serve

aws ecr create-repository --repository-name salescast-serve --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker tag salescast-serve:latest $REPO:latest
docker push $REPO:latest
```

Tag with the git SHA rather than `latest` once more than one person deploys —
`latest` makes rollbacks guesswork.

## 4. Cluster

```bash
eksctl create cluster \
  --name salescast --region $AWS_REGION \
  --nodes 2 --node-type t3.medium --managed
```

The HPA reads pod CPU from the metrics API, which EKS does not install by
default. Without this the HPA reports `<unknown>/70%` and never scales:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

## 5. Apply

```bash
kubectl set image -k k8s/ --dry-run=client   # or edit image: in deployment.yaml
kubectl apply -k k8s/

kubectl rollout status deploy/salescast-api
kubectl get pods,svc,hpa
```

## 6. Confirm autoscaling actually works

Claiming autoscaling without having watched it scale is the same mistake as
claiming a test suite that is an empty file.

```bash
kubectl run load --rm -it --image=busybox --restart=Never -- \
  sh -c 'while true; do wget -q -O- http://salescast-api/health; done'

kubectl get hpa salescast-api --watch
```

Replicas should climb past 2 under load and settle back after the 300s scale-down
window. Screenshot that output for the README — it is the only evidence that
turns the claim into a fact.

## Exposing it

`Service` is `ClusterIP`, so nothing is public yet. Either:

- `kubectl port-forward svc/salescast-api 8000:80` for a private check, or
- add an Ingress with the AWS Load Balancer Controller and terminate TLS at an
  ACM certificate.

## MLflow in a cluster

Training writes to `sqlite:///mlflow.db` by default, which is fine on one
machine and wrong the moment two pods write to it. For a cluster, run a tracking
server backed by RDS Postgres with artifacts in S3, then set
`MLFLOW_TRACKING_URI` on both the training job and the API deployment.