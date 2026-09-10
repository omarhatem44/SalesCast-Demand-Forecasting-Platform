# Deployment

Two paths. **kind** runs a real Kubernetes cluster on your own machine and costs
nothing — it is enough to prove the manifests work and that the HPA scales. **EKS** is
the same manifests on AWS, and costs money.

Nothing here is claimed as running until it has been applied to a live cluster. The
test is `kubectl get hpa` returning real numbers, not the presence of these files.

---

## Path A — local cluster with kind

### 1. Prerequisites

Docker Desktop, with **at least 4 CPUs** allocated (Settings → Resources). The HPA is
allowed to scale to 10 pods at 200m CPU each, which needs 2 CPUs of requests on top of
the load generator; with the default 2 CPUs the extra pods sit `Pending` and the demo
looks like a failure when it is only a resource limit.

```powershell
choco install kind kubernetes-cli
kind create cluster --name salescast
kubectl cluster-info
```

### 2. Train, so the serving image has a model

The serving image loads from `artifacts/`, which is gitignored. Train before building,
or the container starts and fails its readiness probe.

```powershell
python main.py
dir artifacts                  # summary.json, preprocessor.joblib, model_*.joblib
```

### 3. Build and load the image

kind nodes have their own image store and cannot see your local Docker images, so the
image has to be loaded in explicitly. `imagePullPolicy: IfNotPresent` in the Deployment
is what stops Kubernetes trying to pull it from a registry instead.

```powershell
docker build -f Dockerfile.serve -t salescast-serve:latest .
kind load docker-image salescast-serve:latest --name salescast
```

### 4. Metrics server

The HPA reads pod CPU from the metrics API, which no fresh cluster installs. Without
this the HPA reports `<unknown>/70%` forever. On kind it also needs
`--kubelet-insecure-tls`, because kubelet serving certificates are self-signed.

```powershell
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

kubectl patch -n kube-system deploy metrics-server --type=json `
  -p '[{\"op\":\"add\",\"path\":\"/spec/template/spec/containers/0/args/-\",\"value\":\"--kubelet-insecure-tls\"}]'

kubectl -n kube-system rollout status deploy/metrics-server
kubectl top nodes                     # must return numbers before continuing
```

### 5. Apply

```powershell
kubectl apply -k k8s/
kubectl rollout status deploy/salescast-api
kubectl get pods,svc,hpa
```

`kubectl get hpa` should show a real percentage rather than `<unknown>`. If it shows
`<unknown>`, metrics-server is not ready — go back to step 4.

### 6. Check the service works before load-testing it

```powershell
kubectl port-forward svc/salescast-api 8000:80
```

In a second terminal:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST -H "Content-Type: application/json" `
  --data "@forecast-payload.json" http://localhost:8000/forecast
```

A 400 from `/forecast` means the history is too short. The window builder needs
`lookback + horizon` rows *after* the rolling features drop their leading NaNs — about
40 rows in practice. `forecast-payload.json` holds 60, which is comfortably clear.

### 7. Drive load and watch it scale

```powershell
kubectl create configmap forecast-payload --from-file=payload.json=forecast-payload.json
kubectl apply -f k8s/loadtest.yaml

kubectl get hpa salescast-api --watch
```

The load generator posts to `/forecast`, not `/health`, deliberately: `/health` returns
a constant and burns no CPU, so hammering it would never move the HPA off its floor.
`/forecast` runs preprocessing, feature engineering, window building and inference —
the work the service actually does.

Replicas should climb past 2 within a minute or two. Stop the load and they settle back
after the 300-second scale-down window:

```powershell
kubectl delete -f k8s/loadtest.yaml
kubectl delete configmap forecast-payload
```

**Screenshot the `kubectl get hpa --watch` output showing TARGETS above 70% and
REPLICAS climbing.** That image is the only thing that turns "autoscaling" from a claim
into a fact, and it belongs in the README.

### 8. Tear down

```powershell
kind delete cluster --name salescast
```

---

## Path B — AWS EKS

Same manifests, real infrastructure, real cost. Delete the cluster when you are done.

### Push to ECR

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

Tag with the git SHA rather than `latest` once more than one person deploys — `latest`
makes rollbacks guesswork.

### Cluster

```bash
eksctl create cluster \
  --name salescast --region $AWS_REGION \
  --nodes 2 --node-type t3.medium --managed

kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

On EKS metrics-server needs no TLS patch — that step is a kind quirk.

### Apply

Set `image:` in `k8s/deployment.yaml` to the ECR URI and change `imagePullPolicy` to
`Always`, then:

```bash
kubectl apply -k k8s/
kubectl rollout status deploy/salescast-api
kubectl get hpa salescast-api --watch
```

### Exposing it

`Service` is `ClusterIP`, so nothing is public. Either port-forward for a private
check, or add an Ingress with the AWS Load Balancer Controller and terminate TLS at an
ACM certificate.

### Tear down

```bash
eksctl delete cluster --name salescast --region $AWS_REGION
```

---

## MLflow in a cluster

Training writes to `sqlite:///mlflow.db` by default, which is fine on one machine and
wrong the moment two pods write to it. For a cluster, run a tracking server backed by
RDS Postgres with artifacts in S3, then set `MLFLOW_TRACKING_URI` on both the training
job and the API deployment.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| HPA shows `<unknown>/70%` | metrics-server not installed, or not ready, or missing `--kubelet-insecure-tls` on kind |
| Pods stuck `ImagePullBackOff` on kind | image not loaded with `kind load docker-image` |
| Pods stuck `Pending` when scaling | node has no CPU left — raise Docker Desktop's CPU allocation or lower `maxReplicas` |
| Readiness probe failing | `artifacts/` was empty at build time; run `python main.py` and rebuild |
| `/forecast` returns 400 | history too short — needs roughly 40+ rows |
| Replicas never rise under load | load is hitting `/health`; post to `/forecast` instead |