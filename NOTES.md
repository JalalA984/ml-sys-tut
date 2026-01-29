# Conda Env

## I run
conda env export --no-builds > environment.yml

## Anyone else run
conda env create -f environment.yml

# Kubernetes and k3d

## Create cluster
k3d cluster create distml --image rancher/k3s:v1.25.3-rc3-k3s1
kubectl config get-contexts

## Create namespace (idk i feel like not needed)
kubectl create ns basics
kubectl config set-context --current --namespace=basics
kubectl config set-context --current --namespace=default


## Create test pod
kubectl create -f hello_world.yaml

## Detailed pod
kubectl get pod whalesay -o json