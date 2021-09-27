
## Kubernetes
https://kubernetes.io/
```bash
# list all running/completed pods
kubectl get pods
# list all deployments
kubectl get deploy
# list all services
kubectl get svc
# list all jobs
kubectl get jobs
# list all pods with label root-ktayal (can be extended to other resources - deploy, svc, jobs, etc.)
kubectl get pods -l app=root-ktayal
# deletes all pods with label root-ktayal (can be extended to other resources - deploy, svc, jobs, etc)
kubectl delete pods -l app=root-ktayal
# show logs for the specified pod name
kubectl log -f <pod_name>
# show previous logs for the pods, in case the pod restarts (useful for debug segfault like errors)
kubectl log -p <pod_name>
# print the yaml of the pod which can be used to check arguments, image, and if the pod restarted will show status of previous pod status (OOM, ERROR, etc)
kubectl get pods <pod_name> -o yaml
# forward local port to the remote pod port
kubectl port-forward <pod_name> <local_port>:<remote_port>
# if the pod is failing to schedule, or stuck on some step other than running, this may print additional details on the pod status - such as reasons for failure to schedule.
kubectl describe pod <pod_name>
```

## AWS
```bash
# similar to hadoop fs for accessing s3 data
aws s3 help
# this will list the data under that path
# (please use this experimental bucket for storing data that you use for experiments eg, s3://experimental-813987666268/ktayal/ has my experimental data)
# (please make sure you create your own USERNAME folder inside experimental-813987666268/)
aws s3 ls s3://experimental-813987666268/
```