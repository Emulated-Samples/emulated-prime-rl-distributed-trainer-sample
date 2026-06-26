# Prime-RL Kubernetes Deployment Environment

Tests that verify an AI agent can deploy prime-rl to AWS EKS from scratch, including cluster creation, GPU configuration, and end-to-end training verification.

## Overview

This environment tests the complete journey from an empty AWS account to a working prime-rl training setup. The reverse-text example trains for 20 steps to verify the deployment works end-to-end.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       AWS EKS Cluster                           │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   CPU Node      │  │   GPU Node #1   │  │   GPU Node #2   │ │
│  │   (t3.medium)   │  │   (g5.xlarge)   │  │   (g5.xlarge)   │ │
│  │                 │  │                 │  │                 │ │
│  │  orchestrator-0 │  │  inference-0    │  │  trainer-0      │ │
│  │  (CPU only)     │  │  (1x A10G GPU)  │  │  (1x A10G GPU)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                               │                    │           │
│                               └────────┬───────────┘           │
│                                        ▼                       │
│                             ┌─────────────────┐                │
│                             │   EFS Storage   │                │
│                             │   (shared data) │                │
│                             └─────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### Test Phases

| Phase | Duration | Description |
|-------|----------|-------------|
| 1. Prerequisites | ~1 min | AWS credentials, CLI tools, workspace |
| 2. EKS Cluster | ~20 min | Create cluster + CPU node + GPU nodes |
| 3. GPU Operator | ~5 min | Install NVIDIA GPU Operator |
| 4. Storage | ~2 min | EFS filesystem + CSI driver |
| 5. Helm Deploy | ~3 min | Deploy prime-rl via Helm |
| 6. Training | ~10 min | Verify 20 training steps complete |

**Total: ~40-45 minutes**

## Prerequisites

### AWS Service Quotas

**IMPORTANT**: The default AWS account has a vCPU quota of 0 for GPU instances. You must request a quota increase before running.

```bash
# Check current GPU vCPU quota
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --region us-west-2 \
  --query 'Quota.Value'
```

Required quotas:
- **Running On-Demand G and VT instances**: 8+ vCPUs (for 2x g5.xlarge)
- **Running Spot G and VT instances**: 8+ vCPUs (if using spot instances)

### AWS Permissions

The IAM user/role needs permissions for:
- **EKS**: CreateCluster, DeleteCluster, CreateNodegroup, DeleteNodegroup
- **EC2**: RunInstances, TerminateInstances, CreateSecurityGroup
- **EFS**: CreateFileSystem, DeleteFileSystem, CreateMountTarget
- **IAM**: CreateRole, AttachRolePolicy (eksctl creates IAM roles)
- **CloudFormation**: CreateStack, DeleteStack (eksctl uses CF under the hood)

## Quick Start

### 1. Set up credentials

```bash
# Copy the example and fill in your credentials
cp .env.example .env

# Edit .env with your AWS credentials
# AWS_ACCESS_KEY_ID=xxx
# AWS_SECRET_ACCESS_KEY=xxx
# AWS_SESSION_TOKEN=xxx
```

### 2. Build the environment

```bash
cd /hyperfocal/env/environment
npm install
npm run build
```

### 3. Run tests

```bash
# Run all tests
env-orchestrator test

# Or with spot instances disabled (more reliable, more expensive)
USE_SPOT_INSTANCES=false env-orchestrator test
```

### 4. Cleanup

```bash
env-orchestrator cleanup
```

## Configuration

Environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SPOT_INSTANCES` | `true` | Use spot instances for GPU nodes (60-70% cheaper) |
| `CLUSTER_NAME` | `prime-rl-test` | EKS cluster name |
| `AWS_REGION` | `us-west-2` | AWS region |
| `WORKSPACE_PATH` | `/hyperfocal/env/workspace` | Path to prime-rl codebase |

## Manual Commands

### Check Cluster Status

```bash
# Load credentials
source /hyperfocal/env/environment/.env

# Check nodes
kubectl get nodes -o wide

# Check pods
kubectl get pods -l app.kubernetes.io/instance=prime-rl-test

# View trainer logs
kubectl logs prime-rl-test-trainer-0 -f

# View inference logs
kubectl logs prime-rl-test-inference-0 -f

# Check GPU availability
kubectl describe nodes -l nvidia.com/gpu.present=true | grep -A5 'Allocatable:'
```

### Check AWS Resources

```bash
# Load credentials
source /hyperfocal/env/environment/.env

# List EKS clusters
aws eks list-clusters --region us-west-2

# Check nodegroup status
aws eks describe-nodegroup \
  --cluster-name prime-rl-test \
  --nodegroup-name gpu-nodes \
  --region us-west-2 \
  --query 'nodegroup.{status:status,health:health}'

# List CloudFormation stacks
aws cloudformation list-stacks \
  --stack-status-filter CREATE_COMPLETE CREATE_IN_PROGRESS ROLLBACK_COMPLETE \
  --region us-west-2 \
  --query 'StackSummaries[?contains(StackName, `eksctl-prime-rl`)].[StackName,StackStatus]' \
  --output table

# Check EFS filesystems
aws efs describe-file-systems --region us-west-2 \
  --query "FileSystems[?Tags[?Key=='cluster']].{Id:FileSystemId,Name:Name,State:LifeCycleState}"
```

### Manual Cleanup

If `env-orchestrator cleanup` fails, use these commands:

```bash
# Load credentials
source /hyperfocal/env/environment/.env

# 1. Uninstall Helm releases
helm uninstall prime-rl-test 2>/dev/null || true
helm uninstall gpu-operator -n gpu-operator 2>/dev/null || true
helm uninstall aws-efs-csi-driver -n kube-system 2>/dev/null || true

# 2. Delete PVC and StorageClass
kubectl delete pvc prime-rl-test-shared-data --ignore-not-found
kubectl delete storageclass efs-sc --ignore-not-found

# 3. Delete EFS (get ID first)
EFS_ID=$(aws efs describe-file-systems --region us-west-2 \
  --query "FileSystems[?Tags[?Key=='cluster' && Value=='prime-rl-test']].FileSystemId" \
  --output text)

if [ -n "$EFS_ID" ]; then
  # Delete mount targets first
  for MT_ID in $(aws efs describe-mount-targets --file-system-id $EFS_ID \
    --region us-west-2 --query 'MountTargets[*].MountTargetId' --output text); do
    aws efs delete-mount-target --mount-target-id $MT_ID --region us-west-2
  done
  sleep 30
  aws efs delete-file-system --file-system-id $EFS_ID --region us-west-2
fi

# 4. Delete EKS cluster
eksctl delete cluster --name prime-rl-test --region us-west-2

# 5. If eksctl fails, delete CloudFormation stacks manually
# First disable termination protection, then delete
for STACK in $(aws cloudformation list-stacks --region us-west-2 \
  --query "StackSummaries[?contains(StackName, 'eksctl-prime-rl-test')].StackName" \
  --output text); do
  aws cloudformation update-termination-protection \
    --no-enable-termination-protection --stack-name $STACK --region us-west-2
  aws cloudformation delete-stack --stack-name $STACK --region us-west-2
done
```

## Known Issues & Troubleshooting

### Issue 1: VcpuLimitExceeded

**Error:**
```
VcpuLimitExceeded - You have requested more vCPU capacity than your current vCPU limit of 0 allows
```

**Cause:** AWS accounts have a default quota of 0 vCPUs for GPU instance types (G and VT families).

**Solution:**
1. Go to AWS Service Quotas console
2. Search for "Running On-Demand G and VT instances"
3. Request increase to at least 8 vCPUs (for 2x g5.xlarge)
4. Wait for approval (usually 1-24 hours)

```bash
# Check current quota
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --region us-west-2
```

### Issue 2: MaxSpotInstanceCountExceeded

**Error:**
```
MaxSpotInstanceCountExceeded - You have exceeded your maximum limit of Spot Instances
```

**Cause:** Spot instance quota is separate from on-demand quota.

**Solution:** Either:
1. Request spot instance quota increase, OR
2. Use on-demand instances instead:
   ```bash
   USE_SPOT_INSTANCES=false env-orchestrator test
   ```

### Issue 3: eksctl Timeout During GPU Nodegroup Creation

**Error:**
```
waiting for CloudFormation stack "eksctl-prime-rl-test-nodegroup-gpu-nodes": ResourceNotReady
```

**Cause:** GPU nodegroup creation can take 25-35 minutes, exceeding eksctl's internal timeout.

**Solution:** The environment code now handles this by:
1. Creating the GPU nodegroup via eksctl
2. Polling AWS directly for nodegroup status (bypassing eksctl's waiter)
3. Using a 35-minute timeout for GPU nodes specifically

If stuck, check actual status:
```bash
aws eks describe-nodegroup \
  --cluster-name prime-rl-test \
  --nodegroup-name gpu-nodes \
  --region us-west-2 \
  --query 'nodegroup.status'
```

### Issue 4: ExpiredTokenException

**Error:**
```
ExpiredTokenException: The security token included in the request is expired
```

**Cause:** AWS session credentials (from `.env`) have expired during a long-running operation.

**Solution:**
1. Refresh credentials in `.env` file
2. For sandbox accounts, credentials typically expire after 1 hour
3. Re-run the tests

### Issue 5: CloudFormation Stack in ROLLBACK_COMPLETE

**Cause:** A previous nodegroup creation failed and CloudFormation rolled back.

**Solution:**
```bash
# Delete the failed stack (after disabling termination protection)
aws cloudformation update-termination-protection \
  --no-enable-termination-protection \
  --stack-name eksctl-prime-rl-test-nodegroup-gpu-nodes \
  --region us-west-2

aws cloudformation delete-stack \
  --stack-name eksctl-prime-rl-test-nodegroup-gpu-nodes \
  --region us-west-2
```

## FAQ

### Q: Why split EKS cluster creation into two steps?

**A:** The GPU nodegroup creation can take 25-35 minutes, which exceeds eksctl's internal 25-minute timeout. By creating the cluster + CPU nodegroup first (fast, ~15 min), then adding the GPU nodegroup separately with direct AWS polling, we avoid timeout issues and get better error visibility.

### Q: Why use StatefulSets instead of Deployments?

**A:** Prime-rl's distributed training requires predictable pod names for coordination. StatefulSets provide:
- Sequential names: `trainer-0`, `trainer-1`, etc.
- Stable network identities
- Ordered deployment/scaling

Deployments use random suffixes (e.g., `trainer-7d4b5f8c9-x2k9j`), making it impossible to know "I am rank 2 of 3".

### Q: Why EFS instead of EBS?

**A:** Prime-rl needs `ReadWriteMany` (RWX) access mode - multiple pods reading/writing to the same storage simultaneously. EBS only supports `ReadWriteOnce` (RWO). EFS provides:
- Multi-AZ access
- Automatic scaling
- Concurrent access from all pods

### Q: How much does this cost?

**A:** Approximate costs (us-west-2):

| Component | On-Demand/hr | Spot/hr |
|-----------|-------------|---------|
| EKS control plane | $0.10 | $0.10 |
| t3.medium (CPU) | $0.04 | $0.01 |
| g5.xlarge x2 (GPU) | $2.02 | $0.70 |
| **Total** | **~$2.16/hr** | **~$0.81/hr** |

Full test run (~45 min): $1.60 on-demand, $0.60 with spot

### Q: Why does the inference server take so long to start?

**A:** vLLM needs to:
1. Download the model (~2GB for small models)
2. Load model weights into GPU memory
3. Compile CUDA kernels
4. Start the HTTP server

This typically takes 3-5 minutes on first run.

### Q: Can I run this in a different region?

**A:** Yes, set `AWS_REGION` in your `.env` file. However:
- GPU availability varies by region
- us-west-2 and us-east-1 typically have the best GPU capacity
- Check g5.xlarge availability in your target region first

## File Structure

```
environment/
├── README.md              # This file
├── .env                   # AWS credentials (not committed)
├── .env.example           # Example credentials template
├── package.json           # Node.js dependencies
├── problems.yaml          # Problem definitions
├── tsconfig.json          # TypeScript config
└── src/
    ├── index.ts           # Environment class (main entry point)
    ├── config.ts          # Configuration constants
    ├── helpers.ts         # Utility functions
    └── tests/
        ├── index.ts       # Test exports
        ├── infrastructure.ts  # Phases 1-3 (Prerequisites, EKS, GPU)
        └── deployment.ts      # Phases 4-6 (Storage, Helm, Training)
```

## Development

### Building

```bash
npm run build
```

### Type Checking

```bash
npx tsc --noEmit
```

### Adding New Tests

1. Add tests to appropriate file in `src/tests/`
2. Export from `src/tests/index.ts`
3. Tests run sequentially - order matters

## License

Internal use only.
