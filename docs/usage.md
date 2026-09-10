# Usage

## Submit a job

```yaml
# config/jobs/example.yaml
name: troiani-small-pretrain
gpus: 1
min_gpu_memory_gb: 40
priority: opportunistic
preemptible: true
checkpoint_interval: 600
command:
  - python
  - -m
  - troiani_platform.training.dummy_train
```

```bash
troiani-platform job submit config/jobs/example.yaml
```

The child process receives `RUN_ID`, `JOB_ID`, `CHECKPOINT_DIR`, `PLATFORM_ENDPOINT`, and `GPU_ASSIGNMENT`. Training code does not need to be rewritten; a thin checkpoint client is enough.

## Preemption

A researcher process on an assigned GPU, `troiani-platform job preempt`, or `release-all` sends `PREEMPT_REQUEST`. The worker checkpoints, validates, releases the devices, and requeues the job.

## Tracking

Each run records config, git SHA, dataset fingerprint, environment, metrics, and checkpoint lineage. Inspect with:

```bash
troiani-platform run inspect RUN_ID
troiani-platform run lineage RUN_ID
troiani-platform reproduce RUN_ID
```

Dataset identity is a versioned manifest (`dataset_name`, `dataset_version`, `dataset_hash`, `shard`) — not a copy of the data.

## Health

NaN / Inf / stall / throughput collapse pause the job after an emergency checkpoint and emit an event. Automatic rollback is opt-in in policy.
