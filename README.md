# Running Megatron-DeepSpeed with Slurm

Original **Megaton-DeepSpeed** is implemented in:
- Megatron-DeepSpeed from Microsoft: https://github.com/microsoft/Megatron-DeepSpeed
- Megatron-DeepSpeed from BigScience: https://github.com/bigscience-workshop/Megatron-DeepSpeed

where sample slurm scripts in this repository are referring. 

## 3D Parallelism Enabled
- Data Parallelism(DP)
    - DeepSpeed ZeRO-DP stage 1
- Pipeline Parallelism(PP)
    - DeepSpeed pipeline parallelism
- Tensor Parallelism(TP)
    - Megatron-LM tensor slicing

When ZeRO-DP is combined with PP and TP, it typically enables ZeRO stage 1. Though it's techinically possible to use ZeRO-DP stage 2 with PP(optionally TP), it would cause performance degradation due to additional reduce-scatter collective communication for every micro-batch to aggregate the gradients before sharding. The same reason is applied to the case of ZeRO-DP stage 3. That's why I guess ZeRO-DP stage 2 or 3 is **NOT** allowed to use PP by default in the original implementation repos.

## Prerequisite
1. Prepare for a docker container with **DeepSpeed** installed. The simplest form of `Dockerfile` would be like:
```bash
FROM nvcr.io/nvidia/pytorch:22.02-py3
RUN apt-get update
RUN pip install deepspeed
```

2. Build the docker container from the `Dockerfile` with your own tag name.
```bash
docker build --tag $tagname .
```
3. Clone **Megatron-DeepSpeed** implmentation repo
```bash
git clone https://github.com/microsoft/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed
```

4. Prepare for sample dataset
```bash
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz
python tools/preprocess_data.py \
    --input oscar-1GB.jsonl \
    --output-prefix my-gpt2 \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```
There might be no dependency errors if you try going through this step inside the PyTorch Container with DeepSpeed installed(from Prerequisite step 1-2).

## Run Megatron-DeepSpeed with Slurm
**Slurm scheduler** is used to dispatch jobs to the GPU computing cluster. **Note that** the value of variable `CONTAINER_IMAGE` in the slurm scripts should be modified to the tag name of your own container where **DeepSpeed** is properly installed(see Prerequisite step 1-2). In addition, most of the configuration parameters in the scripts are hard-coded just for simplicity. You can modify them according to your preference(e.g., the size of TP/PP, hidden states, batchsize, DeepSpeed configs, etc).

#### Megatron-DeepSpeed on a **single node**
The default number of GPUs is set to be 8(i.e., 8 GPUs) in the script by the variable `N_GPUS`. you can modify the value of this variable if you need. Execute the script with `sbatch` command.
```bash
sbatch megatron_ds_snmg.slurm
```
The value of the variable `N_GPUS` shouldn't be over the number of physical GPU devices in a single node.

#### Megatron-DeepSpeed on **multi-nodes**
The default number of nodes is set to be 2 with 8 GPUs for each. **the number of nodes** can be modified using the `sbatch` argument when executing. If you try using 4 nodes(32 GPUs) for training,
```bash
sbatch --nodes 4 megatron_ds_mnmg.slurm
```
