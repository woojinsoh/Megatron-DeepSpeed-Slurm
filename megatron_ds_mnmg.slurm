#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --output=%x-%j.out

set -e -x

WORKSPACE=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
CONTAINER_IMAGE=pytorch:22.02-py3-deepspeed
NNODES=${SLURM_NNODES}

if [ -z ${NNODES} ]
then
	echo "Number of nodes should be passed"
	exit 1
else
	if [ ${NNODES} -lt 2 ]
	then
		echo "The number of requested nodes should be > 1"
		exit 1
	else
		echo "${NNODES} nodes for distributed training are requested"
	fi
fi



# Paths
DATASET=${WORKSPACE}/my-gpt2_text_document
VOCAB_PATH=${WORKSPACE}/gpt2-vocab.json
MERGE_PATH=${WORKSPACE}/gpt2-merges.txt
CONFIG_JSON=${WORKSPACE}/ds_config.json

# Enable DeepSpeed
USE_DEEPSPEED=1
ZERO_STAGE=1

# Debug model
TP=4
PP=2
LAYERS=8
HIDDEN=512
SEQ=1024
GLOBAL_BATCH=128
MICRO_BATCH=4

# 52B
#TP=4
#PP=16
#HIDDEN=8192
#LAYERS=64
#SEQ=1024
#GLOBAL_BATCH=1024
#MICRO_BATCH=4


#Megatron Options
options=" \
	--tensor-model-parallel-size ${TP} \
	--pipeline-model-parallel-size ${PP} \
        --num-layers ${LAYERS} \
        --hidden-size ${HIDDEN} \
        --num-attention-heads 32 \
        --seq-length ${SEQ} \
        --loss-scale 12 \
        --max-position-embeddings ${SEQ} \
	--micro-batch-size ${MICRO_BATCH} \
	--global-batch-size ${GLOBAL_BATCH} \
	--train-iters 1000 \
        --lr 6.0e-5 \
	--min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 40 \
        --eval-interval 1000 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB_PATH} \
	--merge-file ${MERGE_PATH} \
	--save-interval 1000 \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --fp16 \
	--checkpoint-activations \
	--distributed-backend nccl \
        "

# DeepSpeed Options
if [[ ${USE_DEEPSPEED} -eq 1 ]]; then
	echo "Using DeepSpeed"
	options="${options} \
		--deepspeed \
		--deepspeed-activation-checkpointing \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
		"
fi

# DeepSpeed Configs
cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : ${GLOBAL_BATCH},
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH},
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": ${ZERO_STAGE}
  },

  "gradient_clipping": 1.0,
  "prescale_gradients": false,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

CMD="${WORKSPACE}/pretrain_gpt.py ${options}"
echo $CMD


# Distributed args
GPUS_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_STEP_NODELIST | head -n 1)
MASTER_PORT=$((${SLURM_JOB_ID} % 16384 + 49152))

# Touch launcher
LAUNCHER="torchrun \
	--nproc_per_node ${GPUS_PER_NODE} \
	--nnodes ${NNODES} \
 	--rdzv_id=${SLURM_JOB_ID} \
	--rdzv_backend=c10d \
	--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
	"

# Execution command
srun -l \
	--container-image ${CONTAINER_IMAGE} \
	--container-mounts ${WORKSPACE} \
	bash -c "${LAUNCHER} ${CMD}" \

