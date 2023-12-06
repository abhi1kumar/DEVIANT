PYTHON=${PYTHON:-"python"}

FILE=$1
JOB_NAME=$2
GPUS=${GPUS:-1}
PARTITION="VI_OP_1080TI"
GPUS_PER_NODE=4
CPUS_PER_TASK=5


srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    $PYTHON -u $FILE ${@:3}