#!/usr/bin/env bash
SIZE=320

LOG=FineTune_UCF.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python -m fine-tune --pretrained-model model/inceptionresnet        \
                                  --layer-before-fullc flatten_0                      \
                                  --data-train /workspace/data_recordio/UCF-recordio/ucf-frame-inceptionresnet_train.rec       \
                                  --data-val /workspace/data_recordio/UCF-recordio/ucf-frame-inceptionresnet_val.rec        \
                                  --num-class 101                      \
                                  --image-shape '3,331,331'                      \
                                  --num-examples 2359130                 \
                                  --lr 0.01                                   \
                                  --lr-factor 0.1                                   \
                                  --batch-size 16                          \
                                  --top-k 5                                   \
                                  --data-nthreads 6                                   \
                                  --test-io  0                               \
                                  --num-epochs 10                             \
                                  --model-prefix model/UCF_finetune                             \
                                  --lr-step-epochs 2,5                                   \
                                  --monitor 0                             \
                                  --min-random-scale  1                \
                                  --gpu 0                                   \
                                  >${LOG} 2>&1 &
                                 # --dataset imagenet           \
                                 # --image_set trainall                           \






