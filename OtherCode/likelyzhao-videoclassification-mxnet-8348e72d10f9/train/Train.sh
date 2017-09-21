#!/usr/bin/env bash
SIZE=320

LOG=Train_UCF.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python -m train_ucf --network resnet      \
                                  --data-train /workspace/data/UCF-recordio/ucf-frame_train.rec       \
                                  --data-val /workspace/data/UCF-recordio/ucf-frame_val.rec        \
                                  --num-class 101                      \
                                  --num-examples 2359130                 \
                                  --lr 0.1                                   \
                                  --lr-factor 0.1                                   \
                                  --num-layers 101                        \
                                  --batch-size 32                          \
                                  --top-k 5                                   \
                                  --data-nthreads 6                                   \
                                  --test-io  0                               \
                                  --num-epochs 10                             \
                                  --model-prefix model/UCF                             \
                                  --lr-step-epochs 2,5                                   \
                                  --monitor 0                             \
                                  --min-random-scale  0.8                 \
                                  --gpu 0,1,2,3                                   \
                                  >${LOG} 2>&1 &
                                 # --dataset imagenet           \
                                 # --image_set trainall                           \






