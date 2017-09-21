#!/usr/bin/env bash
SIZE=320

LOG=LOG_MM_Train.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

export MXNET_CPU_WORKER_NTHREADS=6

nohup python -m train_ucf_list --network resnet      \
                                  --data-train /workspace/videoclassification-mxnet/preProcess/train_frame.lst       \
                                  --data-val /workspace/videoclassification-mxnet/preProcess/val_frame.lst        \
                                  --num-class 500                      \
                                  --num-examples 23218263                 \
                                  --lr 0.1                                   \
                                  --lr-factor 0.1                                   \
                                  --num-layers 101                        \
                                  --batch-size 32                          \
                                  --top-k 5                                   \
                                  --data-nthreads 10                                   \
                                  --test-io  0                               \
                                  --num-epochs 10                             \
                                  --model-prefix model/MM                             \
                                  --lr-step-epochs 2,5                                   \
                                  --monitor 0                             \
                                  --min-random-scale  0.8                 \
                                  --gpu 0,1,2,3                                   \
                                  >${LOG} 2>&1 &
                                 # --dataset imagenet           \
                                 # --image_set trainall                           \






