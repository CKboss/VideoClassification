#!/usr/bin/env bash
SIZE=320

LOG=FineTune_UCF.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python -m fine-tune --pretrained-model  model/resnext_finetune2         \
                                  --load-epoch 2                                   \
                                  --data-train /workspace/videoclassification-mxnet/preProcess/train_frame_single.lst   \
                                  --data-val /workspace/videoclassification-mxnet/preProcess/val_frame_single.lst        \
                                  --num-class 500                      \
                                  --num-examples 11649002                 \
                                  --lr 0.005                                   \
                                  --lr-factor 0.1                                   \
                                  --batch-size 32                          \
                                  --top-k 5                                   \
                                  --test-io  0                               \
                                  --num-epochs 10                             \
                                  --model-prefix model/resnext_finetune3                             \
                                  --lr-step-epochs 2,5                                   \
                                  --monitor 0                             \
                                  --gpu 0,1,2,3                                   \
                                  >${LOG} 2>&1 &
                                 # --dataset imagenet           \
                                 # --image_set trainall                           \






