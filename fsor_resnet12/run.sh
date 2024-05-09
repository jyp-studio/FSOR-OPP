#! /bin/bash

# MiniImageNet 5-way 1-shot SOTA 
python train.py --dataset miniImageNet --logroot logs/  --data_root ../data \
                --n_ways 5  --n_shots 1 \
                --restype ResNet12 \
                --pretrained_model_path models/miniImageNet_pre.pth \
                --featype OpenMeta \
                --learning_rate 0.001 \
                --tunefeat 0.0001 \
                --tune_part 4 \
                --cosine \
                --base_seman_calib 1 \
                --train_weight_base 1 \
                --neg_gen_type semang \
                --agg mlp \
                --gpus 2 \
                --n_train_para 2 \
                --n_train_runs 400

# MiniImageNet 5-way 5-shot
# python train.py --dataset miniImageNet --logroot logs/  --data_root .. \
#                 --n_ways 5  --n_shots 5 \
#                 --pretrained_model_path models/miniImageNet_pre.pth \
#                 --featype OpenMeta \
#                 --learning_rate 0.001 \
#                 --tunefeat 0.0001 \
#                 --tune_part 4 \
#                 --cosine \
#                 --base_seman_calib 1 \
#                 --train_weight_base 1 \
#                 --neg_gen_type semang \
#                 --agg mlp \
#                 --gpus 0 \
#                 --n_train_para 1 \
#                 --n_train_runs 800 \
#                 --epochs 130

# TieredImageNet 5-way 1-shot
# python train.py --dataset CIFAR-FS --logroot logs/  --data_root .. \
#                 --n_ways 5  --n_shots 1 \
#                 --pretrained_model_path models/fc100_resnet12.pth \
#                 --featype OpenMeta \
#                 --learning_rate 0.0001 \
#                 --tunefeat 0.0001 \
#                 --tune_part 4 \
#                 --cosine \
#                 --base_seman_calib 1 \
#                 --train_weight_base 1 \
#                 --neg_gen_type semang \
#                 --agg mlp \
#                 --gpus 0 \
#                 --n_train_para 1 \
#                 --n_train_runs 800 \
#                 --epochs 5000 \
#                 --inference_steps 2 \
#                 --funit 10.0
