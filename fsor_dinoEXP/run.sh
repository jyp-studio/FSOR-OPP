#! /bin/bash

python train.py --dataset miniImageNet \
                --logroot logs/  \
                --data_root ./data/ \
                --n_ways 5  --n_shots 1 \
                --restype ResNet12 \
                --pretrained_model_path models/miniImageNet_pre.pth \
                --task FSOR \
                --learning_rate 0.001 \
                --tunefeat 0.0001 \
                --tune_part 4 \
                --cosine \
                --base_seman_calib 1 \
                --train_weight_base 1 \
                --neg_gen_type semang \
                --agg mlp \
                --gpus 1 \
                --n_train_para 2 \
                --n_train_runs 400 \
                --op_loss 0.1 \
                --protonet False

