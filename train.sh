#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
EPOCH=5
SEED=20200401
KFOLD=1
DATASET=bench

rm -r ./data/${DATASET}/${KFOLD}/processed

python trainer.py --data_dir ./data/${DATASET}/${KFOLD} \
  --tabert_path ./model/tabert_base_k3/model.bin \
  --config_file ./model/tabert_base_k3/tb_config.json \
  --gpus 1 \
  --precision 16 \
  --max_epochs ${EPOCH} \
  --weight_decay 0.0 \
  --min_rows 256 \
  --max_tables 2 \
  --lr 5e-5 \
  --gradient_clip_val 1.0 \
  --do_train \
  --train_batch_size 4 \
  --valid_batch_size 4 \
  --test_batch_size 1 \
  --seed ${SEED} \
  --output_dir ./${DATASET}_FOLD${KFOLD}_staticN_3200_E${EPOCH}_S${SEED}/ \
  --accumulate_grad_batches 16 \
