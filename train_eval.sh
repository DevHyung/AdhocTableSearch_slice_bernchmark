#!/bin/bash
export fold_list="1"
for i in $fold_list
do
    DATA=./data/bench/$i
    echo $DATA
    
    mkdir -p ${DATA}/processed
    
    pushd ${DATA}/processed &> /dev/null
    rm -rf test_*.table
    rm -rf train_*.pair
    popd &> /dev/null
    
    pushd output/SIGIR/${i} &> /dev/null
    rm -rf checkpoint-epoch*
    popd &> /dev/null
    
    python trainer.py --data_dir ${DATA} \
                      --tabert_path ./model/tabert_base_k3/model.bin \
                      --config_file ./model/tabert_base_k3/tb_config.json \
                      --gpus 1 \
                      --precision 16 \
                      --max_epochs 1 \
		      --weight_decay 0.0 \
		      --min_rows 10 \
		      --max_tables 10 \
                      --lr 5e-5 \
                      --do_train \
                      --gradient_clip_val 1.0 \
                      --train_batch_size 3 \
                      --valid_batch_size 3 \
                      --output_dir output/bench/${i} \
                      --accumulate_grad_batches 16 
    
    CKPT=`ls output/bench/${i} | sort -k3 -t'=' | head -1`
    
    echo ${CKPT}
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
			      --topk 1000 \
                              --ckpt_file output/bench/${i}/${CKPT} \
                              --hnsw_index | tee ./result/${i}_hnsw.result
    
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
			      --topk 1000 \
                              --ckpt_file output/bench/${i}/${CKPT} | tee ./result/${i}.result
done

python predict.py
