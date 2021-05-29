#!/bin/bash
export fold_list="1"
for i in $fold_list 
do
    DATA=./data/sigir/$i
    echo $DATA

    CKPT=`ls output/sigir/${i} | sort -k3 -t'=' | head -1`
    
    echo ${CKPT}
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
			      --topk 500 \
                              --ckpt_file /workspace/output/sigir/${i}/${CKPT} \
                              --hnsw_index | tee ./result/${i}_hnsw.result
    
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
			      --topk 500 \
                              --ckpt_file /workspace/output/sigir/${i}/${CKPT} | tee ./result/${i}.result
done

python predict.py
