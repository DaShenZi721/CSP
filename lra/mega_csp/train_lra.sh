#!/bin/bash
model_name=mega
# transformer mega
dataset_name=path-x
# listops imdb-4000 aan cifar10 pathfinder path-x

DATA=./dataset/lra/${dataset_name}/
datetime=$(date +"%Y-%m-%d-%H-%M-%S")

case ${dataset_name} in
  listops) 
    # --lr-scheduler cosine \
    model=${model_name}_lra_listop
    SAVE=./outputs/${dataset_name}/${model}/${datetime}/
    mkdir -p ${SAVE}
    cp $0 ${SAVE}/run.sh
    nohup python -u train.py ${DATA} \
        --seed 0 --ddp-backend c10d --find-unused-parameters \
        -a ${model} --task lra-text --input-type text \
        --encoder-layers 6 --n-dim 16 --chunk-size -1 \
        --activation-fn 'silu' --attention-activation-fn 'softmax' \
        --norm-type 'layernorm' --sen-rep-type 'mp' \
        --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --optimizer adam --lr 0.001 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
        --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
        --batch-size 64 --sentence-avg --update-freq 1 --max-update 90000 --max-sentences-valid 256 \
        --lr-scheduler linear_decay --total-num-update 90000 --end-learning-rate 0.0 \
        --warmup-updates 3000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
        --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
        --wandb-project lipschitz-lra-listop \
        > output_logs/${model}/${datetime}.log 2>&1 &
    ;;
  imdb-4000) 
    model=${model_name}_lra_imdb
    SAVE=./outputs/${dataset_name}/${model}/${datetime}/
    mkdir -p ${SAVE}
    cp $0 ${SAVE}/run.sh
    nohup python -u train.py ${DATA} \
        --seed 0 --ddp-backend c10d --find-unused-parameters \
        -a ${model} --task lra-text --input-type text \
        --encoder-layers 4 --n-dim 16 --chunk-size -1 \
        --activation-fn 'silu' --attention-activation-fn 'softmax' \
        --norm-type 'scalenorm' --sen-rep-type 'mp' \
        --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --optimizer adam --lr 0.004 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
        --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
        --batch-size 25 --sentence-avg --update-freq 2 --max-update 25000 --required-batch-size-multiple 1 \
        --lr-scheduler linear_decay --total-num-update 25000 --end-learning-rate 0.0 \
        --warmup-updates 10000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --max-sentences-valid 100 \
        --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
        --wandb-project lipschitz-lra-imdb \
        > output_logs/${model}/${datetime}.log 2>&1 &
    ;;
  aan) 
    model=${model_name}_lra_aan
    SAVE=./outputs/${dataset_name}/${model}/${datetime}/
    mkdir -p ${SAVE}
    cp $0 ${SAVE}/run.sh
    nohup python -u train.py ${DATA} \
        --seed 0 --ddp-backend c10d --find-unused-parameters \
        -a ${model} --task lra-text --input-type text \
        --encoder-layers 6 --n-dim 16 --chunk-size -1 \
        --activation-fn 'silu' --attention-activation-fn 'softmax' \
        --norm-type 'scalenorm' --sen-rep-type 'mp' \
        --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --optimizer adam --lr 0.006 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
        --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.04 \
        --batch-size 8 --sentence-avg --update-freq 8 --max-update 20000 \
        --lr-scheduler linear_decay --total-num-update 91960 --end-learning-rate 0.0 \
        --warmup-updates 10000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --max-sentences-valid 64 \
        --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
        --wandb-project lipschitz-lra-aan \
        > output_logs/${model}/${datetime}.log 2>&1 &
    ;;
  cifar10) 
    model=${model_name}_lra_cifar10
    SAVE=./outputs/${dataset_name}/${model}/${datetime}/
    mkdir -p ${SAVE}
    cp $0 ${SAVE}/run.sh
    nohup python -u train.py ${DATA} \
        --seed 0 --ddp-backend c10d --find-unused-parameters \
        -a ${model} --task lra-image --input-type image --pixel-normalization 0.48 0.24 \
        --encoder-layers 8 --n-dim 16 --chunk-size -1 \
        --activation-fn 'silu' --attention-activation-fn 'laplace' \
        --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
        --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
        --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.02 \
        --batch-size 50 --sentence-avg --update-freq 1 --max-update 180000 \
        --lr-scheduler linear_decay --total-num-update 180000 --end-learning-rate 0.0 \
        --warmup-updates 9000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
        --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
        --wandb-project lipschitz-lra-cifar10 \
        > output_logs/${model}/${datetime}.log 2>&1 &
    ;;
  pathfinder) 
    model=${model_name}_lra_pf32
    SAVE=./outputs/${dataset_name}/${model}/${datetime}/
    mkdir -p ${SAVE}
    cp $0 ${SAVE}/run.sh
    nohup python -u train.py ${DATA} \
        --seed 0 --ddp-backend c10d --find-unused-parameters \
        -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
        --encoder-layers 6 --n-dim 16 --chunk-size -1 \
        --activation-fn 'silu' --attention-activation-fn 'laplace' \
        --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
        --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --optimizer adam --lr 0.03 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
        --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
        --batch-size 64 --sentence-avg --update-freq 1 --max-update 250000 \
        --lr-scheduler linear_decay --total-num-update 250000 --end-learning-rate 0.0 \
        --warmup-updates 50000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --max-sentences-valid 512 \
        --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
        --wandb-project lipschitz-lra-pf32 \
        > output_logs/${model}/${datetime}.log 2>&1 &
    ;;
  path-x) 
    model=${model_name}_lra_pf128
    SAVE=./outputs/${dataset_name}/${model}/${datetime}/
    mkdir -p ${SAVE}
    cp $0 ${SAVE}/run.sh
    nohup python -u train.py ${DATA} \
        --seed 0 --ddp-backend c10d --find-unused-parameters \
        -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
        --encoder-layers 4 --n-dim 16 --chunk-size -1 \
        --activation-fn 'silu' --attention-activation-fn 'laplace' \
        --norm-type 'syncbatchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
        --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
        --dropout 0.5 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
        --batch-size 60 --sentence-avg --update-freq 8 --max-update 125000 \
        --lr-scheduler linear_decay --total-num-update 125000 --end-learning-rate 0.0 \
        --warmup-updates 25000 --warmup-init-lr '1e-07' --warmup-power 2 --keep-last-epochs 1 --max-sentences-valid 12 \
        --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
        --wandb-project lipschitz-lra-pf128 \
        > output_logs/${model}/${datetime}.log 2>&1 &
    ;;
esac
