#!/bin/bash

CHUNKS=8

################ NaVid ################

MODEL_PATH="model_zoo/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split" 
MODEL_NAME="navid" # uni-navid or navid
EXP_SAVE="video-data" # use "data" to accelerate evaluation

#R2R
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
# SAVE_PATH="tmp/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-10-15-split_on_r2r-120" 


#RxR
CONFIG_PATH="VLN_CE/vlnce_baselines/config/rxr_baselines/navid_rxr.yaml"
SAVE_PATH="tmp/results_navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split-rxr" 


################ Uni-NaVid ################

# MODEL_PATH="model_zoo/llama-vid-7b-full-224-video-fps-1-grid-2-panda-encoder-2025-10-12-all-data"
# MODEL_NAME="uni-navid" # uni-navid or navid
# EXP_SAVE="video-data" # use "data" to accelerate evaluation

#R2R
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_r2r.yaml"
# SAVE_PATH="tmp/results_uninavid-7b-full-224-video-fps-1-grid-2-r2r" 


#RxR
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/rxr_baselines/uninavid_rxr.yaml"
# SAVE_PATH="tmp/results_uninavid-7b-full-224-video-fps-1-grid-2-rxr" 




# 这里启动整个程序
# 一次性启动 8 个独立进程，这里主要的目的不是为了分散显存压力，而是为了可以同时跑 8 个 episode，这样子可以更快的出数据
# 同时启动了 8 个 run.py 进程
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo $(( IDX % 8 ))

    # 这里启动一个 run.py 进程，并传入 7 个命令行参数
    # 说明：bash 中以 "\" 续行的命令中间不能插注释行，否则 "#" 会被当作普通参数传给 python，
    #       因此下面把 7 个参数的语义统一写在这里，顺序与命令中的参数顺序一致。
    #       参数定义见 run.py 第 15-67 行的 add_argument，实际使用见第 73-98 行的 run_exp。
    #
    # --exp-config ：实验配置文件（yaml）路径，决定用哪个数据集、哪套 TASK_CONFIG；
    #                在 run.py 里交给 get_config() 解析，是整次评测的配置入口。
    # --split-num  ：把数据集切成多少份，这里传 $CHUNKS（8）；
    #                作用是与 --split-id 配合切分数据集，让多个进程各跑一部分 episode。
    # --split-id   ：当前进程取第几份数据，这里传循环变量 $IDX（0..7）；
    #                run.py 用 get_splits(split_num)[split_id] 取对应子集，
    #                配合固定的随机种子 np.random.seed(42) 保证各进程切片不重叠、不遗漏。
    # --model-path ：模型权重目录，作为 NaVid_Agent / UniNaVid_Agent 的构造入参，
    #                例如 model_zoo/navid-7b-... 这种目录。
    # --result-path：结果输出目录，用于保存日志、统计 json 和可视化结果，
    #                即脚本顶部的 $SAVE_PATH。
    # --model-name ：要评测的模型名，run.py 据此分支加载不同 Agent：
    #                "navid" 走 NaVid_Agent，"uni-navid" 走 UniNaVid_Agent。
    # --exp-save   ：需要保存的结果类型，取值 "video-data" 或 "data"；
    #                传 "data" 时只落统计 json 不存视频，可以显著加快评测速度。
    CUDA_VISIBLE_DEVICES=$(( IDX % 8 )) python run.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --model-path $MODEL_PATH \
    --result-path $SAVE_PATH \
    --model-name $MODEL_NAME \
    --exp-save $EXP_SAVE &
    
done

wait

