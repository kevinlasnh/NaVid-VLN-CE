#!/usr/bin/env python3

import argparse
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config

from habitat import Env
from tqdm import trange
import json
import os

import numpy as np

def main():
    # 接收命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    
    parser.add_argument(
        "--exp-save",
        type=str,
        required=True,
        help="results types requried to be saved",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="names of evaluation model",
    )
    
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )
    
    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )

    args = parser.parse_args()

    # 调用 run_exp 函数，传入命令行参数
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, model_path: str, result_path: str, model_name: str, exp_save: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """

    config = get_config(exp_config, opts)
    # 创建成数据集对象，其中包含了各种关于数据集的必要信息
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)

    # 先对整体的数据集排序，保证每个进程手上的 episode 列表的顺序都是一样的
    # 这行为安全性冗余，如果不加的话，如果进程的顺序彼此一致，那就不会出现问题
    # 反之就会出现数据重复的问题
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    # 固定随机种子
    # 这里比较搞笑的一点就是，这里是伪随机，也就是输入 42 这个固定的数字会导致每次生成的随机数都是一样的
    # 其把 numpy 内部的随机数生成器的状态重设了，这个状态是全局的，幕后的，不通过变量传递
    # 下一行的 split 收到了这行代码的影响，其影响 get_splits 内部的随机数生成器的状态，导致每次切割出来的子数据集都是一样的
    np.random.seed(42)
    # get_splits 返回了一个列表，列表的长度是 split_num，每个元素是一个子数据集，子数据集的长度是原始数据集的长度除以 split_num
    # 相当于是切割成了 8 份，基于现在的 split_num 的值，然后通过 split_id 来选择其中对应的一份数据，不同的进程就是不同的数据块
    dataset_split = dataset.get_splits(split_num)[split_id]

    # 开始评测
    evaluate_agent(config, split_id, dataset_split, model_path, result_path, model_name, exp_save)
  
  
  
  
  
  
def evaluate_agent(config, split_id, dataset, model_path, result_path, model_name, exp_save) -> None:

    # 把数据和仿真器同时塞到一起创建一个真正能跑 episode 的环境对象
    env = Env(config.TASK_CONFIG, dataset)

    # 模型选择
    if model_name == "navid":
        from agent_navid import NaVid_Agent
        agent = NaVid_Agent(model_path, result_path, exp_save)
        
    elif model_name == "uni-navid":
        from agent_uninavid import UniNaVid_Agent
        agent = UniNaVid_Agent(model_path, result_path, exp_save)

    # 存储 episodes 的数量
    num_episodes = len(env.episodes)

    # 把两个早停配置从配置树里面提取出来，缓存成局部变量，方便后续使用
    # 这两个值会在下面的 episode 中被反复循环使用，这样就不用每次都去配置树里面找了，节省了时间
    # 第一个主要防止 agent 卡死在原地空转圈
    # 如果连续 25 步都发现没有缩短目标点的距离，就判断 agent 是在原地打转
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    # 第二个主要防止 agent 卡死在原地不动
    # 这个就是单个 episode 的最大步数限制，超过这个步数就强制结束 episode
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    # 5 个指标，分别是距离目标点的距离、是否成功到达目标点、SPL、路径长度、oracle 成功率
    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    # 一个死变量，没有用到
    count = 0

    # 评测主循环
    for _ in trange(num_episodes, desc=config.EVAL.IDENTIFICATION+"-{}".format(split_id)):
        # 重置环境
        # 结束上一个 episode 开始下一个
        # 返回到第一帧观测
        obs = env.reset()
        # 步数计数器清零
        iter_step = 0
        # 调用之前创建的 agent 对象，让 agent 重置内部状态
        agent.reset()

        # 一个计数器，记录连续多少步没有缩短目标点的距离
        continuse_rotation_count = 0
        # 这里随便给 last_dtg 赋一个大于 0 的值，保证第一次循环的时候 info["distance_to_goal"] != last_dtg
        last_dtg = 999

        # 这里采用了强化学习最进行的智能划分，就是把世界和智能体车体分开
        # env 这里代表了世界，其包含了 3D 仿真器（载入场景之类的东西），还包含了代跑的 episode 列表和任务定义之类的东西
        # agent 这里代表了智能体车体，其包含了模型、动作空间、观测空间、动作决策逻辑之类的东西
        # env 把当前的 obs 观测和 infor 信息传给 agent，agent 根据这些信息加上自己的历史帧去基于 VLM 模型做出决策，然后输出一个动作
        # 然后 env 接收到这个动作之后去执行，执行完之后返回下一帧的 obs 观测和 infor 信息，然后再传给 agent，agent 再做决策，循环往复
        while not env.episode_over:

            # 获取当前的 agent 的状态信息
            info = env.get_metrics()

            # 判断当前的距离目标点的距离是否和上一步距离相同
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            # 每一步的真正决策
            # 输入上一帧结束的 obs 和 infor 和 episode_id，输出一个动作字典
            # 这里是唯一调用模型的地方，模型的推理结果就是这个动作字典
            # 但也不是每次都调用模型去推理，而是先把模型推理出来的动作缓存到一个列表里面，等到下一步的时候再去取这个列表里面的动作
            action = agent.act(obs, info, env.current_episode.episode_id)

            # 这里如果满足了停止条件，直接把动作设置为 0，也就是停止动作，强制结束 episode
            # 更换后的动作直接在这个 step 就执行了，下一步就会结束 episode
            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                action = {"action": 0}

            iter_step+=1
            # 直接执行动作，返回下一帧的观测
            obs = env.step(action)

        # 每个 episode 的跑完之后的收尾阶段
        # 重新拿到终局指标
        info = env.get_metrics()
        # 这个是死代码
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        # 给指标打上 id
        result_dict["id"] = env.current_episode.episode_id
        # 又是一个死变量
        count+=1


        if "data" in exp_save:
            with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result_dict, f, indent=4)



if __name__ == "__main__":
    main()
