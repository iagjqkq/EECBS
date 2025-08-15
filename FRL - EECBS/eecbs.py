# run_training.py
import cactus.algorithms as algorithms
import cactus.maps as maps
import cactus.experiments as experiments
import cactus.data as data
import copy
from cactus.constants import *
import argparse

# 默认参数设置
nr_episodes = 1
params = {}
params['clip_threshold'] = 1.0
params['noise_std'] = 0.1
params[ENV_OBSERVATION_SIZE] = 7
params[HIDDEN_LAYER_DIM] = 64
params[NUMBER_OF_EPOCHS] = 5  # 方便快速测试，实际实验请调大
params[EPISODES_PER_EPOCH] =30
params[LEARNING_RATE] = 0.0005
params[EPOCH_LOG_INTERVAL] = 10
params[ENV_TIME_LIMIT] = 512
params[TEST_INIT_GOAL_RADIUS] = None
params[ENV_GAMMA] = 0.99
params[RENDER_MODE] = False
params[ENV_MAKESPAN_MODE] = False
params[GRAD_NORM_CLIP] = 10
params[VDN_MODE] = False
params[REWARD_SHARING] = False
params[MIXING_HIDDEN_SIZE] = 128
params[WINDOW_WIDTH] = 800
params[WINDOW_HEIGHT] = 600
params[REAL_TIME_LOG] = True
params[ENV_INIT_GOAL_RADIUS] = 3
params[TEST_INIT_GOAL_RADIUS] = 3
params['buffer_size'] = 10000
params['collision_weight'] = 1.0
params['exploration_reward_coef'] = 0.005
params['marl_performance_queue_capacity'] = 20
params['marl_switch_threshold'] = 0.3

# 为不同智能体数量配置LNS参数
agent_configs = {
    10: {'lns_depth': 2, 'neighborhood_size': 8},  # 原为 5
    20: {'lns_depth': 4, 'neighborhood_size': 15},  # 原为 9
    30: {'lns_depth': 5, 'neighborhood_size': 20},  # 原为 12
    40: {'lns_depth': 6, 'neighborhood_size': 22},  # 原为 14
    50: {'lns_depth': 7, 'neighborhood_size': 25},  # 原为 15
    60: {'lns_depth': 8, 'neighborhood_size': 28},  # 原为 17
    70: {'lns_depth': 9, 'neighborhood_size': 30},  # 原为 18
    80: {'lns_depth': 10, 'neighborhood_size': 32},  # 原为 20
    90: {'lns_depth': 11, 'neighborhood_size': 35},  # 原为 21
    100: {'lns_depth': 12, 'neighborhood_size': 38}  # 原为 23
}


def run(algorithm_name: str, args):
    algorithm_name = ALGORITHM_EECBS
    params[ALGORITHM_NAME] = algorithm_name
    algorithm_tag = 'Enhanced_ECBS'

    # 指定要使用的.map文件名称
    map_list = ['1.empty-48-48', '2.random-64-64-20', '3.maze-128-128-2', '4.Berlin_1_256']  # 替换为你的.map文件的实际名称（不包含扩展名）
    for map_name in map_list:
        params[MAP_NAME] = map_name  # Move this line up

        params[DIRECTORY] = f"output/{params[ENV_NR_AGENTS]}agents_{params[MAP_NAME]}/{algorithm_tag}"
        params[DIRECTORY] = data.mkdir_with_timestap(params[DIRECTORY])
        params[TORCH_DEVICE] = {'type': 'cpu'}
        # 调用maps.py的make函数来加载指定的.map文件
        env, _ = maps.make(params)
        training_envs = [copy.deepcopy(env) for _ in range(params[EPISODES_PER_EPOCH])]
        test_envs = [copy.deepcopy(e) for e in training_envs]
        params['env'] = env
        # 获取算法配置，并根据命令行参数强制设置use_centralized_critic和ppo_epoch
        controller_cls, _, _ = algorithms.get_algorithm_config(algorithm_name)  # 获取控制器类，忽略返回值

        # 实例化控制器
        controller = controller_cls(params, env)
        print(f"----- Running Traditional LNS for {params[ENV_NR_AGENTS]} agents -----")


        results = experiments.run_training(training_envs, test_envs, controller, params)
    return controller, results


def main(args):
    params[REAL_TIME_LOG] = args.real_time_log

    for nr_agents, config_values in agent_configs.items():
        params[ENV_NR_AGENTS] = nr_agents
        params[SAMPLE_NR_AGENTS] = nr_agents
        params[LNS_SEARCH_DEPTH] = config_values['lns_depth']
        params[NEIGHBORHOOD_SIZE] = config_values['neighborhood_size']


        # 使用我们为传统LNS定义的常量来运行
        run(ALGORITHM_EECBS, args)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_time_log', type=bool, default=False, help='启用实时训练日志输出')
    args = parser.parse_args()

    # 运行主训练逻辑
    main(args)