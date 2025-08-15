from os.path import join
from cactus.constants import *
import numpy as np
from datetime import datetime
import cactus.data as data
from cactus.data import CSV_FIELD_ORDER
import time

import random



def run_episode(envs, controller, params, training_mode=True, render_mode=False, env_index=None):
    if env_index is not None:
        env = envs[env_index]
    else:
        env = random.choice(envs)
    done = False
    time_step = 0
    observations = env.reset()
    vertex_collisions = env.float_zeros(env.nr_agents)
    edge_collisions = env.float_zeros(env.nr_agents)
    info = {ENV_COMPLETION_RATE: 0.0, 'time_step': 0}
    while not done:
        joint_action = controller.joint_policy(observations)
        next_observations, rewards, terminated, truncated, info = env.step(joint_action)
        vertex_collisions += info[ENV_VERTEX_COLLISIONS].to(FLOAT_TYPE)
        edge_collisions += info[ENV_EDGE_COLLISIONS].to(FLOAT_TYPE)
        done = env.is_done_all()
        time_step += 1
        info['time_step'] = time_step
        if training_mode:
            # 拼接全局状态
            global_state = torch.cat([obs.unsqueeze(0) for obs in observations], dim=0).flatten()
            controller.update(observations, joint_action, rewards, terminated, truncated, done, info,
                              global_state=global_state)
        if render_mode:
            env.render()
        observations = next_observations

    # <-- 新增: 计算路径成本总和 (SoC) -->
    # 对于未完成的智能体，其成本计为时间上限
    costs = torch.where(env.agent_costs > 0, env.agent_costs, env.time_limit)
    sum_of_costs = costs.sum().item()

    return {
        DISCOUNTED_RETURNS: env.discounted_returns,
        UNDISCOUNTED_RETURNS: env.undiscounted_returns,
        VERTEX_COLLISIONS: vertex_collisions,
        EDGE_COLLISIONS: edge_collisions,
        COMPLETION_RATE: info[ENV_COMPLETION_RATE],
        SUM_OF_COSTS: sum_of_costs,
        'time_step': info['time_step']
    }


def run_episodes(nr_episodes, envs, controller, params, training_mode=True, render_mode=False):
    start_time = time.time()
    successes = 0.0
    total_completion_rates = 0.0
    undiscounted_sum = 0.0
    sum_of_costs_total = 0.0  # <-- 新增
    for i in range(nr_episodes):
        result = run_episode(envs, controller, params, training_mode, render_mode, env_index=i)

        # 获取当前环境的可行路径
        feasible_paths = envs[i].get_feasible_paths()
        successful_agents_in_episode = sum(1 for p in feasible_paths if p)
        episode_completion_rate = successful_agents_in_episode / envs[i].nr_agents

        # 累加各环境完成率用于最终平均
        total_completion_rates += episode_completion_rate

        # 更新全成功计数
        if successful_agents_in_episode == envs[i].nr_agents:
            successes += 1

        # 更新控制器指标
        controller.update_metrics(
            success_rate=(successes / (i + 1)) if (i + 1) > 0 else 0.0,
            completion_rate=episode_completion_rate
        )

        undiscounted_sum += result[UNDISCOUNTED_RETURNS].sum()
        sum_of_costs_total += result[SUM_OF_COSTS]  # <-- 新增
        print(f"Env {i} 成功智能体: {successful_agents_in_episode}/{envs[i].nr_agents}")

    final_success_rate = (successes / nr_episodes) if nr_episodes > 0 else 0.0
    final_completion_rate = total_completion_rates / nr_episodes if nr_episodes > 0 else 0.0
    success_rate_variance = final_success_rate * (1.0 - final_success_rate)

    # Ensure update_metrics is called with final values at the end of the epoch if not already done inside the loop
    # controller.update_metrics(final_success_rate, final_completion_rate)

    return {
        "epoch": nr_episodes,
        "success_rate": float(final_success_rate),
        "completion_rate": float(final_completion_rate),
        "sum_of_costs": float(sum_of_costs_total / nr_episodes),
        "training_time": time.time() - start_time
    }





def run_training(envs, test_envs, controller, params):

    episodes_per_epoch = params[EPISODES_PER_EPOCH]
    success_rates = []
    completion_rates = []
    prev_total_time = 0
    total_time = 0
    training_times = []
    areas_under_curve_success = []
    areas_under_curve_completion = []
    sum_of_costs_list = []
    training_results = []
    all_results = []
    training_result = {COMPLETION_RATE: 0, SUCCESS_RATE_VARIANCE: 0}
    result = {  # 初始化result字典
        TOTAL_TIME: 0,
        TIME_PER_EPOCH: 0,
        SUCCESS_RATE: [],
        COMPLETION_RATE: [],
        SUM_OF_COSTS: [],
        AUC_COMPLETION: [],
        AUC_SUCCESS: [],
        TRAINING_TIME: []
    }


    for i in range(params[NUMBER_OF_EPOCHS] + 1):
        start = time.time()




        # 训练执行模块
        training_result = run_episodes(episodes_per_epoch, envs, controller, params,
                                       training_mode=True, render_mode=params[RENDER_MODE])
        training_results.append(training_result)
        all_results.append({
            'epoch': i,
            'success_rate': training_result['success_rate'],
            'completion_rate': training_result['completion_rate'],
            'sum_of_costs': training_result['sum_of_costs'],
            'training_time': time.time() - start
        })

        # 实时日志模块
        if params.get(REAL_TIME_LOG, True):
            # 日志输出代码
            elapsed_time = time.time() - start
            log_msg = (
                f"Epoch {i:04d} | Time: {elapsed_time:.1f}s | "
                f"总回报:{training_result.get(UNDISCOUNTED_RETURNS, 0):.2f} | "
                f"成功率:{training_result[SUCCESS_RATE]:.2f} | "
                f"完成率:{training_result[COMPLETION_RATE]:.2f} | "
                f"路径成本:{training_result.get(SUM_OF_COSTS, 0):.2f}"
            )
            if hasattr(controller, 'current_neighborhood_size'):
                log_msg += f" | 邻域:{controller.current_neighborhood_size}"
            if hasattr(envs[0], 'get_metrics'):
                metrics = envs[0].get_metrics()
                log_msg += f" | 路径长度:{metrics['average_path_length']:.1f}"
                log_msg += f" | 顶点冲突:{metrics['vertex_collisions']}次"
                log_msg += f" | 边冲突:{metrics['edge_collisions']}次"
            print(log_msg)

        end = time.time() - start
        total_time += end
        training_time = total_time - prev_total_time
        prev_total_time = total_time
        print(f"Epoch {i} 训练结果:")
        print(f"- 训练成功率: {training_result[SUCCESS_RATE]:.2%}")
        print(f"- 训练完成率: {training_result[COMPLETION_RATE]:.2%}")
        print(f"- 平均路径成本: {training_result.get(SUM_OF_COSTS, 0):.2f}")  # <-- 新增

        success_rates.append(float(training_result[SUCCESS_RATE]))
        completion_rates.append(float(training_result[COMPLETION_RATE]))
        sum_of_costs_list.append(float(training_result.get(SUM_OF_COSTS, 0)))  # <-- 新增

        if i % params[EPOCH_LOG_INTERVAL] == 0:
            training_times.append(training_time)
        if i > 0 and (i % 500 == 0 or i == params[NUMBER_OF_EPOCHS]):
            controller.save_model_weights(params[DIRECTORY])
            # 添加综合统计指标
            final_results = {
                'avg_success_rate': np.mean(success_rates),  # 平均成功率(所有epoch)
                'avg_completion_rate': np.mean(completion_rates),  # 平均任务完成率
                'final_success_rate': success_rates[-1],  # 最终epoch成功率
                'final_completion_rate': completion_rates[-1],  # 最终epoch完成率
                'total_training_time': sum(training_times),  # 总训练时长(秒)
                'success_rate_std': np.std(success_rates)  # 成功率标准差
            }
            result = {
                SUCCESS_RATE: success_rates[-1],
                COMPLETION_RATE: completion_rates[-1],
                SUM_OF_COSTS: sum_of_costs_list[-1],
                AUC_COMPLETION: areas_under_curve_completion,
                AUC_SUCCESS: areas_under_curve_success,
                TRAINING_TIME: training_times
            }
            # 已移除单个epoch文件保存逻辑
    result = {

        TOTAL_TIME: total_time,
        TIME_PER_EPOCH: total_time * 1.0 / params[NUMBER_OF_EPOCHS],
        SUCCESS_RATE: success_rates,
        COMPLETION_RATE: completion_rates,
        SUM_OF_COSTS: sum_of_costs_list,  # <-- 新增

        AUC_COMPLETION: areas_under_curve_completion,
        AUC_SUCCESS: areas_under_curve_success,
        TRAINING_TIME: training_times,
    }
    if DIRECTORY in params:
        # 使用标准化数据保存
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # unified_filename = f"experiment_results_{timestamp}.json"
        # data.save_json(join(params[DIRECTORY], unified_filename), {
        #         "epoch": len(result[SUCCESS_RATE]),
        #         "success_rate": result[SUCCESS_RATE][-1],
        #         "completion_rate": result[COMPLETION_RATE][-1],
        #         "sum_of_costs": sum_of_costs_list[-1],
        #         "training_time": result[TRAINING_TIME][-1],
        #         "total_time": total_time,
        #         "time_per_epoch": total_time * 1.0 / params[NUMBER_OF_EPOCHS]
        #     })
        # 统一保存所有结果到单个JSON文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unified_filename = join(params[DIRECTORY], f"unified_results_{timestamp}.json")
        data.save_json(unified_filename, {
            "epochs": all_results,
            "total_training_time": total_time,
            "average_success_rate": np.mean(success_rates),
            "average_completion_rate": np.mean(completion_rates)
        })
        
        controller.save_model_weights(params[DIRECTORY])
    return result