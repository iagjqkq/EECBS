from cactus.utils import assertContains, get_param_or_default
from cactus.tensorable import Tensorable
from cactus.constants import *
import torch
import numpy as np
# FRL/cactus/controller/controller.py
# 在文件顶部添加 import
import torch.nn.functional as F


class Memory(Tensorable):

    def __init__(self, params) -> None:
        # ... (保留原有初始化代码)
        self.trajectories = []  # 新增：用于存储完整轨迹的列表

    def save(self, joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated, done, global_state):
        self.episode_buffer.append((joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated))
        self.time_step += 1
        if done:
            self.trajectories.append(list(self.episode_buffer))
            self.episode_buffer.clear()
            self.time_step = 0
            # 限制内存使用
            if len(self.trajectories) > 10000:
                self.trajectories.pop(0)

    def sample_trajectories(self, batch_size, sequence_length):
        """新的采样方法：采样一个批次的轨迹序列"""
        batch_states, batch_actions, batch_rtg, batch_timesteps, batch_mask = [], [], [], [], []

        for _ in range(batch_size):
            # 1. 随机选择一个回合轨迹
            traj_idx = np.random.randint(0, len(self.trajectories))
            traj = self.trajectories[traj_idx]

            # 2. 在该回合中随机选择一个起始点
            start_idx = np.random.randint(0, len(traj))
            sequence = traj[start_idx: start_idx + sequence_length]

            # 3. 提取状态、动作和奖励
            states = torch.stack([s[0] for s in sequence])
            actions = torch.stack([s[1] for s in sequence])
            rewards = torch.stack([s[2] for s in sequence])

            # 4. 计算 "Returns-To-Go" (RTG)
            discounts = self.gamma ** np.arange(len(rewards), device=self.device)
            rtg = [sum(discounts[:len(rewards) - i] * rewards[i:]) for i in range(len(rewards))]
            returns_to_go = self.as_float_tensor(rtg).unsqueeze(-1)

            timesteps = self.as_int_tensor(range(start_idx, start_idx + len(sequence)))

            # 5. 对序列进行填充 (padding)
            pad_len = sequence_length - len(sequence)
            states = F.pad(states, (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_len), 'constant', 0)
            actions = F.pad(actions, (0, 0, 0, pad_len), 'constant', 0)
            returns_to_go = F.pad(returns_to_go, (0, 0, 0, pad_len), 'constant', 0)
            timesteps = F.pad(timesteps, (0, pad_len), 'constant', 0)

            # 6. 创建注意力掩码 (用于训练时忽略填充部分)
            mask = torch.cat([torch.ones(len(sequence)), torch.zeros(pad_len)]).to(self.device)

            batch_states.append(states)
            batch_actions.append(actions)
            batch_rtg.append(returns_to_go)
            batch_timesteps.append(timesteps)
            batch_mask.append(mask)

        return (
            self.stack(batch_states),
            self.stack(batch_actions).long(),  # 动作需要是长整型
            self.stack(batch_rtg),
            self.stack(batch_timesteps),
            self.stack(batch_mask)
        )

    def clear(self):
        # ... (保留原有代码)
        self.trajectories.clear()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.metrics = {}

    def record_metric(self, metric, value):
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics[metric].append(value)

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = numpy.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

class Memory(Tensorable):

    def __init__(self, params) -> None:
        assertContains(params, ENV_NR_AGENTS)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, TORCH_DEVICE)
        assertContains(params, ENV_TIME_LIMIT)
        assertContains(params, ENV_GAMMA)
        self.device = torch.device(params[TORCH_DEVICE]['type']) if isinstance(params[TORCH_DEVICE], dict) else params[TORCH_DEVICE]
        super(Memory, self).__init__(params.get(TORCH_DEVICE, torch.device('cpu')))
        self.buffer = ReplayBuffer(capacity=10000)
        self.observation_dim = params[ENV_OBSERVATION_DIM]
        self.observation_size = numpy.prod(self.observation_dim)
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.time_limit = params[ENV_TIME_LIMIT]
        self.gamma = params[ENV_GAMMA]
        self.episode_buffer = []
        self.joint_observations = []
        self.joint_actions = []
        self.joint_rewards = []
        self.joint_returns = []
        self.joint_dones = []
        self.max_episode_length = 0
        self.episode_count = 0
        self.time_step = 0

    def save(self, joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated, done, global_state):
        self.episode_buffer.append((joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated, global_state))
        self.time_step += 1
        if done:
            self.time_step = 0
            self.episode_count += 1
            episode_length = len(self.episode_buffer)
            assert episode_length <= self.time_limit, f"{episode_length} should not be greater than {self.time_limit}"
            if episode_length > self.max_episode_length:
                self.max_episode_length = episode_length
            returns = self.float_zeros(self.nr_agents)
            joint_observation_buffer = []
            joint_action_buffer = []
            joint_reward_buffer = []
            joint_return_buffer = []
            joint_done_buffer = []
            for transition in reversed(self.episode_buffer):
                obs, actions, rewards, dones, _, global_state = transition
                returns = rewards + self.gamma*returns
                joint_action_buffer.append(actions)
                joint_observation_buffer.append(obs)
                joint_reward_buffer.append(rewards)
                joint_done_buffer.append(dones)
                joint_return_buffer.append(returns.detach().clone())
            joint_observation_buffer.reverse()
            joint_action_buffer.reverse()
            joint_reward_buffer.reverse()
            joint_return_buffer.reverse()
            joint_done_buffer.reverse()
            self.joint_observations.append(self.stack(joint_observation_buffer))
            self.joint_actions.append(self.stack(joint_action_buffer))
            self.joint_rewards.append(self.stack(joint_reward_buffer))
            self.joint_dones.append(self.stack(joint_done_buffer))
            self.joint_returns.append(self.stack(joint_return_buffer))
            self.episode_buffer.clear()

    def get_training_data(self, truncated=False):
        if truncated:
            return self.cat(self.joint_observations).view(-1, self.nr_agents, self.observation_size),\
                self.cat(self.joint_actions).view(-1, self.nr_agents),\
                self.cat(self.joint_returns).view(-1, self.nr_agents),\
                self.cat(self.joint_dones).view(-1, self.nr_agents),\
                self.cat(self.joint_rewards).view(-1, self.nr_agents)
        else:
            observation_tensor = self.float_zeros(\
                [self.max_episode_length, self.episode_count, self.nr_agents] + self.observation_dim)
            action_tensor = self.int_zeros([self.max_episode_length, self.episode_count, self.nr_agents])
            reward_tensor = self.float_zeros([self.max_episode_length, self.episode_count, self.nr_agents])
            return_tensor = self.float_zeros([self.max_episode_length, self.episode_count, self.nr_agents])
            done_tensor = self.bool_ones([self.max_episode_length, self.episode_count, self.nr_agents])
            index = 0
            for obs, action, reward, done, returns in\
                zip(self.joint_observations, self.joint_actions, self.joint_rewards, self.joint_dones, self.joint_returns):
                episode_length = obs.size(0)
                observation_tensor[:episode_length, index, :, :, :, :] = obs
                action_tensor[:episode_length, index, :] = action
                reward_tensor[:episode_length, index, :] = reward
                return_tensor[:episode_length, index, :] = returns
                done_tensor[:episode_length, index, :] = done
                index += 1
            return observation_tensor, action_tensor, return_tensor, done_tensor, reward_tensor

    def clear(self):
        self.joint_observations.clear()
        self.joint_actions.clear()
        self.joint_rewards.clear()
        self.joint_returns.clear()
        self.joint_dones.clear()
        self.max_episode_length = 0
        self.episode_count = 0

class Controller(Tensorable):
    """
    Base class for all controllers
    """
    def __init__(self, params) -> None:
        assertContains(params, ENV_NR_AGENTS)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, TORCH_DEVICE)
        assertContains(params, EPISODES_PER_EPOCH)
        self.device = torch.device(params[TORCH_DEVICE]['type']) if isinstance(params[TORCH_DEVICE], dict) else params[TORCH_DEVICE]
        super(Controller, self).__init__(self.device)
        self.observation_dim = params[ENV_OBSERVATION_DIM]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.episodes_per_epoch = params[EPISODES_PER_EPOCH]
        self.memory = Memory(params)
        self.episode_count = 0
        self.grad_norm_clip = get_param_or_default(params, GRAD_NORM_CLIP, 1)
        self.learning_rate = get_param_or_default(params, LEARNING_RATE, 0.001)
        self.vdn_mode = get_param_or_default(params, VDN_MODE, False)
        self.reward_sharing = get_param_or_default(params, REWARD_SHARING, True)
        self.agent_ids = [i for i in range(self.nr_agents)]
        self.grid_operations = self.int_zeros((NR_GRID_ACTIONS, ENV_2D))
        self.grid_operations[WAIT]  = self.as_int_tensor([ 0,  0])
        self.grid_operations[NORTH] = self.as_int_tensor([ 0,  1])
        self.grid_operations[SOUTH] = self.as_int_tensor([ 0, -1])
        self.grid_operations[WEST]  = self.as_int_tensor([-1,  0])
        self.grid_operations[EAST]  = self.as_int_tensor([ 1,  0])

    def get_parameter_count(self):
        return 0

    def save_model_weights(self, path):
        pass

    def load_model_weights(self, path):
        pass

    def joint_policy(self, joint_observation):
        return torch.randint(0, self.nr_actions, (self.nr_agents,))

    def train(self):
        pass

    def reset_hidden_state(self):
        pass

    def update(self, joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated, done, info, global_state=None):
        self.memory.save(joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated, done, global_state)
        if done:
            self.episode_count += 1
            if self.episode_count > 0 and self.episode_count%self.episodes_per_epoch == 0:
                self.train()
                self.memory.clear()
                self.episode_count = 0
            self.reset_hidden_state()

class LNS2RLController(Controller):
    """
    Base class for LNS2RL controllers with conflict resolution
    """
    def __init__(self, params, env=None):
        super().__init__(params)
        self.conflict_threshold = get_param_or_default(params, 'conflict_threshold', 0.5)
        self.env = env

    def detect_conflicts(self, observations):
        raise NotImplementedError

    def resolve_conflicts(self, actions):
        raise NotImplementedError