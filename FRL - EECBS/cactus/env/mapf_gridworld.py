from cactus.env.collision_gridworld import CollisionGridWorld
from cactus.utils import assertContains
from cactus.data import TensorEncoder
from cactus.constants import *
from gym import spaces
import time


"""
 Represents a navigation task in a 2D grid world environment with collision detection.
"""

class MAPFGridworld(CollisionGridWorld):
    def get_sum_of_costs(self):
        return torch.sum(self.agent_costs).item()
    '''Multi-Agent Path Finding Gridworld Environment'''
    
    @classmethod
    def from_file(cls, map_path: str, num_agents: int):
        grid = super().load_grid(map_path)
        return cls(grid, num_agents=num_agents)
    @classmethod
    def from_file(cls, map_path: str, num_agents: int):
        '''Factory method to create instance from .map file'''
        grid = super().from_file(map_path)
        return cls(grid, num_agents=num_agents)

    def __init__(self, params, init_goal_radius=None) -> None:
        # Set observation dimension before superclass init
        self.observation_size = params.get("observation_size", 7)
        params[ENV_OBSERVATION_DIM] = [9, self.observation_size, self.observation_size]
        # 在父类初始化前设置观察维度
        params['nr_channels'] = 9  # Updated channel count for heatmap integration
        super().__init__(params)
        self.params = params  # 保存参数到实例属性
        self.nr_channels = params['nr_channels']  # 显式设置通道数属性
        # 初始化agent状态追踪属性
        self.actual_goal_reached = self.bool_zeros(self.nr_agents)


        # 在父类初始化后立即创建zero_observation
        # 计算预期元素数量并验证
        expected_elements = self.nr_agents * self.nr_channels * self.observation_size ** 2
        actual_elements = super().joint_observation().numel() if super().joint_observation() is not None else 0
        assert actual_elements == expected_elements, f"观测维度不匹配: 预期{expected_elements}元素 实际{actual_elements}元素"

        self.zero_observation = self.float_zeros((self.nr_agents, 9, self.observation_size, self.observation_size))  # Updated channel dimension
        self.one_observation = self.float_ones((self.nr_agents, self.nr_channels, self.observation_size, self.observation_size)).expand(-1, 9, -1, -1)
        half_size = int(self.observation_size / 2)
        self.observation_dx, self.observation_dy = self.get_delta_tensor(half_size)
        self.min_size = 10
        self.max_size = 80
        self.obstacle_density = 0.3
        self.current_size = params.get(ENV_WIDTH, 20)

    def reset(self):
        super(MAPFGridworld, self).reset()
        self.time_step = 0
        self.actual_goal_reached[:] = False
        # Initialize observation buffer with proper size
        buffer_size = self.params.get('time_limit', 256)
        self.joint_observation_buffer = self.zero_observation.repeat(buffer_size + 1, 1, 1, 1, 1)
        buffer_size = self.params.get('time_limit', 256)
        buffer_idx = self.time_step % (buffer_size + 1)
        return self.joint_observation_buffer[buffer_idx]

    def set_difficulty(self, size, density):
        from cactus.env.env_generator import generate_random_obstacles
        self.current_size = numpy.clip(size, self.min_size, self.max_size)
        self.obstacle_density = numpy.clip(density, 0, 0.3)
        self.obstacles = generate_random_obstacles(self.current_size, self.obstacle_density)
        self._update_observation_space()

    def _update_observation_space(self):
        from cactus.env.env_generator import generate_random_obstacles
        self.observation_space = [
            spaces.Box(low=0, high=1, shape=(9, self.observation_size, self.observation_size)) 
            for _ in range(self.nr_agents)
        ]
        assertContains(self.params, ENV_OBSERVATION_SIZE)
        self.nr_channels = 9  # Align with updated channel count
        self.observation_size = self.params["observation_size"]
        self.params[ENV_OBSERVATION_DIM] = [9, self.observation_size, self.observation_size]  # Fixed channel dimension

        
        self.current_position_map = -self.int_ones_like(self.obstacle_map)
        self.next_position_map = -self.int_ones_like(self.obstacle_map)
        self.vertex_collision_buffer = self.bool_ones(self.nr_agents)
        self.edge_collision_buffer = self.bool_ones(self.nr_agents)


        self.center_mask = self.float_zeros((self.nr_agents, self.observation_size, self.observation_size))
        half_size = int(self.observation_size/2)
        self.center_mask[:,half_size, half_size] = 1.0
        self.center_mask[:,half_size+1, half_size] = 1.0
        self.center_mask[:,half_size-1, half_size] = 1.0
        self.center_mask[:,half_size, half_size+1] = 1.0
        self.center_mask[:,half_size, half_size-1] = 1.0

    def is_agent_active(self, i):
        """检查智能体是否处于活跃状态（未到达目标且未被移除）"""
        return not torch.all(self.current_positions[i] == self.goal_positions[i])

    def find_path(self, start, goal):
        """A*路径规划算法实现"""
        from heapq import heappush, heappop
        open_set = []
        heappush(open_set, (0, tuple(start.tolist())))
        came_from = {}
        start_tuple = tuple(map(int, start.tolist()))
        g_score = {start_tuple: 0}
        
        while open_set:
            current_tuple = heappop(open_set)[1]
            current = torch.tensor(current_tuple, dtype=INT_TYPE, device=self.device)
            if torch.equal(current, goal):
                return self._reconstruct_path(came_from, current)
                
            for neighbor in self._get_neighbors(current):
                current_tuple = tuple(map(int, current.tolist()))
                neighbor_tuple = tuple(map(int, neighbor.tolist()))
                tentative_g = g_score[current_tuple] + 1
                if neighbor_tuple not in g_score or tentative_g < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current_tuple
                    g_score[neighbor_tuple] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    heappush(open_set, (f_score, tuple(neighbor.tolist())))
        return []

    def _heuristic(self, a, b):
        """曼哈顿距离启发函数"""
        return torch.abs(a[0] - b[0]) + torch.abs(a[1] - b[1])

    def _get_neighbors(self, pos):
        """获取可行移动的相邻位置"""
        x, y = pos
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if self.xy_position_in_bounds(nx, ny) and self.obstacle_map[nx, ny] == 0:
                neighbors.append(torch.tensor([nx, ny], dtype=INT_TYPE, device=self.device))
        return neighbors

    def _reconstruct_path(self, came_from, current):
        """重构路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(torch.tensor(current, dtype=INT_TYPE, device=self.device))
        return path[::-1]

    def get_feasible_paths(self):
        """Track valid paths for all agents within time constraints"""
        return [
            self.find_path(self.current_positions[i], self.goal_positions[i])
            for i in range(self.nr_agents)
            if self.is_agent_active(i)
        ]

    def joint_observation(self):
        raw_obs = super(MAPFGridworld, self).joint_observation()
        if raw_obs is None:
            raw_obs = self.zero_observation
        # 动态调整视图形状
        actual_shape = raw_obs.shape
        expected_shape = (self.nr_agents, self.nr_channels, self.observation_size, self.observation_size)
        expected_elements = self.nr_agents * self.nr_channels * (self.observation_size ** 2)
        # Handle shape mismatch by padding/truncating
        actual_elements = raw_obs.numel()
        new_obs = self.zero_observation.clone()
        copy_elements = min(actual_elements, expected_elements)
        new_obs.view(-1)[:copy_elements] = raw_obs.view(-1)[:copy_elements]
        raw_obs = new_obs
        
        obs = raw_obs.view(expected_shape)
        # Validate tensor dimensions
        assert obs.numel() == raw_obs.numel(), f"Reshape error: {raw_obs.shape} -> {obs.shape}"
        obs[:] = 0
        self.current_position_map[:] = -1
        done = self.is_done()
        done = done.unsqueeze(1)\
            .expand(-1, self.nr_channels*self.observation_size*self.observation_size)\
            .view(-1, self.nr_channels, self.observation_size, self.observation_size)
        half_size = int(self.observation_size/2)
        x0 = self.current_positions[:,0]
        y0 = self.current_positions[:,1]
        self.current_position_map[x0, y0] = torch.arange(self.nr_agents, device=self.device)
        x1 = self.goal_positions[:,0]
        y1 = self.goal_positions[:,1]
        dx = x1 - x0
        dy = y1 - y0
        abs_dx = torch.abs(dx)
        abs_dy = torch.abs(dy)
        manhattan_distance = abs_dx + abs_dy
        euclidean_distance = torch.sqrt(dx*dx + dy*dy)
        max_distance = torch.maximum(abs_dx, abs_dy)
        goal_in_sight = max_distance <= half_size

        # Scan position relative to the goal
        x_direction = torch.sign(dx).to(dtype=INT_TYPE)+half_size
        y_direction = torch.sign(dy).to(dtype=INT_TYPE)+half_size
        obs[self.agent_ids,0, x_direction, half_size] = abs_dx/euclidean_distance
        obs[self.agent_ids,0, half_size, y_direction] = abs_dy/euclidean_distance
        obs[self.agent_ids,0, half_size, half_size] = manhattan_distance.to(dtype=FLOAT_TYPE)
        # 添加目标到达状态标记
        goal_reached = torch.all(self.current_positions == self.goal_positions, dim=1)
        obs[goal_reached,1,:,:] = 0.0
        obs[goal_in_sight,1,dx[goal_in_sight]+half_size, dy[goal_in_sight]+half_size] = 1
        obs[goal_reached,1,half_size,half_size] = 1.0

        # Scan surrounding obstacles and boundaries
        dx = (x0.unsqueeze(1) + self.observation_dx).view(-1)
        dy = (y0.unsqueeze(1) + self.observation_dy).view(-1)
        in_bounds = self.xy_position_in_bounds(dx, dy).view(self.nr_agents, self.observation_size, self.observation_size)
        obs[self.agent_ids,2,:,:] = torch.where(in_bounds, 0.0, 1.0)

        # Scan surrounding agents and their manhattan distances to their goals
        x_clamped = dx.clamp(0, self.rows-1)
        y_clamped = dy.clamp(0, self.columns-1)

        zero_obs = self.zero_observation[:,0,:,:]
        neighbor_ids = self.current_position_map[x_clamped, y_clamped]
        is_agents_position = torch.logical_and(neighbor_ids >= 0, in_bounds.view(-1))
        neighbor_ids = neighbor_ids[is_agents_position]
        if neighbor_ids.any():
            neighbor_condition = is_agents_position.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids,3,:,:] = torch.where(neighbor_condition, 1.0, 0.0)
            flattened_view = obs[self.agent_ids,3].view(-1)
            flattened_view[neighbor_condition.view(-1)] = manhattan_distance[neighbor_ids].to(FLOAT_TYPE) + 1
            obs[self.agent_ids,3,:,:] = flattened_view.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids, 3, half_size, half_size] = 0.0
            obs[self.agent_ids,0,:,:] = torch.where(obs[self.agent_ids,3,:,:] > 0, zero_obs, obs[self.agent_ids,0,:,:])
            obs[self.agent_ids,2,:,:] = torch.where(obs[self.agent_ids,3,:,:] > 0, self.one_observation[:,2,:,:], obs[self.agent_ids,2,:,:])
        
        # Scan surrounding goals and their manhattan distances to their respective agents
        goal_ids = self.occupied_goal_positions[x_clamped, y_clamped]
        is_goal_position = torch.logical_and(goal_ids >= 0, in_bounds.view(-1))
        goal_ids = goal_ids[is_goal_position]
        if goal_ids.any():
            goal_condition = is_goal_position.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids,4,:,:] = torch.where(goal_condition, 1.0, 0.0)
            flattened_view = obs[self.agent_ids,4].view(-1)
            distances = manhattan_distance[goal_ids].to(FLOAT_TYPE) + 1
            flattened_view[goal_condition.view(-1)] = distances
            obs[self.agent_ids,4,:,:] = flattened_view.view(self.nr_agents, self.observation_size, self.observation_size)
            max_distance = distances.max()
            obs[self.agent_ids, 4, :, :] -= obs[self.agent_ids, 1, :, :]
            template = obs[self.agent_ids, 4, :, :]
            obs[self.agent_ids, 4, :, :] = torch.maximum(template, self.float_zeros_like(template))

        # obs = torch.cat([obs, heatmap_batch], dim=1)  # Removed redundant channel addition
        return obs
    
    def get_delta_tensor(self, delta):
        assert delta > 0
        x = []
        y = []
        for _ in range(self.nr_agents):
            for dx in range(-delta, delta+1):
                for dy in range(-delta, delta+1):
                    x.append(dx)
                    y.append(dy)
        return self.as_int_tensor(x).view(self.nr_agents, -1),\
               self.as_int_tensor(y).view(self.nr_agents, -1)

    def check_collisions(self, new_positions):
        collision_mask = super().check_collisions(new_positions)
        collided_agents = collision_mask.nonzero(as_tuple=True)[0]
        
        # 分步代价追踪
        path_costs = self._calculate_path_step_costs(new_positions, collided_agents)
        self.log_step_cost(path_costs)
        
        # 输出当前步骤代价
        print(f'Step {self.current_step} 代价详情:')
        for agent_id, costs in path_costs.items():
            print(f'{agent_id}: 移动={costs["movement"]:.2f}, '
                  f'碰撞={costs["collision"]}, '
                  f'目标距离={costs["goal_proximity"]:.2f}')
        
        # 在info字典中记录碰撞的智能体
        nr_agents_in_env = self.nr_agents
        actual_completions = sum(1 for idx in range(nr_agents_in_env) if self.actual_goal_reached[idx])
        episode_completion_rate = actual_completions / nr_agents_in_env if nr_agents_in_env > 0 else 0.0
        info = {ENV_COLLIDED_AGENTS: collided_agents, ENV_COMPLETION_RATE: episode_completion_rate}
        return collision_mask, info

    def _calculate_path_step_costs(self, new_positions, collided_agents):
        step_costs = {}
        for agent_id in range(self.nr_agents):
            current_pos = self.current_positions[agent_id]
            new_pos = new_positions[agent_id]
            
            # 计算移动代价
            movement_cost = torch.norm(new_pos - current_pos).item()
            
            # 更新目标到达状态
            self.actual_goal_reached[agent_id] = torch.all(new_pos == self.goal_positions[agent_id])
            if self.actual_goal_reached[agent_id]:
                print(f'Agent {agent_id} 到达目标位置 {new_pos.tolist()} (目标位置: {self.goal_positions[agent_id].tolist()})')
            
            # 计算冲突代价
            collision_cost = 1.0 if agent_id in collided_agents else 0.0
            
            # 计算目标接近度
            goal_distance = (torch.abs(self.goal_positions[agent_id][0] - new_pos[0]) + torch.abs(self.goal_positions[agent_id][1] - new_pos[1])).item()
            
            step_costs[f'agent_{agent_id}'] = {
                'movement': movement_cost,
                'collision': collision_cost,
                'goal_proximity': goal_distance,
                'timestamp': time.time()
            }
        return step_costs

    def log_step_cost(self, cost_data):
        import json
        import os
        import time
        
        log_entry = {
            'episode': self.current_episode,
            'step': self.current_step,
            'costs': cost_data,

        }
        if not hasattr(self, 'cost_logs'):
            self.cost_logs = []
        self.cost_logs.append(log_entry)
        
        # 保存到JSON文件
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        output_dir = os.path.join('output', f'results_{self.current_episode}_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, f'results_{self.current_step}.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.cost_logs, f, indent=2, ensure_ascii=False, cls=TensorEncoder)