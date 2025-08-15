import torch
import time
import heapq
from cactus.controller.controller import Controller
from cactus.path_planning import PP_SIPPS_Initializer
from cactus.utils import ConstraintTree, detect_conflicts
from .lns_operators import random_removal, greedy_insertion, get_solution_cost
from cactus.constants import LNS_SEARCH_DEPTH, CBS_CONSTRAINT_TREE_DEPTH, WAIT, NORTH, SOUTH, WEST, EAST


# --- 新增：辅助函数 ---

def _actions_to_paths(initial_actions, env):
    """
    一个辅助函数，用于将初始的动作序列转换为路径字典。
    这是一个简化的模拟，它假设智能体成功执行了动作序列。
    """
    paths = {i: [] for i in range(env.nr_agents)}
    current_positions = env.current_positions.clone()

    # (action_id to (dx, dy)) mapping, 必须与环境定义一致
    action_to_delta = {
        WAIT: (0, 0),
        NORTH: (0, 1),
        SOUTH: (0, -1),
        WEST: (-1, 0),
        EAST: (1, 0)
    }

    # 记录初始位置
    for i in range(env.nr_agents):
        paths[i].append(tuple(current_positions[i].tolist()))

    # 模拟执行每一个时间步的动作来构建路径
    # 假设 initial_actions 是 (time_steps, nr_agents) 或类似的形状
    # 这里我们简化为只处理第一步动作，因为初始化器通常只提供一个方向
    if initial_actions.dim() == 1:
        for i in range(env.nr_agents):
            action = initial_actions[i].item()
            delta = action_to_delta.get(action, (0, 0))
            new_pos = current_positions[i] + torch.tensor(delta, device=env.device)
            paths[i].append(tuple(new_pos.tolist()))

    # 在一个更复杂的实现中，这里会有一个循环来处理多步动作序列

    return paths


def _get_action_from_path(current_pos_tuple, path):
    """根据当前位置和路径，计算出下一步的动作。"""
    if not path or len(path) < 2:
        return WAIT

    next_pos_tuple = path[1]
    delta = (next_pos_tuple[0] - current_pos_tuple[0], next_pos_tuple[1] - current_pos_tuple[1])

    action_map = {
        (0, 0): WAIT,
        (0, 1): NORTH,
        (0, -1): SOUTH,
        (-1, 0): WEST,
        (1, 0): EAST
    }
    return action_map.get(delta, WAIT)


# --- 辅助函数结束 ---


class EnhancedECBSController(Controller):
    class ConstraintTree:
        def __init__(self, max_depth=10):
            self.root = {'constraints': {}, 'solution': None, 'children': []}
            self.max_depth = max_depth
            self.current_level = 0

        def expand_node(self, node, constraint):
            if self.current_level < self.max_depth:
                new_constraints = node['constraints'].copy()
                new_constraints.update(constraint)
                child_node = {
                    'constraints': new_constraints,
                    'solution': None,
                    'children': []
                }
                node['children'].append(child_node)
                self.current_level += 1
                return child_node
            return None

        def get_agent_constraints(self, agent_id):
            return [c for c in self.root['constraints'] if c[0] == agent_id]

    def _select_conflict_agents(self, solution):
        """
        从当前解决方案中选择涉及冲突的代理
        Args:
            solution: 当前路径方案字典
        Returns:
            list: 涉及冲突的代理ID列表
        """
        conflict = self._detect_primary_conflict(solution)
        return conflict["agents"] if conflict else []

    def _high_level_search(self, base_solution, constraint_tree):
        """
        实现基于约束树的分层路径规划
        """
        # 1. 生成代理子集
        selected_agents = self._select_conflict_agents(base_solution)
        
        # 2. 构建约束条件下的CBS搜索
        constrained_paths = {}
        for agent in selected_agents:
            path = self._low_level_search(
                agent,
                constraint_tree.get_agent_constraints(agent),
                base_solution[agent]
            )
            if not path:
                return None
            constrained_paths[agent] = path
        
        # 3. 融合新路径与原有路径
        new_solution = base_solution.copy()
        new_solution.update(constrained_paths)
        
        # 4. 验证解决方案的可行性
        if validate_solution(new_solution, self.env):
            return new_solution
        return None
    class ConstraintNode:
        def __init__(self, constraints=None):
            self.constraints = constraints or {}
            self.solution = None
            self.cost = float('inf')

        def update_solution(self, solution, cost):
            self.solution = solution
            self.cost = cost
    """
    EECBS控制器实现，基于冲突树的分层优化框架
    """

    def __init__(self, params, env):
        # 新增EECBS专用参数
        self.constraint_tree = self.ConstraintTree(max_depth=params.get(CBS_CONSTRAINT_TREE_DEPTH, 10))
        self.focal_w = 1.2  # 次优因子
        self.high_level_window = 5  # 高层搜索窗口
        super().__init__(params)
        self.env = env
        self.params = params
        self.training_metrics = {}

        self.iterations = params.get(LNS_SEARCH_DEPTH, 100)
        self.removal_rate = params.get('lns_removal_rate', 0.3)

        # 初始解生成器 (与RL算法共享)
        self.initializer = PP_SIPPS_Initializer(params)

    def _create_constraint(self, conflict):
        """
        根据冲突类型生成对应的约束条件
        Args:
            conflict: 冲突字典，包含冲突类型、参与代理、时间步和位置信息
        Returns:
            dict: 约束条件字典，格式为 {agent_id: [(timestep, position)]}
        """
        constraint = {}
        
        if conflict["type"] == "vertex":
            # 顶点冲突：两个代理同时占据同一位置
            for agent_id in conflict["agents"]:
                constraint.setdefault(agent_id, []).append(
                    (conflict["timestep"], conflict["location"])
                )
        elif conflict["type"] == "edge":
            # 边冲突：两个代理交换位置
            a1, a2 = conflict["agents"]
            t = conflict["timestep"]
            
            # 约束a1在t+1时刻不能从loc1到loc2
            constraint.setdefault(a1, []).append(
                (t, conflict["locations"][0])
            )
            
            # 约束a2在t+1时刻不能从loc2到loc1
            constraint.setdefault(a2, []).append(
                (t, conflict["locations"][1])
            )
        
        return constraint

    def _detect_primary_conflict(self, solution):
        """
        EECBS核心冲突检测方法，识别第一个时空冲突
        """
        max_len = max(len(path) for path in solution.values())
        for t in range(max_len):
            positions = {}
            for agent_id, path in solution.items():
                if t < len(path):
                    pos = path[t] if t < len(path) else path[-1]
                    if pos in positions:
                        return {'type': 'vertex', 'agents': [agent_id, positions[pos]], 'timestep': t, 'location': pos}
                    positions[pos] = agent_id
            
            # 检测边冲突
            for a1 in solution:
                for a2 in solution:
                    if a1 <= a2: continue
                    if t+1 >= len(solution[a1]) or t+1 >= len(solution[a2]): continue
                    if solution[a1][t] == solution[a2][t+1] and solution[a1][t+1] == solution[a2][t]:
                        return {'type': 'edge', 'agents': [a1,a2], 'timestep': t, 'locations': (solution[a1][t], solution[a2][t])}
        return None

    def joint_policy(self, observations):
        """
        重写joint_policy方法，使其执行LNS搜索并返回可执行的第一步动作。
        """
        # 1. 生成初始解
        # --- 最终修改：使用与RL算法完全相同的初始化器 ---
        initial_solution_actions = self.initializer(observations)
        current_solution_paths = _actions_to_paths(initial_solution_actions, self.env)
        # --- 最终修改结束 ---

        # 2. 计算初始解的成本
        best_cost = get_solution_cost(self.env, current_solution_paths)
        best_solution = current_solution_paths.copy()

        # 3. LNS主循环
        for i in range(self.iterations):
            # EECBS冲突解决核心逻辑
            conflict = self._detect_primary_conflict(best_solution)
            if not conflict:
                break
            
            # 构建约束树(CT)
            constraint = self._create_constraint(conflict)
            self.constraint_tree.expand_node(self.constraint_tree.root, constraint)
            
            # 分层路径规划
            new_solution = self._high_level_search(
                best_solution, 
                self.constraint_tree
            )
            if new_solution is None:
                continue
            new_cost = get_solution_cost(self.env, new_solution)

            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost

        # 4. 将最终路径方案转换为第一步的动作
        final_actions = torch.zeros(self.nr_agents, dtype=torch.long)
        for agent_id in range(self.nr_agents):
            current_pos = tuple(self.env.current_positions[agent_id].tolist())
            path = best_solution.get(agent_id)
            action = _get_action_from_path(current_pos, path)
            final_actions[agent_id] = action

        return final_actions

    def train(self):
        pass

    def update_metrics(self, success_rate, completion_rate):
        pass



    def _low_level_search(self, agent_id, constraints, current_path):
        """
        实现基于时空约束的底层路径规划
        """
        start = tuple(self.env.current_positions[agent_id].tolist())
        goal = tuple(self.env.goal_positions[agent_id].tolist())
        
        # 使用改进的时空A*算法
        path = self._space_time_a_star(
            start_node=start,
            goal_node=goal,
            env=self.env,
            reservation_table=constraints,
            time_limit=self.env.time_limit
        )
        return path or current_path

    def _space_time_a_star(self, start_node, goal_node, env, reservation_table, time_limit):
        open_set = []
        heapq.heappush(open_set, (0, start_node, 0, []))
        closed = set()

        while open_set:
            _, current, t, path = heapq.heappop(open_set)
            if (current, t) in closed:
                continue
            
            closed.add((current, t))
            path = path + [current]
            
            if current == goal_node or t >= time_limit:
                return path
            
            for dx, dy in [(0,0), (0,1), (0,-1), (-1,0), (1,0)]:
                neighbor = (current[0]+dx, current[1]+dy)
                if env.obstacle_map[neighbor[0], neighbor[1]]:
                    continue
                if (neighbor, t+1) in reservation_table:
                    continue
                
                h = abs(neighbor[0]-goal_node[0]) + abs(neighbor[1]-goal_node[1])
                heapq.heappush(open_set, (h + t + 1, neighbor, t + 1, path))
        
        return None



def validate_solution(solution, env):
    max_len = max(len(path) for path in solution.values())
    
    for t in range(max_len):
        positions = {}
        # 检查顶点冲突
        for agent_id, path in solution.items():
            if t < len(path):
                pos = path[t]
                if pos in positions:
                    return False
                positions[pos] = agent_id
        
        # 检查边冲突
        for agent_id, path in solution.items():
            if t >= 1 and t < len(path):
                prev = path[t-1]
                curr = path[t]
                for other_id, other_path in solution.items():
                    if agent_id != other_id and t < len(other_path):
                        if other_path[t-1] == curr and other_path[t] == prev:
                            return False
    return True


