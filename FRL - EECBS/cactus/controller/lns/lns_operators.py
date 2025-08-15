# cactus/controller/lns/lns_operators.py
import torch
import random
import numpy as np
import heapq


# (辅助函数 _calculate_collisions 和 _space_time_a_star 保持不变)
def _calculate_collisions(env, solution_paths):
    if not solution_paths: return 0
    num_agents = len(solution_paths)
    max_len = 0
    for path in solution_paths.values():
        if path: max_len = max(max_len, len(path))
    if max_len == 0: return 0
    positions_over_time = {}
    collisions = 0
    for t in range(max_len):
        positions_at_t = {}
        for agent_id, path in solution_paths.items():
            if t < len(path):
                pos = path[t]
                if pos in positions_at_t: collisions += 1
                positions_at_t[pos] = agent_id
        positions_over_time[t] = positions_at_t
    for t in range(max_len - 1):
        for agent1, path1 in solution_paths.items():
            if t + 1 < len(path1):
                for agent2, path2 in solution_paths.items():
                    if agent1 >= agent2: continue
                    if t + 1 < len(path2):
                        if path1[t + 1] == path2[t] and path1[t] == path2[t + 1]: collisions += 1
    return collisions


def _space_time_a_star(start_node, goal_node, env, reservation_table, time_limit):
    open_set = [(0, start_node, 0, [])]
    closed_set = set()
    while open_set:
        f_cost, current_node, g_cost, path = heapq.heappop(open_set)
        time_step = g_cost
        if (current_node, time_step) in closed_set: continue
        path = path + [current_node]
        if current_node == goal_node: return path
        if time_step >= time_limit: continue
        closed_set.add((current_node, time_step))
        for move in [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]:
            neighbor_node = (current_node[0] + move[0], current_node[1] + move[1])
            next_time_step = time_step + 1
            if not (0 <= neighbor_node[0] < env.rows and 0 <= neighbor_node[1] < env.columns and not
            env.obstacle_map[neighbor_node[0]][neighbor_node[1]]): continue
            if reservation_table.get((neighbor_node, next_time_step)): continue
            if reservation_table.get((current_node, next_time_step)) == neighbor_node and reservation_table.get(
                (neighbor_node, time_step)) == current_node: continue
            h_cost = abs(neighbor_node[0] - goal_node[0]) + abs(neighbor_node[1] - goal_node[1])
            new_g_cost = g_cost + 1
            new_f_cost = new_g_cost + h_cost
            heapq.heappush(open_set, (new_f_cost, neighbor_node, new_g_cost, path))
    return None


def get_solution_cost(env, solution_paths):
    total_cost = 0
    num_agents = env.nr_agents
    time_limit = env.time_limit
    collision_penalty_factor = 1000
    for agent_id in range(num_agents):
        path = solution_paths.get(agent_id)
        if path and path[-1] == tuple(env.goal_positions[agent_id].tolist()):
            total_cost += len(path) - 1
        else:
            total_cost += time_limit
    total_collisions = _calculate_collisions(env, solution_paths)
    total_cost += total_collisions * collision_penalty_factor
    return total_cost


def random_removal(current_solution, env, removal_rate=0.2):
    num_to_remove = int(env.nr_agents * removal_rate)
    if num_to_remove == 0: return current_solution.copy(), []
    agents_to_remove = random.sample(list(current_solution.keys()), num_to_remove)
    partial_solution = current_solution.copy()
    for agent_id in agents_to_remove:
        if agent_id in partial_solution: del partial_solution[agent_id]
    random.shuffle(agents_to_remove)
    return partial_solution, agents_to_remove


# --- 新增/修改：实现与LNS2论文一致的修复算子 ---
def prioritized_planning_insertion(partial_solution, agents_to_insert, env):
    """
    优先规划修复算子 (Repair Operator)，与LNS2论文描述一致。
    该算子为待插入的智能体随机分配优先级，然后按顺序依次为它们规划路径。
    """
    new_solution = partial_solution.copy()

    # 为待插入的智能体随机分配一个优先级顺序
    random.shuffle(agents_to_insert)

    # 按优先级顺序依次为智能体规划路径
    for agent_id in agents_to_insert:
        # 1. 为当前智能体构建时空约束表
        # 这个表包含了所有【未被移除】+【已修复好】的智能体的路径
        reservation_table = {}
        max_len = 0
        for path in new_solution.values():
            if path: max_len = max(max_len, len(path))
        for t in range(max_len):
            for existing_agent_id, path in new_solution.items():
                if t < len(path): reservation_table[(path[t], t)] = existing_agent_id

        # 2. 为当前智能体规划路径
        start_node = tuple(env.current_positions[agent_id].tolist())
        goal_node = tuple(env.goal_positions[agent_id].tolist())

        # 使用时空A*算法寻找无碰撞路径
        new_path = _space_time_a_star(
            start_node,
            goal_node,
            env,
            reservation_table,
            env.time_limit
        )

        # 3. 将新路径加入解决方案中，供下一个低优先级智能体避让
        if new_path:
            new_solution[agent_id] = new_path
        else:
            # 如果找不到路径，路径设为空（会导致高成本惩罚）
            new_solution[agent_id] = []

    return new_solution


# 我们将不再使用 greedy_insertion，但暂时保留它以备将来可能需要
greedy_insertion = prioritized_planning_insertion
# --- 修改结束 ---