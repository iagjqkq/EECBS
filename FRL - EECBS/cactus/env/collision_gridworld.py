from cactus.env.gridworld import GridWorld
from cactus.constants import *

"""
 Represents a navigation task in a 2D grid world environment with collision detection.
"""
class CollisionGridWorld(GridWorld):

    def __init__(self, params) -> None:
        super(CollisionGridWorld, self).__init__(params)
        self.agent_ids = self.as_int_tensor([i for i in range(self.nr_agents)])
        self.vertex_collision_buffer = self.bool_ones(self.nr_agents)
        self.edge_collision_buffer = self.bool_ones(self.nr_agents)

    def move_condition(self, new_positions):
        # 检查目标到达状态
        goal_reached = torch.all(self.current_positions == self.goal_positions, dim=1)
        condition = goal_reached.clone()
        
        # 仅对未到达目标的agent进行碰撞检测
        active_agents = ~goal_reached
        if active_agents.any():
            self.current_position_map[:] = -1.0
            active_condition, _ = super(CollisionGridWorld, self).move_condition(new_positions[active_agents])
            # 使用完整维度条件赋值
            active_indices = active_agents.nonzero(as_tuple=True)[0]
            condition[active_indices] = active_condition[torch.arange(len(active_indices)), 0] & active_condition[torch.arange(len(active_indices)), 1]
        self.edge_collision_buffer.fill_(False)
        self.vertex_collision_buffer.fill_(False)
        x0 = self.current_positions[:,0]
        y0 = self.current_positions[:,1]
        self.current_position_map[x0,y0] = self.agent_ids
        x1 = torch.where(condition, new_positions[:,0].clamp(0, self.rows-1), self.current_positions[:,0].clamp(0, self.rows-1))
        y1 = torch.where(condition, new_positions[:,1].clamp(0, self.columns-1), self.current_positions[:,1].clamp(0, self.columns-1))
        self.next_position_map[x1,y1] = self.agent_ids
        self.vertex_collision_buffer[:] = (self.next_position_map[x1,y1] != self.agent_ids)
        other_origins = -self.int_ones(self.nr_agents)
        other_origins = self.current_position_map[x1,y1]
        occupied = other_origins >= 0
        filter_condition = condition[other_origins.clamp(0, self.nr_agents)]
        other_origins = torch.where(torch.logical_and(filter_condition, occupied), other_origins, -1)
        occupied = other_origins >= 0
        not_same = other_origins != self.agent_ids
        edge_condition = torch.logical_and(occupied, not_same)
        indices = other_origins[edge_condition]
        x = new_positions[indices,0]
        y = new_positions[indices,1]
        if edge_condition.any():
            self.edge_collision_buffer[edge_condition] = self.current_position_map[x,y] == self.agent_ids[edge_condition]
        no_collisions = torch.logical_not(torch.logical_or(self.vertex_collision_buffer, self.edge_collision_buffer))
        condition = torch.logical_and(condition, no_collisions)
        active_count = active_agents.sum().item()
        full_vertex_collisions = self.vertex_collision_buffer & active_agents
        full_edge_collisions = self.edge_collision_buffer & active_agents
        return condition.unsqueeze(1).expand(self.nr_agents, ENV_2D), (full_vertex_collisions, full_edge_collisions)

    def reset(self):
        self.current_position_map[:] = -1.0
        return super(CollisionGridWorld, self).reset()