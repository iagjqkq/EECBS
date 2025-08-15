import numpy
import random
import torch

def detect_conflicts(paths):
    conflicts = []
    max_len = max(len(p) for p in paths.values())
    
    for t in range(max_len):
        positions = {}
        # 检测顶点冲突
        for agent, path in paths.items():
            if t < len(path):
                pos = tuple(path[t])
                if pos in positions:
                    conflicts.append(('vertex', positions[pos], agent, pos, t))
                positions[pos] = agent
        
        # 检测边冲突
        for agent, path in paths.items():
            if t >= 1 and t < len(path):
                prev = tuple(path[t-1])
                curr = tuple(path[t])
                for other_agent, other_path in paths.items():
                    if agent != other_agent and t < len(other_path):
                        if tuple(other_path[t-1]) == curr and tuple(other_path[t]) == prev:
                            conflicts.append(('edge', agent, other_agent, prev, curr, t))
    return conflicts


class ConstraintTree:
    def __init__(self, max_depth):
        self.root = {
            'constraints': [],
            'solution': None,
            'cost': float('inf'),
            'children': []
        }
        self.max_depth = max_depth

    def add_constraint(self, agent_id, position, timestep):
        new_constraint = (agent_id, position, timestep)
        if new_constraint not in self.root['constraints']:
            self.root['constraints'].append(new_constraint)

    def find_conflicts(self, paths):
        conflicts = []
        for t in range(max(len(p) for p in paths.values())):
            positions = {}
            for agent, path in paths.items():
                if t < len(path):
                    pos = tuple(path[t])
                    if pos in positions:
                        conflicts.append(('vertex', positions[pos], agent, pos, t))
                    positions[pos] = agent
            # 边冲突检测逻辑
            for agent1, path1 in paths.items():
                if t >= 1 and t < len(path1):
                    prev = tuple(path1[t-1])
                    curr = tuple(path1[t])
                    for agent2, path2 in paths.items():
                        if agent1 != agent2 and t < len(path2):
                            if tuple(path2[t-1]) == curr and tuple(path2[t]) == prev:
                                conflicts.append(('edge', agent1, agent2, prev, curr, t))
        return conflicts

def assertTensorEquals(first, second):
    assert (first == second).all(), "Expected {}, got {}".format(first, second)

def assertEquals(first, second):
    assert first == second, "Expected {}, got {}".format(first, second)

def assertTrue(first):
    assert first, "Expected to be True"

def assertFalse(first):
    assert not first, "Expected to be False"

def assertAtLeast(first, second):
    assert first >= second, "Expected {} >= {}".format(first, second)

def assertGreater(first, second):
    assert first > second, "Expected {} > {}".format(first, second)

def assertAtMost(first, second):
    assert first <= second, "Expected {} <= {}".format(first, second)

def assertLess(first, second):
    assert first < second, "Expected {} < {}".format(first, second)

def assertContains(collection, element):
    assert element in collection, "Expected element in {}, got {}".format(collection, element)

def get_param_or_default(params, label, default_value=None):
    if label in params:
        return params[label]
    else:
        return default_value

def argmax(values):
    max_value = max(values)
    default_index = numpy.argmax(values)
    candidate_indices = []
    for i,value in enumerate(values):
        if value >= max_value:
            candidate_indices.append(i)
    if not candidate_indices:
        return default_index
    return random.choice(candidate_indices)


def monitor_metrics(metrics_list):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            for metric in metrics_list:
                value = getattr(self, metric, None)
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        self.memory.buffer.record_metric(metric, value.detach().cpu().numpy())
                    else:
                        self.memory.buffer.record_metric(metric, numpy.array(value))
            return result
        return wrapper
    return decorator