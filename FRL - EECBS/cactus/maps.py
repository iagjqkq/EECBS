
from cactus.env.primal_gridworld import PRIMALGridWorld
from cactus.utils import get_param_or_default, assertEquals, assertContains
from cactus.constants import *
from os.path import join
import random
import cactus.env.env_generator as env_generator


def load_obstacles(filename):
    with open(filename, "r+") as file:
        lines = file.readlines()
        obstacles = []
        width = len(lines)
        for x, line in enumerate(lines[4:]):
            obstacle_line = []
            for y, character in enumerate(line.strip()):
                obstacle_line.append(character == "@")
            obstacles.append(obstacle_line)
            height = len(line.strip())
        return obstacles, width, height
        
def make_test_map(params):
    assertContains(params, MAP_NAME)
    map_name = params[MAP_NAME]
    assert map_name.startswith("primal-"), f"Invalid test map name {map_name}"
    map_name = map_name.replace("primal-", "")
    filename = join("instances", "primal_test_envs", f"{map_name}.npy")
    params[ENV_PRIMAL_MAP] = numpy.load(filename)
    params[TORCH_DEVICE] = torch.device("cpu")
    return PRIMALGridWorld(params)

def make(params):
    if MAP_NAME in params:
        from .env.mapf_gridworld import MAPFGridworld
        obstacles, width, height = load_obstacles(join('instances', f'{params[MAP_NAME]}.map'))
        params.update({
            ENV_OBSTACLES: obstacles,
            ENV_WIDTH: width,
            ENV_HEIGHT: height,
            TORCH_DEVICE: torch.device("cpu")
        })
        
        env = MAPFGridworld(params)
    return env_generator.generate_mapf_gridworld(params[ENV_NR_AGENTS], width, 0.3, params, 
        window_width=params.get(WINDOW_WIDTH, 800), 
        window_height=params.get(WINDOW_HEIGHT, 600))