# cactus/algorithms.py

from cactus.controller.controller import Controller
from cactus.utils import get_param_or_default
from cactus.constants import *

# --- 修改：只导入传统LNS控制器 ---
from cactus.controller.lns.eecbs_controller import EnhancedECBSController


def make(params):
    """
    简化的make函数，只创建TraditionalLNSController。
    """
    algorithm_name = get_param_or_default(params, ALGORITHM_NAME, DEFAULT_ALGORITHM)

    if algorithm_name == ALGORITHM_EECBS:
        return EnhancedECBSController(params, env=params.get('env'))

    # 保留一个随机算法作为备用
    if algorithm_name == ALGORITHM_RANDOM:
        return Controller(params)

    raise ValueError(
        f"Unknown or unsupported algorithm: '{algorithm_name}'. This project is configured to only run Enhanced_ECBS.")


def get_algorithm_config(algorithm_name):
    """
    简化的配置函数，只返回TraditionalLNSController的配置。
    """
    if algorithm_name == ALGORITHM_EECBS:
        # 对于Enhanced ECBS，后面两个返回值没有意义，可以返回None
        return EnhancedECBSController, None, None

    raise ValueError(
        f"Unknown or unsupported algorithm: '{algorithm_name}'. This project is configured to only run Traditional_LNS.")