import os
import pygame

BLACK = (0, 0, 0)
DARK_GRAY = (125, 125, 125)
GRAY = (175, 175, 175)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 150, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (51, 226, 253)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
MAROON = (128, 0, 0)
CYAN = (0, 255, 255)
TEAL = (0, 128, 128)
PURPLE = (128, 0, 128)

AGENT_COLORS = [RED, BLUE, ORANGE, MAGENTA, PURPLE, TEAL, MAROON, GREEN, DARK_GRAY, CYAN]

class GridworldViewer:
    def __init__(self, width, height, cell_size=10, fps=30):
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid window dimensions: {width}x{height}. Must be positive integers.")
        if cell_size <= 0:
            raise ValueError(f"Invalid cell_size: {cell_size}. Must be positive integer.")
        os.environ['SDL_VIDEODRIVER'] = 'windib'
        try:
            pygame.init()
            pygame.display.init()  # 显式初始化视频子系统
        except pygame.error as e:
            raise RuntimeError(f"视频系统初始化失败: {e}\n当前SDL驱动: {os.environ.get('SDL_VIDEODRIVER', '未设置')}\n请检查显卡驱动或尝试更换SDL_VIDEODRIVER环境变量")
        finally:
            # 调试信息输出
            print(f"[DEBUG] 当前生效的SDL视频驱动: {os.environ.get('SDL_VIDEODRIVER')}")
        try:
            display_info = pygame.display.Info()
        except pygame.error as e:
            raise RuntimeError(f"无法获取显示器信息: {e}\n视频子系统可能未正确初始化")
        max_width = display_info.current_w
        max_height = display_info.current_h
        
        self.cell_size = cell_size
        self.width = min(width, max_width)
        self.height = min(height, max_height)
        
        if width != self.width or height != self.height:
            print(f"[WARNING] 窗口尺寸自动调整为 {self.width}x{self.height} (原始尺寸 {width}x{height} 超出显示器分辨率 {max_width}x{max_height})")
        print(f"[DEBUG] 窗口尺寸: {self.width}x{self.height}")
        self.clock = pygame.time.Clock()
        self.fps = fps
        pygame.display.set_caption("MAPF Environment")
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        # 强制设置DPI缩放
        try:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        except pygame.error as e:
            raise RuntimeError(f"Failed to initialize pygame display: {e}\nCurrent resolution: {self.width}x{self.height}")

    def agent_color(self, agent_id):
        nr_colors = len(AGENT_COLORS)
        return AGENT_COLORS[agent_id%nr_colors]
              
    def draw_state(self, env):
        self.screen.fill(BLACK)
        for x in range(env.rows):
            for y in range(env.columns):
                if not env.obstacle_map[x][y]:
                    self.draw_pixel(x, y, WHITE)
                agent_id = env.occupied_goal_positions[x][y]
                if agent_id >= 0:
                    self.draw_pixel(x, y, self.agent_color(agent_id))
                agent_id = env.current_position_map[x][y]
                if agent_id >= 0:
                    # 检查是否已完成目标
                    if env.actual_goal_reached[agent_id]:
                        # 已完成目标使用半透明颜色
                        color = self.agent_color(agent_id)
                        color = (color[0], color[1], color[2], 128)
                        self.draw_circle(x, y, color)
                    else:
                        self.draw_circle(x, y, self.agent_color(agent_id))
        pygame.display.flip()
        self.clock.tick(self.fps)
        return self.check_for_interrupt()
    
    def draw_pixel(self, x, y, color):
        pygame.draw.rect(self.screen, color,
                        pygame.Rect(
                            x * self.cell_size+1,
                            y * self.cell_size+1,
                            self.cell_size-2,
                            self.cell_size-2),
                        0)
    
    def draw_circle(self, x, y, color):
        radius = int(self.cell_size/2)
        center_x = x * self.cell_size + radius
        center_y = y * self.cell_size + radius
        center = (center_x, center_y)
        pygame.draw.circle(self.screen, color, center, radius-2)

    def check_for_interrupt(self):
        key_state = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or key_state[pygame.K_ESCAPE]:
                return True
        return False

    def close(self):
        pygame.quit()

def render(env, viewer):
    if viewer is None:
        viewer = GridworldViewer(env.columns * 10, env.rows * 10, cell_size=10)
    viewer.draw_state(env)
    return viewer
