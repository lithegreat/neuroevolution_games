import pygame
import numpy as np
import random
import sys
import copy

# ==========================================
# 1. 配置参数
# ==========================================
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
POPULATION_SIZE = 50   # 每一代有多少只鸟
MUTATION_RATE = 0.1    # 变异概率
MUTATION_SCALE = 0.5   # 变异强度
FPS = 60
BIRD_X = 50            # 鸟的固定X坐标

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
BLUE = (50, 50, 255)

# ==========================================
# 2. 神经网络 (大脑) - 纯 Numpy 实现
# ==========================================
class NeuralNetwork:
    def __init__(self):
        # 输入层: 3个神经元 
        # [鸟的高度, 鸟的垂直速度, 距离下一个管子的水平距离, 管子空隙的垂直位置] -> 这里简化为3个主要特征
        self.input_size = 3  
        self.hidden_size = 6
        self.output_size = 1

        # 随机初始化权重 (没有梯度下降，不需要requires_grad)
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.random.randn(self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.random.randn(self.output_size)

    def forward(self, inputs):
        # 简单的全连接前向传播
        # Layer 1
        z1 = np.dot(inputs, self.w1) + self.b1
        a1 = np.tanh(z1) # 激活函数
        # Layer 2
        z2 = np.dot(a1, self.w2) + self.b2
        # Numerically stable sigmoid to prevent overflow
        a2 = np.where(z2 >= 0, 
                      1.0 / (1.0 + np.exp(-z2)), 
                      np.exp(z2) / (1.0 + np.exp(z2)))
        return a2

    def copy_and_mutate(self):
        # 克隆一个新的大脑
        child = NeuralNetwork()
        child.w1 = self.w1.copy()
        child.b1 = self.b1.copy()
        child.w2 = self.w2.copy()
        child.b2 = self.b2.copy()

        # 变异操作: 随机修改部分权重
        # 这就是进化的核心！不是求导，而是随机扰动
        for param in [child.w1, child.b1, child.w2, child.b2]:
            mask = np.random.rand(*param.shape) < MUTATION_RATE
            noise = np.random.randn(*param.shape) * MUTATION_SCALE
            param[mask] += noise[mask]
        
        return child

# ==========================================
# 3. 游戏实体 (鸟和管子)
# ==========================================
class Bird:
    def __init__(self, brain=None):
        self.y = SCREEN_HEIGHT / 2
        self.velocity = 0
        self.gravity = 0.6
        self.lift = -10
        self.brain = brain if brain else NeuralNetwork()
        self.alive = True
        self.score = 0     # 存活时间
        self.color = (random.randint(50, 200), random.randint(50, 200), 255)

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity
        self.score += 1 # 活得越久分越高

    def jump(self):
        self.velocity = self.lift

    def think(self, pipe):
        # 获取输入特征并归一化 (让数值在 0-1 之间，利于NN处理)
        # 输入1: 鸟的高度 (归一化)
        input1 = self.y / SCREEN_HEIGHT
        # 输入2: 距离管子顶部的差值 (关键特征)
        input2 = (pipe.top_height + pipe.gap_size / 2 - self.y) / SCREEN_HEIGHT
        # 输入3: 距离管子的水平距离
        input3 = (pipe.x - BIRD_X) / SCREEN_WIDTH

        inputs = np.array([input1, input2, input3])
        
        # 神经网络决策
        output = self.brain.forward(inputs)
        
        # 如果输出 > 0.5 就跳
        if output > 0.5:
            self.jump()

class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.width = 50
        self.speed = 3
        self.gap_size = 150
        self.top_height = random.randint(50, SCREEN_HEIGHT - self.gap_size - 50)

    def update(self):
        self.x -= self.speed

    def draw(self, screen):
        # 上管子
        pygame.draw.rect(screen, (0, 200, 0), (self.x, 0, self.width, self.top_height))
        # 下管子
        bottom_y = self.top_height + self.gap_size
        pygame.draw.rect(screen, (0, 200, 0), (self.x, bottom_y, self.width, SCREEN_HEIGHT - bottom_y))

# ==========================================
# 4. 主程序 (进化循环)
# ==========================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    # 初始化种群
    birds = [Bird() for _ in range(POPULATION_SIZE)]
    generation = 1
    
    # 记录历史最高分
    best_all_time = 0
    
    # 快进模式标志
    fast_forward = False

    while True:
        # 重置游戏状态
        pipes = [Pipe()]
        pipes_passed = 0
        
        # 本轮游戏循环
        while any(bird.alive for bird in birds):
            screen.fill(BLACK)
            
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        fast_forward = not fast_forward

            # --- 1. 更新管子 ---
            if pipes[-1].x < SCREEN_WIDTH - 200:
                pipes.append(Pipe())
            
            for pipe in pipes:
                pipe.update()
                pipe.draw(screen)
            
            # 移除屏幕外的管子 (安全的列表过滤，不在迭代时修改)
            pipes = [pipe for pipe in pipes if pipe.x + pipe.width >= 0]

            # 获取最近的管子 (用于输入给神经网络)
            if not pipes:  # 防止空列表导致崩溃
                pipes.append(Pipe())
            closest_pipe = pipes[0]
            if closest_pipe.x + closest_pipe.width < BIRD_X:
                if len(pipes) > 1:
                    closest_pipe = pipes[1]

            # --- 2. 更新鸟 ---
            for bird in birds:
                if bird.alive:
                    # AI 思考
                    bird.think(closest_pipe)
                    # 物理更新
                    bird.update()
                    # 绘制
                    pygame.draw.circle(screen, bird.color, (BIRD_X, int(bird.y)), 10)

                    # --- 碰撞检测 ---
                    # 撞天花板或地板
                    if bird.y < 0 or bird.y > SCREEN_HEIGHT:
                        bird.alive = False
                    
                    # 撞管子
                    # 简单的矩形碰撞逻辑
                    bird_rect = pygame.Rect(BIRD_X - 10, bird.y - 10, 20, 20)
                    top_rect = pygame.Rect(closest_pipe.x, 0, closest_pipe.width, closest_pipe.top_height)
                    bottom_rect = pygame.Rect(closest_pipe.x, closest_pipe.top_height + closest_pipe.gap_size, closest_pipe.width, SCREEN_HEIGHT)
                    
                    if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                        bird.alive = False

            # --- UI 信息 ---
            alive_count = sum(1 for b in birds if b.alive)
            text = font.render(f"Gen: {generation} | Alive: {alive_count}", True, WHITE)
            screen.blit(text, (10, 10))
            
            pygame.display.flip()
            if not fast_forward:
                clock.tick(FPS)  # 按空格键切换快进模式

        # ==========================================
        # 进化核心: 自然选择
        # ==========================================
        
        # 1. 找到这代最强的鸟 (Fitness = score)
        birds.sort(key=lambda b: b.score, reverse=True)
        best_bird = birds[0]
        
        print(f"Generation {generation} finished. Best Score: {best_bird.score}")
        
        if best_bird.score > best_all_time:
            best_all_time = best_bird.score
        
        # 2. 繁殖下一代
        new_birds = []
        
        # 精英策略 (Elitism): 直接保留这一代最好的鸟，不修改，防止退化
        elite_brain = copy.deepcopy(best_bird.brain)  # 真正的精英保留，不变异
        new_birds.append(Bird(elite_brain))

        # 剩下的位置，全部由最好的那几只鸟变异产生
        # 我们只取前 5 名进行繁衍
        top_performers = birds[:5]
        
        while len(new_birds) < POPULATION_SIZE:
            parent = random.choice(top_performers)
            # 变异产生后代
            child_brain = parent.brain.copy_and_mutate()
            new_birds.append(Bird(child_brain))

        # 更新种群
        birds = new_birds
        generation += 1

if __name__ == "__main__":
    main()