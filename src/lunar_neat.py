import os
import sys
import pickle

# 依赖检查
try:
    import gymnasium as gym
    import neat
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}. Install with: pip install gymnasium neat-python numpy")
    sys.exit(1)

# ==========================================
# 配置常量
# ==========================================
MAX_STEPS = 1000        # 每个episode的最大步数
NUM_EVAL_EPISODES = 3   # 用于评估fitness的episode数量
NUM_GENERATIONS = 100   # 进化代数 (增加到100代以提高成功率)

# ==========================================
# 1. 定义核心：如何评估一个“大脑”的好坏
# ==========================================
def eval_genomes(genomes, config):
    """
    Fitness Function (适应度函数)
    使用多episode评估来获得更稳定的fitness值。
    """
    
    # 创建环境 (为了加速训练，这里不开启渲染模式 'human')
    env = gym.make("LunarLander-v3")

    for genome_id, genome in genomes:
        # 1. 创建神经网络 (从基因生成表型)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # 2. 多episode评估，取平均值以获得更稳定的fitness
        total_fitness = 0
        
        for episode in range(NUM_EVAL_EPISODES):
            observation, info = env.reset()
            episode_reward = 0
            
            # 3. 开始模拟着陆
            for _ in range(MAX_STEPS):  # 限制最大步数
                # 神经网络决策
                output = net.activate(observation)
                
                # LunarLander 需要离散动作: 0:不喷射, 1:左引擎, 2:主引擎, 3:右引擎
                action = np.argmax(output)
                
                # 执行动作
                observation, reward, terminated, truncated, info = env.step(action)
                
                # --- 多目标优化的体现 ---
                # Gym 的原始 reward 已经包含了：
                # +100/140: 成功着陆/腿部触地
                # -0.3/frame: 引擎喷射惩罚 (即能耗惩罚)
                # -100: 坠毁
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_fitness += episode_reward
        
        # 取平均值作为最终fitness
        genome.fitness = total_fitness / NUM_EVAL_EPISODES
                
    env.close()

# ==========================================
# 2. 运行训练主循环
# ==========================================
def run_neat(config_file):
    # 加载配置
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # 创建种群
    p = neat.Population(config)

    # 添加统计报告 (在终端打印进度)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # 添加检查点保存 (每5代保存一次，防止训练中断丢失进度)
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(checkpoint_dir, 'neat-checkpoint-')))

    # --- 开始进化 ---
    winner = p.run(eval_genomes, NUM_GENERATIONS)

    # 输出最优结果
    print('\nBest genome:\n{!s}'.format(winner))
    
    # 保存冠军模型
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'winner_model.pkl')
    with open(model_path, "wb") as f:
        pickle.dump(winner, f)
    print(f'Model saved to: {model_path}')
        
    return winner, config

# ==========================================
# 3. 可视化演示 (Demo)
# ==========================================
def run_demo(genome, config, num_episodes=3):
    """
    可视化演示最佳模型，支持多次回放。
    """
    print(f"\n--- Starting Demo with Best Model ({num_episodes} episodes) ---")
    # 这次开启 render_mode="human" 这样你能看到动画
    env = gym.make("LunarLander-v3", render_mode="human")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    for ep in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        
        while True:
            output = net.activate(observation)
            action = np.argmax(output)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"Episode {ep + 1}/{num_episodes} Finished. Reward: {total_reward:.2f}")
                break
    
    env.close()
    print("--- Demo Complete ---")

# ==========================================
# 主入口
# ==========================================
if __name__ == '__main__':
    # 配置文件路径 (静态文件，位于 config/ 目录)
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, '..', 'config', 'config-feedforward.txt')
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # 训练
    winner, config = run_neat(config_path)

    # 演示冠军
    run_demo(winner, config)