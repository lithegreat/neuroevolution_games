import gymnasium as gym
import neat
import os
import numpy as np
import pickle

# ==========================================
# 1. 定义核心：如何评估一个“大脑”的好坏
# ==========================================
def eval_genomes(genomes, config):
    """
    Fitness Function (适应度函数)
    这里模拟并行的嵌入式控制器测试。
    """
    
    # 创建环境 (为了加速训练，这里不开启渲染模式 'human')
    # 如果电脑强劲，可以在这里并行化 (ParallelEvaluator)
    env = gym.make("LunarLander-v3")

    for genome_id, genome in genomes:
        # 1. 创建神经网络 (从基因生成表型)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # 2. 初始化环境
        observation, info = env.reset()
        genome.fitness = 0
        
        # 3. 开始模拟着陆
        for _ in range(600): # 限制最大步数，模拟看门狗定时器 (Watchdog Timer)
            # 神经网络决策
            output = net.activate(observation)
            
            # LunarLander 需要离散动作: 0:不喷射, 1:左引擎, 2:主引擎, 3:右引擎
            # 输出层有4个节点，取最大值的索引作为动作
            action = np.argmax(output)
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            # --- 多目标优化的体现 ---
            # Gym 的原始 reward 已经包含了：
            # +100/140: 成功着陆/腿部触地
            # -0.3/frame: 引擎喷射惩罚 (即能耗惩罚)
            # -100: 坠毁
            genome.fitness += reward
            
            if terminated or truncated:
                break
                
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

    # --- 开始进化 ---
    # 运行 50 代 (Generations)
    # eval_genomes 是我们需要自己写的函数
    winner = p.run(eval_genomes, 50)

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
def run_demo(genome, config):
    print("\n--- Starting Demo with Best Model ---")
    # 这次开启 render_mode="human" 这样你能看到动画
    env = gym.make("LunarLander-v3", render_mode="human")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    observation, info = env.reset()
    total_reward = 0
    
    while True:
        output = net.activate(observation)
        action = np.argmax(output)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Demo Finished. Total Reward: {total_reward}")
            break
    env.close()

# ==========================================
# 4. 辅助：自动生成配置文件 (完整修复版)
# ==========================================
def create_config_file(config_output_path):
    """
    NEAT 需要一个配置文件。我们把它写入到指定的路径中。
    """
    config_content = """
[NEAT]
fitness_criterion      = max
fitness_threshold      = 200
pop_size               = 50
reset_on_extinction    = False
no_fitness_termination = False

[DefaultGenome]
# Node activation options
activation_default       = tanh
activation_mutate_rate   = 0.0
activation_options       = tanh

# Node aggregation options
aggregation_default      = sum
aggregation_mutate_rate  = 0.0
aggregation_options      = sum

# Node bias options
bias_init_mean           = 0.0
bias_init_stdev          = 1.0
bias_init_type           = gaussian
bias_max_value           = 30.0
bias_min_value           = -30.0
bias_mutate_power        = 0.5
bias_mutate_rate         = 0.7
bias_replace_rate        = 0.1

# Node response options
response_init_mean       = 1.0
response_init_stdev      = 0.0
response_init_type       = gaussian
response_max_value       = 30.0
response_min_value       = -30.0
response_mutate_power    = 0.0
response_mutate_rate     = 0.0
response_replace_rate    = 0.0

# Genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# Connection add/remove rates
conn_add_prob            = 0.5
conn_delete_prob         = 0.5

# Connection enable options
enabled_default          = True
enabled_mutate_rate      = 0.01
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0

# Feedforward network options
feed_forward             = True
initial_connection       = full

# Node add/remove rates
node_add_prob            = 0.2
node_delete_prob         = 0.2

# Network parameters
num_hidden               = 0
num_inputs               = 8
num_outputs              = 4

# Connection weight options
weight_init_mean         = 0.0
weight_init_stdev        = 1.0
weight_init_type         = gaussian
weight_max_value         = 30
weight_min_value         = -30
weight_mutate_power      = 0.5
weight_mutate_rate       = 0.8
weight_replace_rate      = 0.1

# --- 修复点：添加 structural_mutation_surer ---
single_structural_mutation = False
structural_mutation_surer  = default

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 1
    """
    with open(config_output_path, 'w') as f:
        f.write(config_content)

# ==========================================
# 主入口
# ==========================================
if __name__ == '__main__':
    # 1. 确定配置文件的绝对路径
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(local_dir, '..', 'config')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'config-feedforward.txt')

    # 2. 生成配置 (传入路径)
    create_config_file(config_path)

    # 3. 训练
    winner, config = run_neat(config_path)

    # 4. 演示冠军
    run_demo(winner, config)