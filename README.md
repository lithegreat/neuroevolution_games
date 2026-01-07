# Neuroevolution Games

Neuroevolution experiments using genetic algorithms and NEAT to train AI agents for classic games: Flappy Bird and Lunar Lander.

## 🎮 Projects

### 1. Flappy Bird Evolution
A pure NumPy neural network implementation that learns to play Flappy Bird through genetic algorithm-based evolution.

**Features:**
- Custom neural network (3 inputs → 6 hidden → 1 output)
- Genetic algorithm with elitism and mutation
- Real-time visualization with PyGame
- Fast-forward training mode (press Space)

**Architecture:**
- **Inputs:** Bird height, distance to pipe gap, horizontal distance to pipe
- **Selection:** Top 5 performers breed the next generation
- **Mutation:** Random weight perturbation (no gradient descent)

**Run:**
```bash
uv run src/flappy_evolution.py
```

### 2. Lunar Lander (NEAT)
Uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve neural networks for the OpenAI Gymnasium Lunar Lander environment.

**Features:**
- NEAT topology evolution (nodes + connections)
- Multi-episode fitness evaluation
- Checkpoint saving every 5 generations
- Automated demo visualization after training

**Run:**
```bash
# Train
uv run src/lunar_neat.py

# Or resume from checkpoint
uv run python -c "import neat, pickle; p = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-50'); # ... continue training"
```

## 📦 Installation

**Requirements:** Python 3.14+

```bash
uv sync
```

Or manually install dependencies:
```bash
uv pip install gymnasium[box2d] neat-python numpy pygame
```

## 🧬 How It Works

### Genetic Algorithm (Flappy Bird)
1. **Initialize** population with random neural networks
2. **Evaluate** fitness (survival time)
3. **Select** top performers
4. **Reproduce** via mutation
5. **Repeat** until convergence

### NEAT (Lunar Lander)
- Evolves both **network topology** and **weights**
- Uses **speciation** to protect innovation
- **Crossover** between similar networks
- Solves multi-objective optimization (landing success + fuel efficiency)

## 📊 Project Structure

```
neuroevolution_games/
├── src/
│   ├── flappy_evolution.py  # Flappy Bird with custom GA
│   └── lunar_neat.py         # Lunar Lander with NEAT
├── config/
│   └── config-feedforward.yaml  # NEAT configuration
├── checkpoints/              # Training checkpoints (auto-saved)
└── pyproject.toml            # Dependencies
```

## 🎯 Training Tips

**Flappy Bird:**
- Press **Space** to toggle fast-forward mode
- Typical convergence: 50-200 generations
- Watch diversity in bird colors

**Lunar Lander:**
- Training saves checkpoints every 5 generations in `checkpoints/`
- Success threshold: fitness > 200
- Demo runs automatically after training

## 🔧 Configuration

Edit [config/config-feedforward.yaml](config/config-feedforward.yaml) to tune NEAT parameters:
- `pop_size`: Population size (default: 100)
- `fitness_threshold`: Success criteria (default: 200)
- `conn_add_prob`: Connection mutation rate
- `bias_mutate_rate`: Weight mutation rate

## 📝 License

MIT

## 🤝 Contributing

Feel free to open issues or PRs for improvements!
