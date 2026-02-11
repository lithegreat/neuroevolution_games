# Neuroevolution Games

Hands-on projects for the course "Multi-Criteria Optimization and
Decision Analysis." The main experience is Aion's Edge (a Streamlit
strategy game), plus two classic neuroevolution demos.

## 🎮 Projects

### 1. Aion's Edge (Main Game)
A Streamlit colony-strategy game that teaches Linear Programming,
Multi-Objective Optimization, and Voting Theory/MCDA through three
playable levels.

**Features:**
- Level 1: LP survival planning with random events
- Level 2: Pareto-front exploration and Nadir analysis
- Level 3: Voting systems and coalition outcomes

**Run:**
```bash
uv run streamlit run src/app.py
```

### 2. Flappy Bird Evolution
A pure NumPy neural network learns to play Flappy Bird through a
genetic algorithm.

**Features:**
- Custom neural network (3 inputs → 6 hidden → 1 output)
- Genetic algorithm with elitism and mutation
- Real-time visualization with PyGame
- Fast-forward training mode (press Space)

**Run:**
```bash
uv run other_games/flappy_evolution.py
```

### 3. Lunar Lander (NEAT)
Uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve neural
networks for the Gymnasium Lunar Lander environment.

**Features:**
- NEAT topology evolution (nodes + connections)
- Multi-episode fitness evaluation
- Checkpoint saving every 5 generations
- Automated demo visualization after training

**Run:**
```bash
# Train
uv run other_games/lunar_neat.py

# Or resume from checkpoint
uv run python -c "import neat; p = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-50')"
```

## 📦 Installation

**Requirements:** Python 3.14+

```bash
uv sync
```

Or manually install dependencies:
```bash
uv pip install gymnasium[box2d] neat-python numpy pygame pyyaml scipy streamlit
```

## 🧬 How It Works

### Aion's Edge Math Engine
The core logic lives in `src/OptimizationEngine.py`:
- `LPSolver`: linear programming for survival planning
- `MOOSolver`: Pareto-front detection and Nadir analysis
- `VotingSystem`: voting rules for MCDA decisions

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
│   ├── app.py                 # Aion's Edge (Streamlit app)
│   └── OptimizationEngine.py  # LP, MOO, and voting solvers
├── other_games/
│   ├── flappy_evolution.py    # Flappy Bird with custom GA
│   └── lunar_neat.py          # Lunar Lander with NEAT
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
