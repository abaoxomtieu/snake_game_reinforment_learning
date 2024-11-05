
# Snake Game with Reinforcement Learning (Q-learning with PyTorch)

This project implements a classic **Snake Game** using **Pygame** and applies **Q-learning** (a reinforcement learning technique) to train the snake agent to play the game autonomously. The Q-learning algorithm is built using **PyTorch** for neural network modeling and policy updates.

## Table of Contents
- [Overview](#overview)
- [Technologies](#technologies)
- [Installation](#installation)
- [Game Rules](#game-rules)
- [Reinforcement Learning Approach](#reinforcement-learning-approach)
  - [Q-Learning](#q-learning)
  - [Environment](#environment)
  - [State Representation](#state-representation)
  - [Rewards](#rewards)
  - [Action Space](#action-space)
- [Training](#training)
- [Running the Game](#running-the-game)
- [Customization](#customization)
- [License](#license)

## Overview
This project showcases how reinforcement learning can be applied to teach an agent (the snake) to play a Snake game. The project uses **Q-learning** to make the agent learn the best actions to take in various states in order to maximize the game score, which is determined by the length of the snake.

The goal is for the snake to learn to move towards food, avoid hitting the walls, and avoid colliding with itself.

## Technologies
- **Python** 3.8+
- **PyTorch** (for Q-learning neural network)
- **Pygame** (for the Snake game environment)
- **Numpy** (for mathematical operations)
- **Matplotlib** (for plotting results, if needed)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/abaoxomtieu/snake_game_reinforment_learning.git
cd snake_game_reinforment_learning
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

Requirements in `requirements.txt`:
```txt
pygame
torch
numpy
matplotlib
```

## Game Rules
- The snake moves in 4 directions (up, down, left, right).
- The objective is to eat as much food as possible.
- The snake grows when it eats food.
- The game ends if the snake collides with the walls or itself.

## Reinforcement Learning Approach

### Q-Learning
The agent learns via **Q-learning**, which is a value-based reinforcement learning algorithm. The Q-values (quality of actions) are stored in a table (Q-table) and updated based on the agent’s interaction with the environment. However, to deal with the complexity of the environment (infinite possible states), we use a **neural network** to approximate the Q-function.

The main components of the learning algorithm include:

- **State**: Representation of the environment.
- **Action**: The possible moves the agent can make.
- **Reward**: A scalar feedback that the agent receives from the environment.
- **Q-value update**: The agent updates its knowledge using the Bellman equation and learns over time through exploration and exploitation.

### Environment
The snake's world consists of:
- **The Snake**: Controlled by the agent.
- **Food**: Randomly placed on the grid. The snake is rewarded for eating food.
- **Boundaries and Obstacles**: The snake dies if it hits the wall or its own body.

### State Representation
The state is represented by a vector capturing information about:
- The position of the food relative to the snake.
- The current direction of movement.
- Potential collisions in the snake’s path (with the wall or its body).

### Rewards
- **+10** for eating food.
- **-10** for colliding with the wall or itself.
- **0** normal move.

### Action Space
The possible actions for the snake are:
- Move **left**.
- Move **right**.
- Move **forward**.

## Training
The training loop includes the following key steps:
1. **Initialize Q-values** (using a neural network as a function approximator).
2. For each episode:
   - Get the current **state**.
   - Select an **action** based on an ε-greedy policy.
   - Execute the action and observe the **reward** and **next state**.
   - **Update** the Q-values (neural network weights) using the Bellman equation.
   - Update the state to the next state.
3. The agent learns by maximizing the **cumulative reward**.

To start training, run:

```bash
python agent.py
```

During training, you can observe the agent’s performance and the score it achieves.

## Running the Game
Once the agent is trained, you can run the game to see the agent in action:

```bash
python play_game.py
```

This will load the trained model and start the Snake game with the AI agent playing it.

## Customization
You can adjust the game and training parameters by modifying the following in the code:
- **Grid size**: Customize the size of the game grid.
- **Training hyperparameters**: Adjust the learning rate, discount factor, epsilon for exploration-exploitation balance, and the number of episodes.
- **Neural network architecture**: Modify the layers in the Q-network to experiment with different architectures.

