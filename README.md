# DRL-Singledish-PPO: Deep Reinforcement Learning for Single-Dish Telescope Control 🔭

This repository contains the implementation of Proximal Policy Optimization (PPO) agents designed to control a simulated single-dish radio telescope environment built using Gymnasium (SingleDish-v4). The goal of the agent is to efficiently track celestial targets while navigating system constraints and avoiding physical obstacles, utilizing complex image and vector observations.

## 🌟 Overview

This project explores the application of Deep Reinforcement Learning (DRL) to the dynamic, multi-modal control problem of telescope pointing. The core components include:

SingleDish-v4 Environment: A custom Gymnasium environment providing image-based observations alongside low-dimensional vector data. The simulation includes realistic physics for antenna movement (motors, inertia, noise) and models for tracking targets (RA/Dec to Alt/Az conversion using astropy).

PPO Agents (CNN-LSTM & CNN-MLP): Implementations of the PPO algorithm optimized for continuous control. The policies incorporate dual convolutional encoders and, in one variant, an LSTM layer to process temporal information from the image sequence.

## ⚙️ Algorithms and Architecture

The agents are implemented in PyTorch, leveraging the cleanRL framework structure. Both models use a dual-stream CNN encoder architecture before feeding into the policy and value networks.

### Shared Model Architecture

The Agent class in both scripts utilizes separate convolutional networks for processing distinct parts of the image observation:

encoder: Processes a 3-channel input (likely wide-view, target map, etc.) and outputs a 256-dimensional feature vector.

bore_encoder: Processes a 1-channel input (likely the narrow, boresight view) and outputs a 256-dimensional feature vector.

Feature Fusion: The concatenated 512 features (256 + 256) are combined with 7 non-spatial features (e.g., episode progress, historical actions) before being passed to the main latent_feature MLP.

### PPO Variants

| File | Policy Core | Description |
|---|---|---|
| ppo_continuous_action.py | CNN/MLP | Standard PPO implementation. The features are processed by a sequence of ReLU-activated Linear layers (self.actor) to produce the action mean and log standard deviation. |
| ppo_lstm_continuous_action.py | CNN/LSTM | PPO variant incorporating a nn.LSTM(256, 256) layer. This allows the agent to utilize temporal context and motion history across timesteps. |
### Core Training Parameters

| Parameter | Default Value | Description |
|---|---|---|
| Environment ID | SingleDish-v4 | The custom Gymnasium environment name. |
| Total Timesteps | 5,000,000 | Total timesteps for the experiment.|
| Learning Rate | 5e-5 | Adam optimizer learning rate, with optional annealing. |
| Number of Envs | 8 | Number of parallel environment instances (gym.vector.AsyncVectorEnv). |
| Steps per Rollout | 2048 | Number of steps collected per environment per iteration. |
| Discount Factor ($\gamma$) | 0.99 | Future reward discount rate. | 

## 🌐 Environment Details (SingleDish-v4)

The environment logic is defined primarily in single_dish_env_image_obs.py.

- Observation Space (Image + Vector):

  - Image: Box(6, 64, 64, dtype=np.float32). The 6 channels encode necessary information about the sky, antenna, and target density.

  - Vector (7 features): Non-spatial state features accessible via canvas[0, 0:7, 0].

- Action Space (Continuous):

  - Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for normalized commands to the two motor axes.

  - Target Dynamics: Targets are celestial objects (Radec class) whose positions are calculated dynamically (RA/Dec to Alt/Az) using astropy based on a simulated ground station (EarthLocation).

  - Sun Tracking: The dedicated Sun class integrates accurate Sun positioning and implements a basic SQLite caching mechanism (db/sun.db) for efficient lookups, minimizing runtime astronomical computations.

## 🛠️ Installation and Setup

### Prerequisites

- Python 3.8+

- NVIDIA GPU (Recommended for speed via autocast and GradScaler in training scripts).

### Setup

Clone the repository:
```
git clone [https://github.com/bombonTH/DRL-Singledish-PPO.git](https://github.com/bombonTH/DRL-Singledish-PPO.git)
cd DRL-Singledish-PPO
```

Install dependencies:
```
# Install core scientific and RL libraries
pip install torch numpy gymnasium tyro astropy opencv-python

# You may also need additional packages depending on your environment, e.g., for W&B tracking or video:
# pip install wandb moviepy
```

Create Assets Directory:
The environment relies on pre-generated NumPy arrays (assets) for boundaries and reward weighting. You must create the asset directory and place the files (e.g., bound64px.npy, canvas.npy, weight_matrix_exp.npy, weight_matrix_lnr.npy) inside it.
```
mkdir -p single_dish_env_image_obs/canvas
### Place required .npy files here
```

Database Setup (Optional, for Sun Tracking):
If using the Sun class, ensure the db/sun.db SQLite database is accessible. You may need to create a dummy database file structure if running tests.

## 🚀 Usage

### Training the PPO Agent

Use the provided scripts with arguments managed by tyro.

1. Train the CNN-MLP PPO Agent:
```
python ppo_continuous_action.py \
    --exp_name "ppo_cnn_mlp" \
    --track \
    --total_timesteps 5000000 \
    --num_envs 8 \
    --seed 1
```

2. Train the CNN-LSTM PPO Agent:
```
python ppo_lstm_continuous_action.py \
    --exp_name "ppo_cnn_lstm" \
    --track \
    --total_timesteps 5000000 \
    --num_envs 8 \
    --seed 1
```

### Resuming Training

To resume training from saved checkpoints (assuming weights are available in a run folder), use the --resume flag:

```
python ppo_lstm_continuous_action.py --resume
```

## 📄 Citation

If you use this code or the environment design in your research, please consider citing the corresponding paper:
```
@article{puangragsa2025DRLinTCS,
  title = {Implementation of Deep Reinforcement Learning for Radio Telescope Control and Scheduling},
  author = {Sarut Puangragsa, Tanawit Sahavisit, Popphon Laon, and Pattarapong Phasukkit},
  journal = {Galaxies, Submitted for Review (202)},
  year = {2025},
  publisher = {KMITL / MDPI}
}
```
