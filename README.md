# Reinforcement Learning for Atari
Experiments with Reinforcement Learning to play Atari games. See the original [Google Deepmind Atari paper](https://arxiv.org/pdf/1312.5602). 

This repo is a modified fork of [Reinforcement Learning repo](https://github.com/BaptisteBell/ReinforcementLearning) authored by [@BaptisteBell](https://github.com/BaptisteBell) and [@bribridu59](https://github.com/bribridu59).

## Modifications & Improvements
The following list describes the diff between this fork and the original repo

**Training:**
- Double DQN
- Removed batch normalization
- Modified hyperparameters

**Performance:**
- PyTorch compilation
- Fused AdamW optimizer
- TF32 precision
- Gradient clipping

**Quality of life:**
- CSV exports for rewards and losses during each training run
- Commandline args for profiling and rendering

## Usage

```bash
# Training
python pong.py
python pong.py --enable-perf-logs

# Render Demonstration of model gameplay
python pong.py --demo
python pong.py --demo -m ./models/good_model.pth
```

## Results

**Rendering**

Green = [good_model.pth](https://github.com/AzamAbdul/Atari_RL_Playground/tree/master/models)

![alt text](https://raw.githubusercontent.com/AzamAbdul/Atari_RL_Playground/refs/heads/master/media/atari_pong_good_model.gif)

***Loss @ 500 episodes (each plot is a unique training run)***
![alt text](https://raw.githubusercontent.com/AzamAbdul/Atari_RL_Playground/refs/heads/master/media/avg_loss_20251021.png)
***Reward @ 500 episodes (each plot is a unique training run)***
![alt text](https://raw.githubusercontent.com/AzamAbdul/Atari_RL_Playground/refs/heads/master/media/total_rewards_2025-10-21.png)










