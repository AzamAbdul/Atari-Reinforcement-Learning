# Reinforcement Learning for Atari
Experiments with Reinforcement Learning to play Atari games. See the original [Google Deepmind Atari paper](https://arxiv.org/pdf/1312.5602). 

At the time of writing, the described experiments are limited to 'Pong' with further expansion for other games planned.  

## Credits

Fork of [Reinforcement Learning repo](https://github.com/BaptisteBell/ReinforcementLearning).

Authored by [@BaptisteBell](https://github.com/BaptisteBell) and [@bribridu59](https://github.com/bribridu59).

## Modifications & Improvements
Following list describes the diff between this fork and the original repo

**Training:**
- Double DQN (instead of target vs online network trick)
- Removed batch normalization
- Modified hyperparameters (γ=0.98, slower ε decay, 10k warmup)

**Performance:**
- PyTorch compilation (~10-20% speedup)
- Fused AdamW optimizer
- TF32 precision
- Gradient clipping

**Quality of life:**
- CSV exports for rewards and losses for each training run
- Commandline args for: profiling, and rendering of a provided model playing 'Pong'
  
## Results

**Rendering**

Green = [good_model.pth](https://github.com/AzamAbdul/Atari_RL_Playground/tree/master/models)

![alt text](https://raw.githubusercontent.com/AzamAbdul/Atari_RL_Playground/refs/heads/master/media/atari_pong_good_model.gif)

***Loss @ 500 episodes***
![alt text](https://raw.githubusercontent.com/AzamAbdul/Atari_RL_Playground/refs/heads/master/media/avg_loss_20251021.png)
***Reward @ 500 episodes***
![alt text](https://raw.githubusercontent.com/AzamAbdul/Atari_RL_Playground/refs/heads/master/media/total_rewards_2025-10-21.png)

## Future modifications
- Checkpointing
- Reward and Loss logging gated by cmdline flag
- Expanded profiling
- Experiments on more Atari games (beyond Pong)
- Network architecture refinements


## Usage

```bash
python pong.py              # Train
python pong.py --log-timing # Train with timing logs
```





