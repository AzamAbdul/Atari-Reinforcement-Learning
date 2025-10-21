# [WIP] RL for Atari
Experiments with Reinforcement Learning to play Atari games.
See the original [Google Deepmind Atari paper](https://arxiv.org/pdf/1312.5602). 
 
## Credits

Fork of the following [Reinforcement Learning repo](https://github.com/BaptisteBell/ReinforcementLearning). Authored by [@BaptisteBell](https://github.com/BaptisteBell) and [@bribridu59](https://github.com/bribridu59).

## Modifications and Improvements
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
- Optional execution time profiler (`--log-timing`)

  
## Results (so far)

**Pong Training Loss @ 500 episodes**
![alt text](https://i.imgur.com/WF3MBli.png)
**Pong Reward @ 500 episodes**
![alt text](https://i.imgur.com/NcaBhcF.png)

## TBD Feature Plan
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



