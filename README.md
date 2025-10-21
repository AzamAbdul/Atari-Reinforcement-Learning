# Pong DQN Solution

## Credits

Fork of the [original Pong DQN implementation](https://github.com/BaptisteBell/ReinforcementLearning). Authored by [@BaptisteBell](https://github.com/BaptisteBell) and [@bribridu59](https://github.com/bribridu59).

## Modifications and Improvements

**Algorithm:**
- Double DQN (instead of target vs online network trick)
- Removed batch normalization
- Modified hyperparameters (γ=0.98, slower ε decay, 10k warmup)

**Performance:**
- PyTorch compilation (~10-20% speedup)
- Fused AdamW optimizer
- TF32 precision
- Gradient clipping

**Logging:**
- CSV exports for rewards/losses
- Optional timing profiler (`--log-timing`)

## Usage

```bash
python pong.py              # Train
python pong.py --log-timing # Train with timing logs
```
