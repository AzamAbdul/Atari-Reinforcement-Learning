import gymnasium
import numpy as np
from agent import DeepQLearningAgent
from model_utils import *
from training_logger import TrainingLogger
from step_timer import StepTimer
import warnings
from datetime import datetime
import argparse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")

class HyperParams(BaseModel):
    gamma: float = Field(0.98, ge=0.9, le=1.0)
    batch_size: int = Field(64, ge=1)
    epochs: int = Field(500, ge=1)
    learning_freq: int = Field(1, ge=1)
    min_replay_size: int = Field(10_000, ge=0)
    learning_rate: float = Field(0.000_05, le=1)
    target_nn_update_freq: int = Field(10_000, ge=1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Pong DQN')
    parser.add_argument('--enable-perf-logs', action='store_true', default=False,
                        help='Enable detailed performance logging (default: False)')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Run in demo mode (play only, no training)')
    parser.add_argument('-m', '--model', type=str, default='./models/model.pth',
                        help='Path to model file (default: model.pth)')
    return parser.parse_args()

def train(hyper_params: HyperParams, enable_perf_logs=False):
    env = gymnasium.make("ALE/Pong-v5")
    n_actions = env.action_space.n
    agent = DeepQLearningAgent(learning_rate=hyper_params.learning_rate, gamma=hyper_params.gamma, n_actions=n_actions)

    print("Starting training...")
    agent.model.train()
    rewards_per_epoch = []
    agent.timestep = 0

    # Pre-allocate tensors to reduce allocation overhead
    device = agent.model.device
    print(f"Training on device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    logger = TrainingLogger(timestamp, enable_perf_logs)
    timer = StepTimer(enabled=enable_perf_logs)

    for epoch in range(hyper_params.epochs):
        state, _ = env.reset()
        stacked_frames = [preprocess(state)] * 4
        state, stacked_frames = stack_frames(stacked_frames, state, is_new_episode=True)
        done = False
        total_reward = 0
        epoch_losses = []


        while not done:
            timer.start('total_step')

            with timer.time('action_selection'):
                action = agent.choose_action(state)

            with timer.time('env_step'):
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            with timer.time('preprocessing'):
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, is_new_episode=False)

            with timer.time('memory_add'):
                agent.add_memory(state, action, reward, next_state, done)

            with timer.time('online_net_training'):
                memory_reached_threshold = len(agent.memory) > hyper_params.min_replay_size
                is_learning_step = agent.timestep % hyper_params.learning_freq == 0 
                if memory_reached_threshold and is_learning_step:
                    loss = agent.learn(hyper_params.batch_size)
                    epoch_losses.append(float(loss.item()))

            with timer.time('target_net_training'):
                target_network_stale = agent.timestep % hyper_params.target_nn_update_freq == 0
                if target_network_stale:
                    agent.update_target_network()
                    print(f"Timestep {agent.timestep}, Epoch {epoch}, Target Network updated")

            state = next_state
            total_reward += reward

            timer.stop('total_step')

            logger.add_timing_data([
                agent.timestep,
                epoch,
                timer.get('action_selection'),
                timer.get('env_step'),
                timer.get('preprocessing'),
                timer.get('memory_add'),
                timer.get('online_net_training'),
                timer.get('target_net_training'),
                timer.get('total_step')
              ])

            if agent.timestep % 10000 == 0:
                 logger.flush_timing_buffer()
            
            agent.timestep += 1

        rewards_per_epoch.append(total_reward)

        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.log_loss(epoch, avg_loss)

        logger.log_reward(epoch, total_reward)
        print(f"Epoch {epoch}/{hyper_params.epochs - 1} completed - Total Reward: {total_reward}")

    print("Training completed.")
    save_model(agent.model)
    print("Model saved.")
    env.close()

def play_pong(model_path='model.pth'):
    print("Loading model for evaluation...")
    model = load_model(model_path)
    model.eval()
    total_rewards = []

    env = gymnasium.make("ALE/Pong-v5", render_mode="human")

    for episode in range(10):
        state, _ = env.reset()
        stacked_frames = [preprocess(state)] * 4
        state, stacked_frames = stack_frames(stacked_frames, state, is_new_episode=True)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(model.device)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, is_new_episode=False)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print(f"Average Reward over 10 Episodes: {np.mean(total_rewards)}")
    env.close()

if __name__ == "__main__":
    args = parse_args()

    if args.demo:
        print("Running demo mode")
        play_pong(model_path=args.model)
    else:
        print("Running training pipeline")
        pong_hyper_params = HyperParams()
        train(enable_perf_logs=args.enable_perf_logs, hyper_params=pong_hyper_params)
