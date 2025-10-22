import gymnasium
import numpy as np
from agent import DeepQLearningAgent
from model_utils import *
from training_logger import TrainingLogger
import warnings
import time
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
    parser.add_argument('--log-timing', action='store_true', default=False,
                        help='Enable detailed timing logging (default: False)')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Run in demo mode (play only, no training)')
    parser.add_argument('-m', '--model', type=str, default='./models/model.pth',
                        help='Path to model file (default: model.pth)')
    return parser.parse_args()

def train(hyper_params: HyperParams, log_timing=False):
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

    # Initialize training logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    logger = TrainingLogger(timestamp, log_timing)

    for epoch in range(hyper_params.epochs):
        state, _ = env.reset()
        stacked_frames = [preprocess(state)] * 4
        state, stacked_frames = stack_frames(stacked_frames, state, is_new_episode=True)
        done = False
        total_reward = 0
        epoch_losses = []


        while not done:
            if log_timing:
                step_start = time.time()
                action_start = time.time()

            action = agent.choose_action(state)

            if log_timing:
                action_time = (time.time() - action_start) * 1000
                env_start = time.time()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if log_timing:
                env_time = (time.time() - env_start) * 1000
                preprocess_start = time.time()

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, is_new_episode=False)

            if log_timing:
                preprocess_time = (time.time() - preprocess_start) * 1000
                memory_start = time.time()

            agent.add_memory(state, action, reward, next_state, done)

            if log_timing:
                memory_time = (time.time() - memory_start) * 1000
                learn_start = time.time()

            learned = 0
            if len(agent.memory) > hyper_params.min_replay_size and agent.timestep % hyper_params.learning_freq == 0:
                loss = agent.learn(hyper_params.batch_size)
                learned = 1

                # Collect loss data for epoch averaging
                if loss is not None:
                    loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
                    epoch_losses.append(loss_value)

                if agent.timestep % 1000 == 0:
                    print(f"loss: {loss}")

            if log_timing:
                learn_time = (time.time() - learn_start) * 1000

            state = next_state
            total_reward += reward
            agent.timestep += 1

            if log_timing:
                total_step_time = (time.time() - step_start) * 1000
                logger.add_timing_data([agent.timestep, epoch + 1, action_time, env_time,
                                       preprocess_time, memory_time, learn_time, total_step_time, learned])

                if agent.timestep % 10000 == 0:
                    logger.flush_timing_buffer()

            if agent.timestep % hyper_params.target_nn_update_freq == 0:
                agent.update_target_network()
                print(f"Timestep {agent.timestep}, Epoch {epoch + 1}, Epsilon: {agent.epsilon:.3f}")

        rewards_per_epoch.append(total_reward)

        # Write loss data for this epoch
        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.log_loss(epoch + 1, avg_loss)

        # Write reward data for this epoch
        logger.log_reward(epoch + 1, total_reward)

        print(f"Epoch {epoch + 1}/{hyper_params.epochs} completed - Total Reward: {total_reward}")

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
        train(log_timing=args.log_timing, hyper_params=pong_hyper_params)
