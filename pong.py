import typing as t
import gymnasium
import numpy as np
from agent import DeepQLearningAgent
import random
from collections import deque
from model_utils import *
import ale_py
import warnings
import time
import csv
import os
from datetime import datetime
import argparse
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")

env = gymnasium.make("ALE/Pong-v5")

gamma = 0.98
batch_size = 64  # Increased for better GPU utilization
n_actions = env.action_space.n
epochs = 500
learning_freq = 1  # Learn every N steps instead of every step
min_replay_size = 10000  # Wait for sufficient experience before learning

agent = DeepQLearningAgent(learning_rate=0.00005, gamma=gamma, n_actions=n_actions)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Pong DQN')
    parser.add_argument('--log-timing', action='store_true', default=False,
                        help='Enable detailed timing logging (default: False)')
    return parser.parse_args()

def train_pong(log_timing=False):
    print("Starting training...")
    agent.model.train()
    rewards_per_epoch = []
    agent.timestep = 0

    # Pre-allocate tensors to reduce allocation overhead
    device = agent.model.device
    print(f"Training on device: {device}")

    # Generate timestamp suffix for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Initialize timing logging if enabled
    if log_timing:
        csv_filename = f'training_timing_{timestamp}.csv'
        csv_headers = ['step', 'epoch', 'action_selection_ms', 'env_step_ms', 'preprocessing_ms',
                       'memory_add_ms', 'learning_ms', 'total_step_ms', 'learned']

        # Create CSV file with headers
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)

        # Buffer for timing data (write every 10000 steps)
        timing_buffer = []
        print(f"Timing data will be logged to {csv_filename} every 10000 steps")
    else:
        print("Timing logging disabled - use --log-timing to enable")

    # Initialize loss logging
    loss_filename = f'training_loss_{timestamp}.csv'
    loss_headers = ['epoch', 'avg_loss']

    # Create loss CSV file with headers
    with open(loss_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(loss_headers)

    print(f"Loss data will be logged to {loss_filename} after each epoch")

    # Initialize reward logging
    reward_filename = f'training_rewards_{timestamp}.csv'
    reward_headers = ['epoch', 'total_reward']

    # Create reward CSV file with headers
    with open(reward_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(reward_headers)

    print(f"Reward data will be logged to {reward_filename} after each epoch")

    for epoch in range(epochs):
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
            if len(agent.memory) > min_replay_size and agent.timestep % learning_freq == 0:
                loss = agent.learn(batch_size)
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
                timing_buffer.append([agent.timestep, epoch + 1, action_time, env_time,
                                    preprocess_time, memory_time, learn_time, total_step_time, learned])

                if agent.timestep % 10000 == 0:
                    with open(csv_filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(timing_buffer)
                    print(f"Wrote {len(timing_buffer)} timing records to {csv_filename}")
                    timing_buffer.clear()

            if agent.timestep % 10000 == 0:
                agent.update_target_network()
                print(f"Timestep {agent.timestep}, Epoch {epoch + 1}, Epsilon: {agent.epsilon:.3f}")

        rewards_per_epoch.append(total_reward)

        # Write loss data for this epoch
        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            with open(loss_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1, avg_loss])

        # Write reward data for this epoch
        with open(reward_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, total_reward])

        print(f"Epoch {epoch + 1}/{epochs} completed - Total Reward: {total_reward}")
        if (epoch + 1) % 10 == 0:
            if log_timing:
                print(f"Timing data logged to {csv_filename} - {agent.timestep} total steps recorded")

    print("Training completed.")
    save_model(agent.model)
    print("Model saved.")

def play_pong():
    print("Loading model for evaluation...")
    model = load_model()
    model.eval()
    total_rewards = []

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

# Run training and evaluation
if __name__ == "__main__":
    args = parse_args()
    train_pong(log_timing=args.log_timing)
    play_pong()
    env.close()
