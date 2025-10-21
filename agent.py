import random
from model import *
import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DeepQLearningAgent:
    def __init__(self,
                 learning_rate: float,
                 gamma: float,
                 n_actions: int,
                 input_dim: t.Tuple[int] = (4, 84, 84),
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.999996):

        self.lr = learning_rate
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.input_dim = input_dim
        self.memory = deque(maxlen=500000)
        self.timestep = 0
        self.epsilon_start = epsilon

        self.model, self.optimizer, self.loss_fn = build_dqn(input_dim, n_actions, lr=learning_rate)

        self.target_net, _, _ = build_dqn(input_dim, n_actions, lr=learning_rate)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        # Enable TensorFloat-32 for better performance on modern GPUs
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            print("TensorFloat-32 enabled for faster matrix operations")

        # Compile models for 10-20% speedup (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model)
            self.target_net = torch.compile(self.target_net)
            print("Models compiled successfully for optimized inference")
        except Exception as e:
            print(f"Model compilation not available: {e}")
            print("Using standard models (consider upgrading to PyTorch 2.0+)")

    def update_target_network(self):
        # Handle compiled models properly
        if hasattr(self.model, '_orig_mod') and hasattr(self.target_net, '_orig_mod'):
            # Both models are compiled
            self.target_net._orig_mod.load_state_dict(self.model._orig_mod.state_dict())
        elif hasattr(self.model, '_orig_mod'):
            # Only main model is compiled
            self.target_net.load_state_dict(self.model._orig_mod.state_dict())
        elif hasattr(self.target_net, '_orig_mod'):
            # Only target model is compiled
            self.target_net._orig_mod.load_state_dict(self.model.state_dict())
        else:
            # Neither model is compiled
            self.target_net.load_state_dict(self.model.state_dict())

    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=self.model.device)
            state = state.unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item() 

    def learn(self, batch):
        if len(self.memory) < batch:
            return
        

        states, actions, rewards, next_states, dones = self.sample_memory(batch)

        # Direct device allocation to reduce tensor copy overhead
        device = self.model.device
        states = torch.tensor(states, dtype=torch.float, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=device)
        dones = torch.tensor(dones, dtype=torch.float, device=device)


        actions = actions.unsqueeze(1)
        q_values = self.model(states).gather(1, actions).squeeze(-1)


        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)              # select (online)
            next_q_tgt   = self.target_net(next_states).gather(1, next_actions).squeeze(1)  # evaluate (target)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_tgt



        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Return loss without .item() to avoid GPU→CPU sync
        # Only convert to CPU when actually needed for logging
        return loss.detach()
