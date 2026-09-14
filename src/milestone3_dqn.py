"""Milestone 3: a small custom DQN for the adaptive pruning environment.

Per this project's own environment notes ("a small custom DQN loop is fine
and easier to explain line by line"), this is a from-scratch DQN, not a
library -- also keeps it directly inspectable for the XAI layer later.

MONITORING IS BUILT IN FROM THE START, not added after observing a problem
(same pattern as the rest of this project): train_dqn tracks, per episode,
whether that episode's trajectory reached a "deep" pruning region (default
>40% removed), what epsilon was at the time, and whether the agent chose
STOP before the removal budget was exhausted -- plus periodic snapshots of
what fraction of the replay buffer's stored transitions come from deep
trajectories. This lets a later question like "why does the trained policy
always max out the budget?" be answered by looking at logged data (was the
region ever explored? was it explored before epsilon got too greedy? does
the Q-network's own STOP-vs-continue estimate at states it trained on
actually favor stopping?) instead of guessing.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.milestone3 import PruningEnv, STOP_ACTION


class QNetwork(nn.Module):
    """Small MLP: state (STATE_SIZE-dim) -> one Q-value per action."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity uniform-random experience replay."""

    def __init__(self, capacity: int = 5000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def fraction_deep(self, threshold: float = 0.4) -> float:
        """Fraction of buffered transitions whose STATE (before the action
        that produced this transition) already had cumulative_removed_fraction
        > threshold. state[0] is cumulative_removed_fraction by construction
        (see PruningEnv._compute_state). Monitoring hook, checked throughout
        training rather than assumed adequate.
        """
        if not self.buffer:
            return 0.0
        deep_count = sum(1 for t in self.buffer if t.state[0] > threshold)
        return deep_count / len(self.buffer)


def select_action(q_network: QNetwork, state: np.ndarray, epsilon: float, action_size: int, action_mask: Optional[np.ndarray] = None) -> int:
    """Epsilon-greedy: random VALID action with probability epsilon, else argmax Q over valid actions.

    action_mask (if given) excludes actions for already-exhausted buckets --
    without it, both random exploration and the greedy policy can pick a
    dead "prune bucket X" action that removes nothing, wasting a step (or,
    for the greedy policy, getting stuck repeating it until max_steps).
    """
    valid_actions = np.arange(action_size) if action_mask is None else np.flatnonzero(action_mask)
    if random.random() < epsilon:
        return int(random.choice(valid_actions))
    with torch.no_grad():
        q_values = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    if action_mask is not None:
        q_values = q_values.masked_fill(torch.from_numpy(~action_mask), float("-inf"))
    return int(q_values.argmax().item())


def train_step(q_network: QNetwork, target_network: QNetwork, optimizer: torch.optim.Optimizer, batch: List[Transition], gamma: float) -> float:
    """One TD-learning gradient step on a sampled minibatch.

    Standard DQN target: r + gamma * max_a' Q_target(s', a') for non-terminal
    transitions, just r for terminal ones. The TARGET network (a periodically
    synced, lagged copy of q_network) computes the bootstrapped next-state
    value -- using the same live network for both the prediction and its own
    target is a well-known source of training instability, since the target
    would shift under the network's feet every single update.
    """
    states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
    next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
    dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32)

    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_dqn(
    env: PruningEnv,
    n_episodes: int = 300,
    gamma: float = 0.95,
    lr: float = 1e-3,
    batch_size: int = 32,
    buffer_capacity: int = 5000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 200,
    target_sync_every: int = 10,
    min_buffer_before_training: int = 200,
    deep_threshold: float = 0.4,
    low_epsilon_threshold: float = 0.2,
    seed: int = 0,
) -> Tuple[QNetwork, Dict]:
    """Train a small DQN on env. Returns (trained Q-network, training log).

    The log dict is deliberately rich (not just loss/return) so exploration
    adequacy can be diagnosed AFTER training without rerunning anything --
    see the module docstring.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    # A training step samples batch_size transitions; starting training before
    # the buffer holds at least that many would make random.sample fail.
    min_buffer_before_training = max(min_buffer_before_training, batch_size)

    q_network = QNetwork(env.state_size, env.action_size)
    target_network = QNetwork(env.state_size, env.action_size)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    log: Dict[str, list] = {
        "episode_epsilon": [],
        "episode_return": [],
        "episode_discounted_return": [],
        "episode_final_removed": [],
        "episode_reached_deep": [],
        "episode_stopped_early": [],
        "episode_first_deep_step": [],
        "buffer_deep_fraction_checkpoints": [],
        "loss": [],
    }
    first_deep_episode: Optional[int] = None

    for episode in range(n_episodes):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / epsilon_decay_episodes)
        state = env.reset()
        episode_return = 0.0
        # Supervisor-flagged fix: episode_return above is a plain, UNDISCOUNTED
        # sum -- it weighs a reward received on step 1 the same as one
        # received on step 15, which is not how this agent's own objective
        # (the gamma-discounted DQN Bellman target, see train_step above)
        # values them. Diagnosing whether STOP-ing early helped or hurt an
        # episode must use the SAME discounting the agent is actually
        # optimizing for, or the diagnosis is answering a different question
        # than "did the agent's actual objective improve." episode_rewards
        # collects the raw per-step sequence so the proper discounted return
        # (sum of gamma**i * r_i) can be computed once the episode ends.
        episode_rewards: List[float] = []
        reached_deep = False
        first_deep_step: Optional[int] = None
        stopped_early = False
        info = {"cumulative_removed_fraction": 0.0}

        done = False
        step = 0
        while not done:
            action = select_action(q_network, state, epsilon, env.action_size, action_mask=env.action_mask())
            next_state, reward, done, info = env.step(action)
            buffer.push(Transition(state, action, reward, next_state, done))
            episode_return += reward
            episode_rewards.append(reward)

            if action == STOP_ACTION and info["cumulative_removed_fraction"] < env.max_removal_fraction:
                stopped_early = True
            if info["cumulative_removed_fraction"] > deep_threshold and not reached_deep:
                reached_deep = True
                first_deep_step = step

            state = next_state
            step += 1

            if len(buffer) >= min_buffer_before_training:
                batch = buffer.sample(batch_size)
                loss = train_step(q_network, target_network, optimizer, batch, gamma)
                log["loss"].append(loss)

        if reached_deep and first_deep_episode is None:
            first_deep_episode = episode
        if episode % target_sync_every == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Complete discounted return G_0 = sum_i gamma**i * r_i over the WHOLE
        # episode (computed once the episode is over, so every term's full
        # discount weight is known) -- this is what "diagnose STOP behaviour
        # using complete discounted returns" means: compare THIS number
        # between stopped-early and ran-to-budget episodes, not the plain sum.
        discounted_return = sum((gamma ** i) * r for i, r in enumerate(episode_rewards))

        log["episode_epsilon"].append(epsilon)
        log["episode_return"].append(episode_return)
        log["episode_discounted_return"].append(discounted_return)
        log["episode_final_removed"].append(info["cumulative_removed_fraction"])
        log["episode_reached_deep"].append(reached_deep)
        log["episode_stopped_early"].append(stopped_early)
        log["episode_first_deep_step"].append(first_deep_step)

        if episode % 20 == 0 or episode == n_episodes - 1:
            log["buffer_deep_fraction_checkpoints"].append((episode, buffer.fraction_deep(deep_threshold)))

    log["first_deep_episode"] = first_deep_episode
    log["low_epsilon_episode"] = next((i for i, e in enumerate(log["episode_epsilon"]) if e <= low_epsilon_threshold), None)
    log["final_buffer_deep_fraction"] = buffer.fraction_deep(deep_threshold)
    log["replay_buffer"] = buffer  # kept for post-training Q-value inspection

    return q_network, log


def run_greedy_episode(env: PruningEnv, q_network: QNetwork) -> Dict:
    """Roll out ONE episode under the trained policy, fully greedy (epsilon=0).

    Deterministic given a fixed env/graph/model, so a single rollout is
    sufficient -- there is no stochasticity left once epsilon=0.
    """
    state = env.reset()
    trajectory = []
    done = False
    while not done:
        action = select_action(q_network, state, epsilon=0.0, action_size=env.action_size, action_mask=env.action_mask())
        next_state, reward, done, info = env.step(action)
        trajectory.append(
            {
                "action": action,
                "reward": reward,
                "cumulative_removed_fraction": info["cumulative_removed_fraction"],
                "containment_ratio": env._current_containment_ratio,
                "utility_ratio": env._current_utility_ratio,
            }
        )
        state = next_state

    final = trajectory[-1]
    return {
        "trajectory": trajectory,
        "stopped_early": final["action"] == STOP_ACTION and final["cumulative_removed_fraction"] < env.max_removal_fraction,
        "final_removed": final["cumulative_removed_fraction"],
    }


def inspect_deep_state_q_values(q_network: QNetwork, buffer: ReplayBuffer, deep_threshold: float = 0.4, n_samples: int = 5, seed: int = 0) -> List[Dict]:
    """For actual replay-buffer states that were already deep (>threshold
    removed) when observed DURING TRAINING, report the trained Q-network's
    Q-value for STOP vs. the best non-STOP action.

    This is the direct check for "exploration failure vs. genuine learned
    preference": if the network was trained on this region (these states are
    proof it was) and its own Q-values still clearly favor continuing over
    STOP, that's a real learned preference, not a lack of data. If the
    network was never trained on any deep states at all (empty result here),
    that IS an exploration gap.
    """
    rng = random.Random(seed)
    deep_transitions = [t for t in buffer.buffer if t.state[0] > deep_threshold]
    if not deep_transitions:
        return []
    sample = rng.sample(deep_transitions, min(n_samples, len(deep_transitions)))

    results = []
    with torch.no_grad():
        for t in sample:
            q_values = q_network(torch.tensor(t.state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            results.append(
                {
                    "cumulative_removed_fraction": float(t.state[0]),
                    "q_values": q_values.tolist(),
                    "q_stop": q_values[STOP_ACTION].item(),
                    "q_best_non_stop": q_values[:STOP_ACTION].max().item(),
                    "action_taken_during_collection": t.action,
                    "reward_observed": t.reward,
                }
            )
    return results
