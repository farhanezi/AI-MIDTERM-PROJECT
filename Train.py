import os
import random
import time
import sys
from collections import deque, namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rocket_env import SimpleRocketEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

cfg = {
    "seed": 42,
    "env_steps_per_variant": 100_000,
    "max_episode_steps": 800,
    "buffer_size": 300_000,
    "batch_size": 256,
    "gamma": 0.995,
    "lr": 5e-4,
    "target_update_freq": 1000,
    "train_freq": 1,
    "learning_starts": 10000,
    "eval_every": 25_000,
    "eval_episodes": 20,
    "epsilon_start": 1.0,
    "epsilon_final": 0.01,
    "epsilon_decay_steps": 600_000,
    "save_dir": "./models",
    "variants": ["DQN", "DoubleDQN", "DuelingDQN"],
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_frames": 400_000,
    "gradient_clip": 10.0,
    "polyak_tau": 0.001,
    "reward_scale": 0.01,
    "use_soft_update": True,
    "huber_delta": 1.0,
}

os.makedirs(cfg["save_dir"], exist_ok=True)
torch.backends.cudnn.benchmark = True

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError("Buffer tidak memiliki cukup sampel.")
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.size = 0

    def push(self, *args):
        transition = Transition(*args)
        if self.size < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            raise ValueError("Empty buffer")
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = Transition(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, pr in zip(indices, priorities):
            self.priorities[idx] = pr

    def __len__(self):
        return self.size

class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, dueling=False):
        super().__init__()
        self.dueling = dueling
        
        self.torso = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        if not dueling:
            self.head = nn.Linear(256, n_actions)
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        x = self.torso(x)
        if not self.dueling:
            return self.head(x)
        else:
            v = self.value_stream(x)
            a = self.advantage_stream(x)
            return v + (a - a.mean(dim=1, keepdim=True))


class DQNAgent:
    def __init__(self, obs_dim, n_actions, variant="DQN", cfg=cfg):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.variant = variant
        self.cfg = cfg

        dueling = (variant == "DuelingDQN" or variant == "PER")
        
        self.online_net = QNetwork(obs_dim, n_actions, dueling=dueling).to(DEVICE)
        self.target_net = QNetwork(obs_dim, n_actions, dueling=dueling).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = optim.AdamW(
            self.online_net.parameters(), 
            lr=cfg["lr"],
            eps=1e-8,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=cfg["env_steps_per_variant"] // cfg["train_freq"],
            eta_min=1e-5
        )

        if variant == "PER":
            self.buffer = PrioritizedReplayBuffer(cfg["buffer_size"], alpha=cfg["per_alpha"])
        else:
            self.buffer = ReplayBuffer(cfg["buffer_size"])

        self.total_steps = 0
        self.learning_steps = 0
        
        self.q_values_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        self.successful_landings = 0
        self.perfect_landings = 0

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = self.online_net(state_t)
            self.q_values_history.append(q.max().item())
            action = int(q.argmax(1).item())
        return action

    def push_transition(self, state, action, reward, next_state, done):
        reward = reward * self.cfg["reward_scale"]
        self.buffer.push(state, action, reward, next_state, done)

    def soft_update(self, tau):
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def compute_td_loss(self, batch_size, beta=0.4):
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            batch, indices, weights = self.buffer.sample(batch_size, beta=beta)
            weights = torch.FloatTensor(weights).to(DEVICE)
        else:
            batch = self.buffer.sample(batch_size)
            indices = None
            weights = torch.ones(batch_size, device=DEVICE)

        states = torch.FloatTensor(np.array(batch.state)).to(DEVICE)
        actions = torch.LongTensor(np.array(batch.action)).to(DEVICE).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(batch.reward)).to(DEVICE)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(DEVICE)
        dones = torch.FloatTensor(np.array(batch.done).astype(np.float32)).to(DEVICE)

        q_values = self.online_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.variant == "DQN":
                next_q_target = self.target_net(next_states).max(1)[0]
            else:
                next_q_online = self.online_net(next_states)
                next_actions = next_q_online.argmax(1, keepdim=True)
                next_q_target = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            expected_q = rewards + (1.0 - dones) * self.cfg["gamma"] * next_q_target

        td_errors = (q_values - expected_q).detach()
        
        element_wise_loss = F.smooth_l1_loss(
            q_values, expected_q, 
            reduction='none', 
            beta=self.cfg["huber_delta"]
        )
        
        loss = (weights * element_wise_loss).mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0, indices, np.zeros(batch_size)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), 
            self.cfg["gradient_clip"]
        )
        
        self.optimizer.step()
        self.scheduler.step()
        self.learning_steps += 1
        
        if self.cfg.get("use_soft_update", False):
            self.soft_update(self.cfg["polyak_tau"])
        
        self.loss_history.append(loss.item())

        return loss.item(), indices, np.abs(td_errors.cpu().numpy()) + 1e-6

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

def linear_epsilon(step, cfg):
    eps_start = cfg["epsilon_start"]
    eps_final = cfg["epsilon_final"]
    eps_decay = cfg["epsilon_decay_steps"]
    if step >= eps_decay:
        return eps_final
    else:
        return eps_final + (eps_start - eps_final) * (1 - step/eps_decay)


def evaluate_agent(agent, n_episodes=15):
    env = SimpleRocketEnv(render_mode=None)
    rewards = []
    successes = 0
    perfect_landings = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0.0
        while True:
            action = agent.act(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated:
                if reward > 200:
                    successes += 1
                if reward > 600:
                    perfect_landings += 1
                break
            if truncated:
                break
        rewards.append(total)
    
    env.close()
    success_rate = successes / n_episodes
    perfect_rate = perfect_landings / n_episodes
    return float(np.mean(rewards)), float(np.std(rewards)), success_rate, perfect_rate

def train_variant(variant, cfg):
    print(f"\n{'='*70}")
    print(f"Training variant: {variant}")
    print(f"{'='*70}")
    
    env = SimpleRocketEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, variant=variant, cfg=cfg)

    episode_rewards = []
    episode_lengths = []
    successful_episodes = deque(maxlen=100)
    mean_rewards_eval = []
    steps = 0
    episode = 0
    state, _ = env.reset()
    ep_reward = 0.0
    ep_length = 0
    start_time = time.time()

    per_beta = cfg["per_beta_start"]
    beta_slope = (1.0 - cfg["per_beta_start"]) / max(1, cfg["per_beta_frames"])

    train_curve = deque(maxlen=5000)
    loss_history = deque(maxlen=1000)
    success_rate_history = []
    
    running_reward = None
    alpha_reward = 0.02

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Starting training for {cfg['env_steps_per_variant']:,} steps...")

    while steps < cfg["env_steps_per_variant"]:
        eps = linear_epsilon(steps, cfg)
        action = agent.act(state, epsilon=eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done_flag = bool(terminated or truncated)
        agent.push_transition(state, action, reward, next_state, done_flag)

        state = next_state
        ep_reward += reward
        ep_length += 1
        steps += 1
        agent.total_steps += 1

        if done_flag:
            episode += 1
            
            if running_reward is None:
                running_reward = ep_reward
            else:
                running_reward = alpha_reward * ep_reward + (1 - alpha_reward) * running_reward
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            train_curve.append(running_reward)
            
            if ep_reward > 200:
                agent.successful_landings += 1
                successful_episodes.append(1)
                if ep_reward > 600:
                    agent.perfect_landings += 1
            else:
                successful_episodes.append(0)
            
            state, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0

        if (len(agent.buffer) >= cfg["learning_starts"]) and (steps % cfg["train_freq"] == 0):
            if variant == "PER":
                per_beta = min(1.0, per_beta + beta_slope)
                beta = per_beta
            else:
                beta = 0.4

            loss, indices, td_errors = agent.compute_td_loss(cfg["batch_size"], beta=beta)
            
            if loss > 0 and not np.isnan(loss) and not np.isinf(loss):
                loss_history.append(loss)
            
            if variant == "PER" and indices is not None:
                agent.buffer.update_priorities(indices, td_errors.tolist())

            if not cfg.get("use_soft_update", False):
                if agent.learning_steps % cfg["target_update_freq"] == 0:
                    agent.update_target()

        if steps % cfg["eval_every"] == 0 and steps > 0:
            mean_eval, std_eval, success_rate, perfect_rate = evaluate_agent(agent, cfg["eval_episodes"])
            mean_rewards_eval.append((steps, mean_eval, std_eval))
            success_rate_history.append((steps, success_rate, perfect_rate))
            
            recent_success = np.mean(successful_episodes) if len(successful_episodes) > 0 else 0
            
            avg_loss = np.mean(loss_history) if loss_history else 0
            avg_q = np.mean(agent.q_values_history) if agent.q_values_history else 0
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            elapsed = time.time() - start_time
            steps_per_sec = steps / elapsed
            
            print(f"[{variant}] Step {steps:>7,}/{cfg['env_steps_per_variant']:,} | "
                      f"Eval: {mean_eval:>7.1f}±{std_eval:>5.1f} | "
                      f"Suc: {success_rate:>5.1%} | Perf: {perfect_rate:>5.1%} | "
                      f"Rec: {recent_success:>5.1%} | "
                      f"Loss: {avg_loss:>6.3f} | "
                      f"Q: {avg_q:>6.1f} | "
                      f"RunRew: {running_reward:>6.1f} | "
                      f"LR: {current_lr:.2e} | "
                      f"SPS: {steps_per_sec:.0f}")
    
    env.close()
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Finished {variant}:")
    print(f"  Total steps: {steps:,}")
    print(f"  Total episodes: {episode:,}")
    print(f"  Training time: {total_time:.1f}s ({steps/total_time:.0f} steps/s)")
    print(f"  Successful landings: {agent.successful_landings:,}")
    print(f"  Perfect landings: {agent.perfect_landings:,}")
    print(f"{'='*70}")
    
    mean_eval, std_eval, success_rate, perfect_rate = evaluate_agent(agent, cfg["eval_episodes"] * 2)
    print(f"Final evaluation ({cfg['eval_episodes'] * 2} episodes):")
    print(f"  Mean reward: {mean_eval:.2f} ± {std_eval:.2f}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Perfect landing rate: {perfect_rate:.1%}")

    os.makedirs(cfg["save_dir"], exist_ok=True)
    model_path = os.path.join(cfg["save_dir"], f"{variant}_model.pt")
    
    try:
        torch.save({
            "online_state": agent.online_net.state_dict(),
            "target_state": agent.target_net.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "cfg": cfg,
            "steps": steps,
            "episodes": episode,
            "success_rate": success_rate,
            "perfect_rate": perfect_rate,
            "mean_reward": mean_eval
        }, model_path)
        print(f"✓ Successfully saved {variant} model to {model_path}")
    except Exception as e:
        print(f"✗ Error saving {variant} model: {e}")

    return list(train_curve), mean_eval, std_eval, list(loss_history), success_rate_history

def main():
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])

    print("\n" + "="*70)
    print("ROCKET LANDING - OPTIMIZED DQN TRAINING 🚀")
    print("="*70)
    print(f"Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Save directory: {os.path.abspath(cfg['save_dir'])}")
    print(f"  Total steps per variant: {cfg['env_steps_per_variant']:,}")
    print(f"  Variants: {', '.join(cfg['variants'])}")
    print(f"  Learning rate: {cfg['lr']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Buffer size: {cfg['buffer_size']:,}")
    print(f"  Soft updates: {cfg['use_soft_update']}")
    print(f"  Reward scale: {cfg['reward_scale']}")
    print("="*70 + "\n")

    all_curves = {}
    eval_results = {}
    loss_histories = {}
    success_histories = {}

    for variant in cfg["variants"]:
        cfg_variant = dict(cfg)
        curve, mean_eval, std_eval, losses, success_hist = train_variant(variant, cfg_variant)
        all_curves[variant] = curve
        eval_results[variant] = (mean_eval, std_eval)
        loss_histories[variant] = losses
        success_histories[variant] = success_hist

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    for variant, curve in all_curves.items():
        if len(curve) > 0:
            x = np.arange(len(curve))
            ax1.plot(x, curve, label=variant, linewidth=2, alpha=0.8)
    ax1.set_title("Training Rewards (Running Average)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=200, color='g', linestyle='--', alpha=0.5, label='Success')
    ax1.axhline(y=600, color='b', linestyle='--', alpha=0.5, label='Perfect')
    
    ax2 = axes[0, 1]
    for variant, losses in loss_histories.items():
        if len(losses) > 100:
            window = 100
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            x = np.arange(len(smoothed))
            ax2.plot(x, smoothed, label=variant, linewidth=2, alpha=0.8)
    ax2.set_title("Training Loss (100-step MA)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    for variant, success_hist in success_histories.items():
        if success_hist:
            steps, success_rates, perfect_rates = zip(*success_hist)
            ax3.plot(steps, [r * 100 for r in success_rates], 
                         label=f'{variant} (Any)', linewidth=2, alpha=0.8, marker='o')
            ax3.plot(steps, [r * 100 for r in perfect_rates], 
                         label=f'{variant} (Perfect)', linewidth=2, alpha=0.6, marker='s', linestyle='--')
    ax3.set_title("Success Rate", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Training Step", fontsize=12)
    ax3.set_ylabel("Success Rate (%)", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    ax4 = axes[1, 1]
    variants_list = list(eval_results.keys())
    means = [eval_results[v][0] for v in variants_list]
    stds = [eval_results[v][1] for v in variants_list]
    x_pos = np.arange(len(variants_list))
    bars = ax4.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7)
    ax4.set_title("Final Evaluation", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Variant", fontsize=12)
    ax4.set_ylabel("Mean Reward", fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(variants_list, fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=200, color='g', linestyle='--', alpha=0.5)
    ax4.axhline(y=600, color='b', linestyle='--', alpha=0.5)
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{mean:.1f}±{std:.1f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(cfg["save_dir"], "training_analysis.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved training plot to {plot_path}")

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    for v, (m, s) in eval_results.items():
        print(f"{v:<15} {m:>12.2f} ± {s:>10.2f}")
    print("="*70)


if __name__ == "__main__":
    main()