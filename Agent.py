import torch
import pygame
import sys
from rocket_env import SimpleRocketEnv
from train import DQNAgent

MODEL_PATH = "models/DuelingDQN_model.pt"
VARIANT = "DuelingDQN" # or DQN / DoubleDQN
FRAME_RATE = 50
SEPARATOR = "=" * 50

def initialize_agent_and_env(model_path: str, variant: str):
    env = SimpleRocketEnv(render_mode="human")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, variant=variant)
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        agent.online_net.load_state_dict(checkpoint["online_state"])
        agent.target_net.load_state_dict(checkpoint["target_state"])
    except FileNotFoundError:
        print(f"Error: Model file tidak ditemukan di {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        sys.exit(1)

    print(f"Agen {variant} berhasil dimuat dari {model_path}.")
    return env, agent

def print_episode_summary(episode_num: int, step_count: int, total_reward: float):
    print(f"\n{SEPARATOR}")
    print(f"Episode {episode_num} Complete!")
    print(f"Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"{SEPARATOR}\n")

def main():
    env, agent = initialize_agent_and_env(MODEL_PATH, VARIANT)
    
    state, _ = env.reset()
    done = False
    truncated = False
    clock = pygame.time.Clock()
    running = True

    episode_num = 1
    step_count = 0
    total_reward = 0.0

    print(f"\n{SEPARATOR}")
    print(f"Starting Episode {episode_num}")
    print(f"{SEPARATOR}\n")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                done = True
                truncated = True
        
        if not (done or truncated):
            action = agent.act(state, epsilon=0.0)
            
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            state = next_state
            
            if done or truncated:
                print_episode_summary(episode_num, step_count, total_reward)
                
                episode_num += 1
                step_count = 0
                total_reward = 0.0
                state, _ = env.reset()
                done = False
                truncated = False
                
                print(f"Starting Episode {episode_num}")
                print(f"{SEPARATOR}\n")
        
        env.render()
        clock.tick(FRAME_RATE)

    env.close()
    print("\n" + SEPARATOR)
    print("Game ended.")
    print(f"Total episodes completed: {episode_num - 1}")
    print(SEPARATOR)

if __name__ == "__main__":
    main()