# main.py

import numpy as np
import config
from tor_circuit_env import CircuitEnv
from q_agent import QLearningAgent

def main():
    np.random.seed(config.RANDOM_SEED)

    env = CircuitEnv()

    agent = QLearningAgent(num_relays=config.NUM_RELAYS)

    print("Training...")
    print(f"Episodes: {config.NUM_EPISODES}")
    print(f"Relays: {config.NUM_RELAYS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epsilon: {config.EPSILON} -> {config.MIN_EPSILON}\n")

    for episode in range(config.NUM_EPISODES):
        obs, _ = env.reset()
        terminated = False
        episode_reward = 0
        steps = 0

        while not terminated:
            action = agent.policy(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)

            obs = next_obs
            episode_reward += reward
            steps += 1

        agent.decay_epsilon()

        if episode % config.LOG_FREQUENCY == 0:
            print(f"Episode {episode:5d}/{config.NUM_EPISODES} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Steps: {steps} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
    print("Training Complete!")
    print(f"Final Epsilon: {agent.epsilon:.3f}")

    if env.exit_relay is not None:
        entry_guard = env.relays[env.entry_guard]
        middle_relay = env.relays[env.middle_relay]
        exit_relay = env.relays[env.exit_relay]
        print("\nCircuit on Termination")   
        print(f"Entry Guard:  #{env.entry_guard:3d}: Bandwidth = {entry_guard['bandwidth']:6.2f} MB/s, Latency = {entry_guard['latency']:6.2f} ms")
        print(f"Middle Relay: #{env.middle_relay:3d}: Bandwidth = {middle_relay['bandwidth']:6.2f} MB/s, Latency = {middle_relay['latency']:6.2f} ms")
        print(f"Exit Relay: #{env.exit_relay:3d}: Bandwidth = {exit_relay['bandwidth']:6.2f} MB/s, Latency = {exit_relay['latency']:6.2f} ms")

        circuit_bandwidth = min(entry_guard['bandwidth'], middle_relay['bandwidth'], exit_relay['bandwidth'])
        circuit_latency = entry_guard['latency'] + middle_relay['latency'] + exit_relay['latency']

        print("\nCircuit Performance")
        print(f"Total Bandwidth: {circuit_bandwidth:.2f} MB/s")
        print(f"Total Latency: {circuit_latency:.2f} ms")
        print(f"Final Reward: {episode_reward:.2f}")
    else:
        print("\nLast episode failed")



if __name__ == "__main__":
    main()
