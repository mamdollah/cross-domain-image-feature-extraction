import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="rgb_array")

episodes = 10

for episode in range(1, episodes+1):
    observation, info = env.reset(seed=42)
    finished = False
    accumulated_reward = 0

    while not finished:
        frame = env.render()
        print(frame.shape)

        time.sleep(0.01)  # Add a small delay to slow down rendering
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        accumulated_reward += reward
        finished = terminated or truncated

        if terminated or truncated:
            observation, info = env.reset()
            print("Episode {} terminated".format(episode))
            print("Accumulated Reward: {}".format(accumulated_reward))

env.close()
