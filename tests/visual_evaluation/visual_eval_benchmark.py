from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder
from utils import evaluate_policy

log_dir = "logs/benchmark_hyperparam_stage_4"

model = PPO.load("models/benchmark_best.zip")

vec_eval_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=14)
vec_eval_env = VecFrameStack(vec_eval_env, n_stack=4)
vec_eval_env = VecTransposeImage(vec_eval_env)


episode_rewards, episode_lengths = evaluate_policy(
    model,
    vec_eval_env,
    n_eval_episodes=5,
    render=True,
    fps=200,
    return_episode_rewards=True,
)

# Log the detailed results
print(f"Episode rewards: {episode_rewards}")
print(f"Episode lengths: {episode_lengths}")