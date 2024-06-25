import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from wandb.integration.sb3 import WandbCallback
from feature_extraction.callbacks.wandb_on_training_end_callback import WandbOnTrainingEndCallback
from feature_extraction.feature_extractors.resnet.block_feature_extractor import BlockFeatureExtractor
from feature_extraction.wrappers.vec_feature_extractor import VecFeatureExtractor
from utils import linear_schedule, make_resnet_atari_env
from collections import OrderedDict

from torchvision.models import ResNet50_Weights, resnet50
from utils import evaluate_policy

log_dir = "logs/benchmark_hyperparam_stage_4"


model = PPO.load("models/benchmark_hyperparam_stage_4.zip")

resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)
feature_extractor = BlockFeatureExtractor(resnet_model, 16, log_dir)

# Create Evaluation Environment
vec_eval_env = make_resnet_atari_env(
    "BreakoutNoFrameskip-v4",
    n_envs=1,
    seed=14,
)


vec_eval_env = VecTransposeImage(vec_eval_env)

vec_eval_env = VecFrameStack(vec_eval_env, n_stack=4)

vec_eval_env = VecFeatureExtractor(vec_eval_env, feature_extractor, n_stacks=4)

episode_rewards, episode_lengths = evaluate_policy(
    model,
    vec_eval_env,
    n_eval_episodes=5,
    render=True,
    fps=1000,
    return_episode_rewards=True
)

# Log the detailed results
print(f"Episode rewards: {episode_rewards}")
print(f"Episode lengths: {episode_lengths}")