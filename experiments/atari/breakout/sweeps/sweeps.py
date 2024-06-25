import datetime
import multiprocessing
import random
import time
import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from wandb.integration.sb3 import WandbCallback
from feature_extraction.callbacks.wandb_on_training_end_callback import WandbOnTrainingEndCallback
from feature_extraction.feature_extractors.resnet.block_feature_extractor import BlockFeatureExtractor
from feature_extraction.wrappers.vec_feature_extractor import VecFeatureExtractor
from utils import linear_schedule, make_resnet_atari_env

from torchvision.models import ResNet50_Weights, resnet50
import wandb
import sys

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO experiments with dynamic configurations.")
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--clip_range', type=float)
    parser.add_argument('--ent_coef', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--n_steps', type=int)
    parser.add_argument('--normalize_advantage', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--vf_coef', type=float)
    return parser.parse_args()

# Function to create a configuration based on parsed arguments
def create_config_from_args(args):
    project_name = "ablation_study_sweep"
    run_name = "block16_sweep" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config = {
        'project_name': project_name,
        'run_name': run_name,
        'env_id': "BreakoutNoFrameskip-v4",
        'n_envs': 8,
        'env_wrapper': ['stable_baselines3.common.atari_wrappers.AtariWrapper'],
        'frame_stack': 4,
        'training_seed': 12,
        'evaluation_seed': 14,
        'algo': 'PPO',
        'policy': 'MlpPolicy',
        'batch_size': args.batch_size,
        'n_steps': args.n_steps,
        'n_epochs': args.n_epochs,
        'n_timesteps': 1_000_000,
        'learning_rate': args.learning_rate,
        'clip_range': args.clip_range,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'normalize_advantage': args.normalize_advantage,
        'n_eval_episodes': 10,
        'record_n_episodes': 10,
        'n_final_eval_episodes': 20,
        'log_frequency': 25_000,
        'verbose': 0,
        'block_nbr': 16  # Example value, modify as needed
    }
    return config

def run_experiment(config):
    wandb.login()

    # Initialize the wandb run
    wandb.init(project=config['project_name'],
               name=config['run_name'],
               config=config, save_code=True,
               sync_tensorboard=True)

    config = wandb.config

    log_dir = f"logs/{config.run_name}"
    # feature_extractor = StageFeatureExtractor()

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model.to(device)
    feature_extractor = BlockFeatureExtractor(model, config.block_nbr, log_dir, log_to_wandb=True)

    # Create Evaluation Environment
    vec_eval_env = make_resnet_atari_env(
        config.env_id,
        n_envs=config.n_envs,
        seed=config.evaluation_seed,
    )

    print("------- PRINTING WRAPPER OBSERVATION SPACES ------")
    print("original_observation_space", vec_eval_env.observation_space.shape)
    vec_eval_env = VecTransposeImage(vec_eval_env)
    print("vec_transpose_obs_space", vec_eval_env.observation_space.shape)

    vec_eval_env = VecFrameStack(vec_eval_env, n_stack=config.frame_stack)
    print("vec_frame_stack_obs_space", vec_eval_env.observation_space.shape)

    vec_eval_env = VecFeatureExtractor(vec_eval_env, feature_extractor, n_stacks=config.frame_stack)
    print("vec_feature_extractor_obs_space", vec_eval_env.observation_space.shape)
    print("------- FINISHED PRINTING WRAPPER OBSERVATION SPACES ------")

    # Create Training Environment
    vec_train_env = make_resnet_atari_env(config.env_id, n_envs=config.n_envs, seed=config.training_seed)
    vec_train_env = VecTransposeImage(vec_train_env)
    vec_train_env = VecFrameStack(vec_train_env, n_stack=config.frame_stack)
    vec_train_env = VecFeatureExtractor(vec_train_env, feature_extractor, n_stacks=config.frame_stack)

    # Define the keys for PPO-specific hyperparameters
    ppo_params_keys = [
        'batch_size',
        'ent_coef',
        'n_epochs',
        'n_steps',
        'policy',
        'vf_coef',
        'normalize_advantage',
    ]

    # Filter the config dictionary to extract only the PPO hyperparameters
    ppo_hyperparams = {key: config[key] for key in ppo_params_keys if key in config}

    # Additional hyperparameters not in the initial filter that require custom handling
    learning_rate_schedule = linear_schedule(config.learning_rate)
    clip_range_schedule = linear_schedule(config.clip_range)

    # Instantiate the PPO model with the specified hyperparameters and environment
    model = PPO(
        **ppo_hyperparams,
        learning_rate=learning_rate_schedule,
        clip_range=clip_range_schedule,
        env=vec_train_env,
        verbose=1,
        tensorboard_log=f"{log_dir}",
    )

    # Save best model
    eval_callback = EvalCallback(
        eval_env=vec_eval_env,
        eval_freq=max(config.log_frequency // config.n_envs, 1),
        n_eval_episodes=config.n_eval_episodes,
        best_model_save_path=log_dir,
        log_path=log_dir,
        deterministic=True,
        render=False,
        verbose=0
    )

    # Needs to be changed, so it uses run instead of wandb
    wandb_callback = WandbCallback(
        verbose=1,
        gradient_save_freq=256,
    )

    wandb_on_training_end_callback = WandbOnTrainingEndCallback(
        model=model,
        eval_env=vec_eval_env,
        log_dir=log_dir,
        n_eval_episodes=config.n_final_eval_episodes,
        record_n_episodes=config.record_n_episodes,
    )
    callbacks = CallbackList([wandb_callback, eval_callback, wandb_on_training_end_callback])

    model.learn(
        total_timesteps=config.n_timesteps,
        callback=callbacks,
    )

    wandb.finish()

# Main function that uses the parsed arguments to run experiments
def main():
    args = parse_args()
    config = create_config_from_args(args)
    run_experiment(config)

if __name__ == "__main__":
    main()
