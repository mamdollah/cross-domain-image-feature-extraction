import datetime
import multiprocessing
import random
import time

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


def create_config(project_name, run_name, block_nbr):
    """
    Creates a new configuration as an OrderedDict with a specified project and run name,
    and sets the training_seed and evaluation_seed to random integers,
    while using a predefined set of default settings for the rest.

    Parameters:
    - project_name (str): Name of the project.
    - run_name (str): Name of the run.

    Returns:
    - OrderedDict: A new configuration dictionary with the specified project and run names,
                   random seeds, and default settings for other parameters.
    """
    training_seed = random.randint(0, 9999)  # Should later on be same for all experiments to ensure comparability
    evaluation_seed = random.randint(0, 9999)  # Should later on be same for all experiments to ensure comparability

    default_config = OrderedDict([
        # Environment settings
        ('project_name', project_name),
        ('run_name', run_name),
        ('env_id', "BreakoutNoFrameskip-v4"),
        ('n_envs', 8),
        ('env_wrapper', ['stable_baselines3.common.atari_wrappers.AtariWrapper']),
        ('frame_stack', 4),
        ('training_seed', 12),
        ('evaluation_seed', 14),

        # Algorithm and policy settings
        ('algo', 'PPO'),
        ('policy', 'MlpPolicy'),

        # Training hyperparameters
        ('batch_size', 256),
        ('n_steps', 128),
        ('n_epochs', 4),
        ('n_timesteps', 10_000_000),
        ('learning_rate', 0.00025),
        ('learning_rate_schedule', 'linear'),
        ('clip_range', 0.1),
        ('clip_range_schedule', 'linear'),
        ('ent_coef', 0.01),
        ('vf_coef', 0.5),
        ('normalize_advantage', False),

        # Resnet
        ('block_nbr', block_nbr),
        # Evaluation and logging settings
        ('n_eval_episodes', 5),
        ('record_n_episodes', 10),
        ('n_final_eval_episodes', 25),
        ('log_frequency', 50_000),

        # Other settings
        ('verbose', 1)
    ])

    return default_config


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
    learning_rate_schedule = linear_schedule(2.5e-4)
    clip_range_schedule = linear_schedule(0.1)

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

def run_experiments_in_parallel(config_list):
    processes = []
    for config in config_list:
        p = multiprocessing.Process(target=run_experiment, args=(config,))
        p.start()
        processes.append(p)
        time.sleep(60)  # Sleep to avoid file write conflicts

    for p in processes:
        p.join()


configs = []

block_nbrs = [16]
runs = []

for block_nbr in block_nbrs:
    runs.append((f"breakout_block{block_nbr}_avg", block_nbr))

for run_name, block_nbr in runs:
    timestamp = datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

    project_name = "ablation_study"
    run_name = "exp-7-skolan" + timestamp
    new_config = create_config(project_name, run_name, block_nbr)
    configs.append(new_config)

# Run experiments in sequentially
for config in configs:
    run_experiment(config)
