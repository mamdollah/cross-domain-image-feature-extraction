import os
import datetime
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder

from wandb.integration.sb3 import WandbCallback

from utils import linear_schedule
from feature_extraction.callbacks.wandb_on_training_end_callback import WandbOnTrainingEndCallback
from collections import OrderedDict

progress_bar = True
project_name = "ablation_study"
#Human readable timestamp
timestamp = datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

run_name = "experiment2" + timestamp
log_dir = "logs"

wandb.login()

config = OrderedDict([
    # Environment settings
    ('env_id', "BreakoutNoFrameskip-v4"),
    ('n_envs', 8),
    ('env_wrapper', ['stable_baselines3.common.atari_wrappers.AtariWrapper']),
    ('frame_stack', 4),
    ('training_seed', 12),
    ('evaluation_seed', 14),

    # Algorithm and policy
    ('algo', 'PPO'),
    ('policy', 'CnnPolicy'),

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

    # Evaluation and logging
    ('n_eval_episodes', 5),
    ('n_final_eval_episodes', 25),
    ('record_n_episodes', 10),
    ('log_frequency', 50_000),

    # Other settings
    ('verbose', 1)
])
root_logdir = os.getcwd()
#wandb.tensorboard.patch(root_logdir=root_logdir)

wandb.init(
    project=project_name,
    name=run_name,  # Name of the run
    config=config,
    save_code=True,
    sync_tensorboard=True,
    #monitor_gym=True,
)

config = wandb.config

# # Create Evaluation Environment


from utils import make_custom_atari_wrapper

vec_eval_env = make_custom_atari_wrapper(config.env_id, n_envs=config.n_envs, seed=config.evaluation_seed)

vec_eval_env = VecFrameStack(vec_eval_env, n_stack=config.frame_stack)
vec_eval_env = VecTransposeImage(vec_eval_env)

# # Create Training Environment

vec_train_env = make_custom_atari_wrapper(config.env_id, n_envs=config.n_envs, seed=config.training_seed)
vec_train_env = VecFrameStack(vec_train_env, n_stack=config.frame_stack)
vec_train_env = VecTransposeImage(vec_train_env)

# # Create Model


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

# Create Callbacks
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
    gradient_save_freq=config.log_frequency,
)

wandb_on_training_end_callback = WandbOnTrainingEndCallback(
    model=model,
    eval_env=vec_eval_env,
    log_dir=log_dir,
    n_eval_episodes=config.n_final_eval_episodes,
    record_n_episodes=config.record_n_episodes,
)
callbacks = CallbackList([wandb_callback, eval_callback, wandb_on_training_end_callback])

# # Train Model with callbacks
model.learn(
    total_timesteps=config.n_timesteps,
    callback=callbacks,
)

# # Cleanup
wandb.finish()
