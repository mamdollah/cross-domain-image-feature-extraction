import os
import sys

import torch
from torchinfo import summary
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder
import wandb


class WandbOnTrainingEndCallback(BaseCallback):
    def __init__(self, model, eval_env, wandb_run, log_dir, n_eval_episodes, record_n_episodes, verbose=0):
        super(WandbOnTrainingEndCallback, self).__init__(verbose)
        self.model = model
        self.eval_env = eval_env
        self.wandb_run = wandb_run
        self.log_dir = log_dir
        self.n_eval_episodes = n_eval_episodes
        self.record_n_episodes = record_n_episodes

    def record_best_model(self):
        """
        Record a video of the best model playing.

        :param record_steps: Number of steps to record.
        """
        video_folder = os.path.join(self.log_dir, "evaluation_videos")
        os.makedirs(video_folder, exist_ok=True)

        # Assuming the environment is already vectorized, wrap it with VecVideoRecorder
        # If it's not vectorized, you might need to wrap it in a DummyVecEnv or similar
        video_env = VecVideoRecorder(self.eval_env, video_folder,
                                     record_video_trigger=lambda x: True, # Start recording immediately
                                     video_length=sys.maxsize,  # Ensures all episodes are recorded
                                     name_prefix="evaluation")

        # Load the best model
        best_model_path = f"{self.log_dir}/best_model.zip"
        self.model.load(best_model_path)

        evaluate_policy(self.model, video_env, n_eval_episodes=self.record_n_episodes)

        video_env.close()

        # Upload the recorded video to W&B
        video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")]
        if video_files:
            latest_video_file = max(video_files, key=os.path.getmtime)  # Get the latest video file
            self.wandb_run.log({"recorded_video": wandb.Video(latest_video_file, fps=4, format="mp4")})

    def evaluate_model(self):
        """
        Evaluate the model using the best saved model and log the results.
        """
        # Load the best model - assumes EvalCallback has already saved the best_model to a file
        best_model_path = f"{self.log_dir}/best_model.zip"

        self.model.load(best_model_path)

        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)

        # Log the results to W&B
        self.wandb_run.log({"best_model/mean_reward": mean_reward, "best_model/std_reward": std_reward})

    def log_model_architecture(self):
        """
        Logs the model architecture to Weights & Biases using torchinfo.summary,
        directly as an HTML object for detailed analysis.
        """
        # Generate model summary
        model_summary = summary(self.model, input_size=(1, 4, 84, 84), verbose=0, depth=3)

        # Convert the summary to HTML string (basic conversion)
        # You might want to enhance HTML formatting based on your needs
        model_summary_html = f"<pre>{model_summary}</pre>"

        # Log the model architecture directly as an HTML object to W&B
        self.wandb_run.log({"model_architecture_html": wandb.Html(model_summary_html)})

    def _on_step(self) -> bool:
        # Continue training without interruption.
        return True

    def _on_training_end(self) -> None:
        final_model_path = f"{self.log_dir}/final_model.zip"
        onnx_model_path = f"{self.log_dir}/model.onnx"
        evaluations_path = f"{self.log_dir}/evaluations.npz"

        # Save the final model locally
        self.model.save(final_model_path)

        # Convert the model to ONNX format
        dummy_input = torch.randn(1, 4, 84, 84)  # Example input format
        torch.onnx.export(self.model.policy,  # Model's policy to export
                          dummy_input,  # Example input for the model
                          onnx_model_path)  # Path to save the ONNX model

        # Evaluate the model
        self.evaluate_model()
        self.record_best_model()

        self.log_model_architecture()
        # Use wandb_run for saving files to W&B
        self.wandb_run.save(final_model_path, policy="end")
        # Assumes EvalCallback has already saved the evaluations to a file
        self.wandb_run.save(evaluations_path, policy="end")
        self.wandb_run.save(onnx_model_path, policy="end")