import os
import sys

import torch
from torchinfo import summary
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder
import wandb


class WandbOnTrainingEndCallback(BaseCallback):
    def __init__(self, model, eval_env, log_dir, n_eval_episodes, record_n_episodes, verbose=0):
        super(WandbOnTrainingEndCallback, self).__init__(verbose)
        self.model = model
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.n_eval_episodes = n_eval_episodes
        self.record_n_episodes = record_n_episodes
        # Before the torch.onnx.export call, add:
         # Get the device from the model's parameters
        self.dummy_input = torch.randn((1,) + model.observation_space.shape).to(model.policy.device) # For feature extraction


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
                                     video_length=2_000,  # Ensures all episodes are recorded
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
            wandb.log({"recorded_video": wandb.Video(latest_video_file, fps=4.0, format="mp4")})

    def evaluate_model(self):
        """
        Evaluate the model using the best saved model and log the results.
        """
        # Load the best model - assumes EvalCallback has already saved the best_model to a file
        best_model_path = f"{self.log_dir}/best_model.zip"

        self.model.load(best_model_path)

        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}, n_eval_episodes: {self.n_eval_episodes}")

        # Log the results to W&B
        wandb.log({"best_model/mean_reward": mean_reward, "best_model/std_reward": std_reward})

    def log_model_architecture(self):
        """
        Logs the model architecture to Weights & Biases using torchinfo.summary,
        directly as an HTML object for detailed analysis.
        """
        # Generate model summary
        model_summary = summary(self.model.policy, input_data=self.dummy_input)
        print("Model Summary:", model_summary)

        # Convert the summary to HTML string (basic conversion)
        # You might want to enhance HTML formatting based on your needs
        model_summary_html = f"<pre>{model_summary}</pre>"

        # Log the model architecture directly as an HTML object to W&B
        wandb.log({"model_architecture_html": wandb.Html(model_summary_html)})

    def _on_step(self) -> bool:
        # Continue training without interruption.
        return True

    def _on_training_end(self) -> None:
        final_model_path = f"{self.log_dir}/final_model.zip"
        onnx_model_path = f"{self.log_dir}/model.onnx"
        evaluations_path = f"{self.log_dir}/evaluations.npz"
        best_model_path = f"{self.log_dir}/best_model.zip"

        # Save the final model locally
        self.model.save(final_model_path)

        # Convert the model to ONNX format
        # Dummy input should be changed based for block runs
        torch.onnx.export(self.model.policy,  # Model's policy to export
                          self.dummy_input,  # Example input for the model
                          onnx_model_path)  # Path to save the ONNX model



        # Evaluate the model
        print("Evaluating model...")
        self.evaluate_model()
        print("Model evaluation done.")
        print("Logging model architecture...")
        self.log_model_architecture()
        print("Model architecture logged.")

        print("Uploading files to W&B...")
        # Assumes EvalCallback has already run
        wandb.save(best_model_path, policy="end")
        wandb.save(final_model_path, policy="end")
        wandb.save(evaluations_path, policy="end")
        wandb.save(onnx_model_path, policy="end")
        print("Files uploaded to W&B.")