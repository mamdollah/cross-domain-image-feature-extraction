from stable_baselines3.common.callbacks import BaseCallback
import wandb


class WandbRewardLoggingCallback(BaseCallback):
    """
    Callback for logging the mean evaluation reward to Weights & Biases (wandb).

    Designed to be used with EvalCallback, it logs the mean reward after each evaluation.
    """

    def _on_step(self) -> bool:
        """
        Called by the EvalCallback after each evaluation to log evaluation metrics.

        Logs the mean_reward to wandb with the current number of timesteps as the step.
        """
        mean_reward = self.parent.last_mean_reward  # Access the mean reward from EvalCallback

        wandb.log({"eval/mean_reward": mean_reward}, step=self.num_timesteps)  # Log to wandb

        return True
