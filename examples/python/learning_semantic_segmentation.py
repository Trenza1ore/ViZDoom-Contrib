#!/usr/bin/env python3

#####################################################################
# Example script of training agents with stable-baselines3 on ViZDoom
# with the Gymnasium API, using semantic segmentation input
#
# Note: For this example to work, you need to install stable-baselines3:
#       pip install stable-baselines3
#       Optionally, to reduce usage, install sb3-extra-buffers:
#       pip install "sb3-extra-buffers[fast]"
#       To measure peak memory usage on Windows, install psutil
#       pip install psutil
#
# See more stable-baselines3 documentation here:
#   https://stable-baselines3.readthedocs.io/en/master/index.html
#####################################################################

import platform
import resource
from argparse import ArgumentParser

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

import vizdoom.gymnasium_wrapper  # noqa


DEFAULT_ENV = "VizdoomDefendLine-v1"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]  # type: ignore

# Training parameters
TRAINING_TIMESTEPS = int(4e6)
N_STEPS = 4096
N_ENVS = 4
BATCH_SIZE = 32
L_RATE = 1e-3
FRAME_SKIP = 4


class ObservationWrapper(gymnasium.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the Semantic Segmentation result in label form).

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """

    def __init__(self, env):
        super().__init__(env)
        print(env.observation_space)
        self.observation_space = env.observation_space["segmentation"]

    def observation(self, observation):
        observation = observation["segmentation"]
        return observation


def main(args):
    # Create multiple environments: this speeds up training with PPO
    # We apply two wrappers on the environment:
    #  1) The above wrapper that modifies the observations (takes only the semantic segmentation buffer)
    #  2) A reward scaling wrapper. Normally the scenarios use large magnitudes for rewards (e.g., 100, -100).
    #     This may lead to unstable learning, and we scale the rewards by 1/100
    def wrap_env(env):
        env = ObservationWrapper(env)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env

    # Use SubprocVecEnv for multi-processing parallelism
    envs = make_vec_env(
        args.env,
        n_envs=N_ENVS,
        wrapper_class=wrap_env,
        env_kwargs=dict(frame_skip=FRAME_SKIP, semantic_classes=("default", "label")),
    )
    envs = VecTransposeImage(envs)  # Wrap in VecTransposeImage to ensure correct shape

    # Repeat the same for evaluation environments
    eval_envs = make_vec_env(
        args.env,
        n_envs=N_ENVS,
        wrapper_class=wrap_env,
        env_kwargs=dict(frame_skip=FRAME_SKIP, semantic_classes=("default", "label")),
    )
    eval_envs = VecTransposeImage(
        eval_envs
    )  # Wrap in VecTransposeImage to ensure correct shape

    try:
        assert args.compress
        from sb3_extra_buffers.compressed import (
            CompressedRolloutBuffer,
            find_buffer_dtypes,
        )

        buffer_dtypes = find_buffer_dtypes(
            obs_shape=envs.unwrapped.observation_space.shape
        )
        agent = PPO(
            "CnnPolicy",
            envs,
            learning_rate=L_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            rollout_buffer_class=CompressedRolloutBuffer,
            rollout_buffer_kwargs=dict(
                compression_method=args.compress, dtypes=buffer_dtypes
            ),
            policy_kwargs=dict(normalize_images=False),
            verbose=1,
        )
    except (ImportError, AssertionError):
        agent = PPO(
            "CnnPolicy",
            envs,
            learning_rate=L_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            policy_kwargs=dict(normalize_images=False),
            verbose=1,
        )

    eval_callback = EvalCallback(eval_envs, n_eval_episodes=10)

    # Do the actual learning
    # This will print out the results in the console.
    # If agent gets better, "ep_rew_mean" should increase steadily
    try:
        agent.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            callback=eval_callback,
            progress_bar=True,
        )
    except ImportError:
        print(
            "Progress bar not available, install via `pip install tqdm rich` to enable"
        )
        agent.learn(total_timesteps=TRAINING_TIMESTEPS, callback=eval_callback)


if __name__ == "__main__":
    if platform.system().lower() != "windows":
        initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    else:
        initial_memory = 0

    parser = ArgumentParser(
        "Train stable-baselines3 PPO agents on ViZDoom via Semantic Segmentation."
    )
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        choices=AVAILABLE_ENVS,
        help="Name of the environment to play",
    )
    parser.add_argument(
        "-c",
        "--compress",
        default="zstd-5",
        type=str,
        help="Buffer compression option",
    )
    args = parser.parse_args()
    main(args)

    if platform.system().lower() == "windows":
        try:
            import psutil

            peak_memory_usage = psutil.Process().memory_info().peak_wset / 1024 / 1024
        except ImportError:
            peak_memory_usage = -1
    else:
        final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_memory_usage = (final_memory - initial_memory) / 1024
        if platform.system().lower() == "macos":
            peak_memory_usage /= 1024
    print(f"Peak Memory Usage: {round(peak_memory_usage)}MB")
