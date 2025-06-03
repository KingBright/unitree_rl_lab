import argparse
import os
import torch # Good practice, though not directly used for config here

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

# Import helpers for loading environment and rl_games runner
# These paths might differ based on Isaac Lab version / installation structure
# Assuming omni.isaac.lab_tasks is available for these utilities
RL_GAMES_RUNNER_AVAILABLE = False
try:
    # Attempt to import the RlGamesRunner designed for Isaac Lab tasks
    from omni.isaac.lab_tasks.utils.runners import RlGamesRunner
    RL_GAMES_RUNNER_AVAILABLE = True
    print("Using RlGamesRunner from omni.isaac.lab_tasks.utils.runners")
except ImportError:
    print("Warning: Could not import RlGamesRunner from omni.isaac.lab_tasks. "
          "This script might need adjustments for direct rl_games.torch_runner usage, "
          "or ensure omni.isaac.lab_tasks is correctly installed and in PYTHONPATH.")
    # As a fallback, one might try to use the direct rl_games runner,
    # but it requires manual environment wrapping (e.g., with RlGamesGpuEnv, RlGamesVecEnvWrapper).
    # from rl_games.torch_runner import Runner as RlGamesRunner # Direct rl_games runner needs manual env setup

# Environment specific imports
# Assuming robot_isaac_lab_env.py is in the same directory or PYTHONPATH
try:
    # Import the specific environment configuration instances and the base class for type hinting if needed.
    from robot_isaac_lab_env import h1_env_cfg, g1_env_cfg, RobotIsaacLabEnvCfg
except ImportError:
    print("Error: Could not import environment configurations from robot_isaac_lab_env.py.")
    print("Please ensure 'robot_isaac_lab_env.py' is in the current directory or accessible in PYTHONPATH.")
    exit(1) # Critical import failed

# Default path for the PPO configuration file
DEFAULT_PPO_CONFIG = "./ppo_config.yaml" # Generic PPO config, robot specifics handled by env_cfg

def main(args):
    # Select environment config based on --robot argument
    if args.robot == "h1":
        selected_env_cfg_name = "h1_env_cfg" # This is the variable name in robot_isaac_lab_env.py
        task_name = "H1_IsaacLab"
    elif args.robot == "g1":
        selected_env_cfg_name = "g1_env_cfg" # This is the variable name in robot_isaac_lab_env.py
        task_name = "G1_IsaacLab"
    else:
        # This should not happen if choices are correctly enforced by argparse
        raise ValueError(f"Unsupported robot: {args.robot}. This should be caught by argparse.")

    env_cfg_entry_point = f"robot_isaac_lab_env:{selected_env_cfg_name}"

    # Launch Isaac Sim application
    # The RlGamesRunner or the environment itself might handle simulation_app context,
    # but it's good practice to initialize it early.
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app # Keep a reference, might be passed to runner or used by env

    # --- PPO Configuration Check ---
    if not os.path.exists(args.ppo_config):
        print(f"Error: PPO config file not found at '{args.ppo_config}'")
        simulation_app.close()
        return

    # task_name is now determined by the --robot argument

    if not RL_GAMES_RUNNER_AVAILABLE:
        print(f"Error: Isaac Lab specific RlGamesRunner is not available. Cannot proceed with training for {args.robot}.")
        print("Please ensure omni.isaac.lab_tasks is installed and accessible.")
        simulation_app.close()
        return

    # --- Initialize RlGamesRunner (from omni.isaac.lab_tasks.utils.runners) ---
    # This runner is specifically designed for Isaac Lab environments.
    # It handles environment creation, wrapping, and rl_games boilerplate.
    try:
        # Configuration for the RlGamesRunner itself
        runner_cfg_params = {
            "train": True,  # Specify that we are training
            "load_run": args.checkpoint if args.checkpoint else -1, # -1 for last run, or specify path/iteration
            "checkpoint": args.checkpoint if args.checkpoint else None, # Path to checkpoint to load
            "sigma": None,  # Initial noise sigma for actions (if applicable, PPO usually learns sigma)
            # The runner will load the algorithm_config from args.ppo_config internally
        }

        # The RlGamesRunner from lab_tasks.utils will handle environment creation
        # using the provided task_name and env_cfg_entry_point.
        # It also sets up the VecEnvWrapper (RlGamesVecEnvWrapper).
        runner = RlGamesRunner(
            algo_cfg_path=args.ppo_config,     # Path to the YAML config file for rl_games agent
            runner_cfg=runner_cfg_params,      # Runner specific settings (train, load, checkpoint)
            env_cfg_entry_point=env_cfg_entry_point, # Use dynamically selected entry point
            task_name=task_name,               # Use dynamically selected task name
            # Optional: Pass simulation_app if the runner requires it explicitly.
            # Some runner versions might pick it up from the global AppLauncher context.
            # simulation_app=simulation_app
        )
    except Exception as e:
        print(f"Error initializing RlGamesRunner: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return

    # --- Launch Training ---
    try:
        print(f"Launching training for robot: {args.robot}, task: {task_name} with PPO config: {args.ppo_config}")
        # The run() method of RlGamesRunner starts the training process.
        # It typically handles the entire training loop, including environment steps,
        # agent updates, logging, and checkpointing.
        runner.run()
        print(f"Training for {args.robot} finished.")
    except Exception as e:
        print(f"An error occurred during training for {args.robot}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: Close the simulation application
        print(f"Closing simulation application for {args.robot} training.")
        simulation_app.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a robot (H1 or G1) in Isaac Lab using rl_games and PPO.")
    parser.add_argument(
        "--robot", type=str, required=True, choices=["h1", "g1"],
        help="Name of the robot to train (h1 or g1)."
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run Isaac Sim in headless mode (no GUI)."
    )
    parser.add_argument(
        "--ppo_config", type=str, default=DEFAULT_PPO_CONFIG,
        help=f"Path to the PPO YAML configuration file (default: {DEFAULT_PPO_CONFIG})."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a checkpoint directory or specific .pth file to resume training from (e.g., 'runs/MyExperiment/nn/MyExperiment')."
    )
    # Note: num_envs, seed, and other rl_games specific parameters are typically handled
    # by the PPO configuration file (ppo_config.yaml) or overridden via
    # rl_games' own command-line argument parsing if the RlGamesRunner relays them.

    cli_args = parser.parse_args()
    main(cli_args)
