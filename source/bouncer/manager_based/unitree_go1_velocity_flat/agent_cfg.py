# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

import os
import math
import torch
from datetime import datetime
from dataclasses import MISSING
from typing import Literal
import numpy as np
from numpy.random import uniform as runf
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

# Simulator
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils

# Utilities
from isaaclab.utils import configclass
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# import isaaclab.utils.math as math_utils

# Scene
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Terrain
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# Assets
from isaaclab_assets.unitree import UNITREE_GO1_CFG  # isort: skip
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
# import omni.isaac.core.utils.prims as prim_utils

# RL's components
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

# Task
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Environment
# from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow, BaseEnvWindow

# Simulator
from isaaclab.sim import SimulationCfg

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


# Learning agent
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


# region Defining the agent config -------------------------------------------------------------------------------------------------------------------------------------

@configclass
class UnitreeGo1FlatPPORunnerCfg():

    # region Attributes from RslRlOnPolicyRunnerCfg -------------------------------------------------------------------

    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    # endregion Attributes from RslRlOnPolicyRunnerCfg ----------------------------------------------------------------


    # region Attributes from UnitreeGo1RoughPPORunnerCfg --------------------------------------------------------------

    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    experiment_name = "unitree_go1_flat"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(init_noise_std=1.0,
                                    actor_hidden_dims=[128, 128, 128],
                                    critic_hidden_dims=[128, 128, 128],
                                    activation="elu")

    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0,
                                     use_clipped_value_loss=True,
                                     clip_param=0.2,
                                     entropy_coef=0.01,
                                     num_learning_epochs=5,
                                     num_mini_batches=4,
                                     learning_rate=1.0e-3,
                                     schedule="adaptive",
                                     gamma=0.99,
                                     lam=0.95,
                                     desired_kl=0.01,
                                     max_grad_norm=1.0)
    
    # endregion Attributes from UnitreeGo1RoughPPORunnerCfg -----------------------------------------------------------

# endregion Defining the agent config ----------------------------------------------------------------------------------------------------------------------------------
