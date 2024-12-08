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
from omni.isaac.lab.sim import SimulationContext
import omni.isaac.lab.sim as sim_utils

# Utilities
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# import omni.isaac.lab.utils.math as math_utils

# Scene
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg

# Terrain
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# Assets
from omni.isaac.lab_assets.unitree import UNITREE_GO1_CFG  # isort: skip
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
# import omni.isaac.core.utils.prims as prim_utils

# RL's components
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

# Task
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Environment
# from omni.isaac.lab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from omni.isaac.lab.envs.common import ViewerCfg
from omni.isaac.lab.envs.ui import ManagerBasedRLEnvWindow, BaseEnvWindow

# Simulator
from omni.isaac.lab.sim import SimulationCfg

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


# Learning agent
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
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
