# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# from . import agents

##
# Register Gym environments.
##

gym.register(id="Isaac-Velocity-Flat-Unitree-Go1-NoBS",
             entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f'{__name__}.env_cfg:UnitreeGo1FlatEnvCfg',
                     "rsl_rl_cfg_entry_point": f'{__name__}.agent_cfg:UnitreeGo1FlatPPORunnerCfg'})

gym.register(id="Isaac-Velocity-Flat-Unitree-Go1-NoBS-Play",
             entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f'{__name__}.env_cfg:UnitreeGo1FlatEnvCfg_PLAY',
                     "rsl_rl_cfg_entry_point": f'{__name__}.agent_cfg:UnitreeGo1FlatPPORunnerCfg'})