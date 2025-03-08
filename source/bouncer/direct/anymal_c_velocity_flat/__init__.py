# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

# from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-Direct-NoBS",
    entry_point=f"{__name__}.anymal_c_env:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_env:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.anymal_c_env:AnymalCFlatPPORunnerCfg",
    },
)