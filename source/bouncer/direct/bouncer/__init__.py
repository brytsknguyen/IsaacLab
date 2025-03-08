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
    id="Bouncer",
    entry_point=f"{__name__}.bouncer:BouncerGo1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bouncer:BouncerGo1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.bouncer:BouncerGo1PPORunnerCfg",
    },
)