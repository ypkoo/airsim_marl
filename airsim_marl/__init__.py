from gym.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id='MultiagentOccupy-v0',
    entry_point='airsim_marl.envs:MultiAgentEnv',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)
