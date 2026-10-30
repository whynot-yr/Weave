import gymnasium as gym

gym.register(
    id='G1-Inspire-Getup-v0',
    entry_point=f'{__name__}.env:GetupEnv',
    disable_env_checker=True,
    kwargs={'env_cfg_entry_point': f'{__name__}.env_cfg:GetupEnvCfg',
            'rsl_rl_cfg_entry_point': f'{__name__}.agents.rsl_rl_ppo_cfg:GetupRunnerCfg'},
)
