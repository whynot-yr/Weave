"""Short integration check: getup reset, terminal history, and one MuonPPO update."""
import argparse
import faulthandler
import traceback
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='outputs/getup_smoke')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json
from pathlib import Path
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from g1_hoi_learning.tasks.getup.env import GetupEnv
from g1_hoi_learning.tasks.getup.env_cfg import GetupEnvCfg
from g1_hoi_learning.algorithms.ppo.multi_critic_networks import MultiCriticSimBaActorCritic
from g1_hoi_learning.algorithms.ppo.multi_critic_algorithm import MultiCriticMuonPPO
from g1_hoi_learning.tasks.getup.agents.rsl_rl_ppo_cfg import GetupRunnerCfg
from g1_hoi_learning.tasks.getup.runner import GetupOnPolicyRunner as OnPolicyRunner


def main():
    faulthandler.dump_traceback_later(60, repeat=True)
    torch.set_num_threads(4)
    cfg = GetupEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.pose.randomize_joints = False
    raw = GetupEnv(cfg)
    env = RslRlVecEnvWrapper(raw)
    obs, _ = env.reset()
    # Exercise both seeds deterministically, then restore the real mixed sampler.
    for i, name in enumerate(('supine', 'prone')):
        cfg.pose.reset_mode = name
        raw._reset_idx(torch.tensor([i], device=env.device))
    cfg.pose.reset_mode = 'mixed'
    obs = env.get_observations()
    assert obs['robot_proprio'].shape == (2, 462)
    assert torch.allclose(obs['robot_proprio'][:, :154], obs['robot_proprio'][:, 154:308])
    robot = raw.scene['robot']
    state = raw.getup_state
    expected_root = state.seed_root_pos[state.episode_seed_id]
    torch.testing.assert_close(robot.data.root_pos_w-raw.scene.env_origins, expected_root)
    expected_seed = state.seed_joint_pos[state.episode_seed_id]
    torch.testing.assert_close(robot.data.joint_pos[:, state.drive], expected_seed)
    assert not torch.allclose(robot.data.default_joint_pos[:, state.drive], expected_seed)
    torch.testing.assert_close(state.action._offset, robot.data.default_joint_pos[:, state.drive])
    assert torch.count_nonzero(robot.data.joint_vel) == 0
    # Smaller width only for this integration test; training config remains WEAVE 2048.
    policy = MultiCriticSimBaActorCritic(
        obs, {'policy': ['robot_proprio'], 'critic': ['robot_proprio']}, 41,
        actor_hidden_dim=32, critic_hidden_dim=32, actor_num_blocks=2, critic_num_blocks=2,
        expansion=1, latent_dim=16, encoder_hidden_dims={'robot_proprio': {'type':'mlp', 'hidden_dims':[32]}},
        init_noise_std=.8, compile=False,
    ).to(env.device)
    alg = MultiCriticMuonPPO(policy, device=env.device, num_learning_epochs=1, num_mini_batches=1,
                            entropy_coef=.01, normalize_advantage_per_mini_batch=False)
    alg.init_storage('rl', 2, 2, obs, [41])
    with torch.inference_mode():
        for i in range(2):
            action = alg.act(obs)
            if i == 1:
                raw.episode_length_buf[0] = raw.max_episode_length-1
            obs, reward, done, extras = env.step(action)
            assert extras['reward_groups'].shape == (2,4)
            assert torch.isfinite(extras['reward_groups']).all()
            if i == 1:
                assert done[0] and extras['bootstrap_timeouts'][0]
                reset_hist = obs['robot_proprio'][0].reshape(3,154)
                assert torch.equal(reset_hist[0], reset_hist[1])
                assert not torch.equal(extras['terminal_proprio'][0], obs['robot_proprio'][0])
            alg.process_env_step(obs, reward, done, extras)
        alg.compute_returns(obs)
    metrics = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    runner_cfg = GetupRunnerCfg()
    assert runner_cfg.policy.actor_hidden_dim == runner_cfg.policy.critic_hidden_dim == 2048
    assert runner_cfg.policy.latent_dim == 256 and runner_cfg.policy.expansion == 1
    runner_cfg.policy.actor_hidden_dim = runner_cfg.policy.critic_hidden_dim = 32
    runner_cfg.policy.compile = False
    runner_cfg.num_steps_per_env = 2
    runner_cfg.algorithm.num_learning_epochs = runner_cfg.algorithm.num_mini_batches = 1
    print('[GETUP SMOKE] Construct native runner', flush=True)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(path/'runner'), device=env.device)
    print('[GETUP SMOKE] Native runner learn', flush=True)
    runner.learn(1, init_at_random_ep_len=False)
    checkpoint = str(path/'smoke.pt')
    runner.save(checkpoint)
    probe = env.get_observations()
    before = runner.get_inference_policy()(probe).detach().clone()
    runner.load(checkpoint)
    torch.testing.assert_close(before, runner.get_inference_policy()(probe))
    (path/'result.json').write_text(json.dumps({'passed':True, 'metrics':metrics}, indent=2))
    print('[GETUP SMOKE] PASSED', metrics, flush=True)
    faulthandler.cancel_dump_traceback_later()
    env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
        app.close(wait_for_replicator=False)
