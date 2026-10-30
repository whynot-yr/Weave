"""Numerical multi-critic and history checks without starting Isaac Sim."""
import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import pytest
import torch
from tensordict import TensorDict

PACKAGE = Path(__file__).resolve().parents[1] / 'g1_hoi_learning'


@pytest.fixture
def modules(monkeypatch):
    torch.set_num_threads(2)
    for suffix in ('', '.algorithms', '.algorithms.ppo'):
        name = 'g1_hoi_learning'+suffix
        mod = ModuleType(name)
        mod.__path__ = [str(PACKAGE.joinpath(*suffix.strip('.').split('.')))] if suffix else [str(PACKAGE)]
        monkeypatch.setitem(sys.modules, name, mod)
    def load(suffix):
        name = 'g1_hoi_learning.'+suffix
        spec = importlib.util.spec_from_file_location(name, PACKAGE.joinpath(*suffix.split('.')).with_suffix('.py'))
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod
    for suffix in ('algorithms.networks', 'algorithms.optimizers', 'algorithms.ppo.networks', 'algorithms.ppo.algorithm'):
        load(suffix)
    return SimpleNamespace(storage=load('algorithms.ppo.multi_critic_storage'),
                           network=load('algorithms.ppo.multi_critic_networks'),
                           algorithm=load('algorithms.ppo.multi_critic_algorithm'))


def test_group_gae_and_weights(modules):
    obs = TensorDict({'robot_proprio': torch.zeros(2, 462)}, batch_size=[2])
    storage = modules.storage.MultiCriticRolloutStorage('rl', 2, 2, obs, [41], reward_group_weights=[2.5,1,.1,1])
    storage.step = 2
    storage.values[:] = 0
    storage.rewards[0] = torch.tensor([[1,2,3,4],[2,4,6,8.]])
    storage.rewards[1] = torch.tensor([[3,6,9,12],[4,8,12,16.]])
    storage.dones[0, 0] = 1
    storage.compute_returns(torch.zeros(2,4), gamma=.9, lam=.8)
    expected = storage.rewards.clone()
    expected[0,1] += .9*.8*expected[1,1]
    torch.testing.assert_close(storage.returns, expected)
    normalized = (expected-expected.mean((0,1)))/(expected.std((0,1))+1e-8)
    torch.testing.assert_close(storage.advantages, (normalized*torch.tensor([2.5,1,.1,1])).sum(-1,keepdim=True))
    # Changing only style must leave all other return targets unchanged.
    before = storage.returns.clone()
    storage.rewards[...,1] += 7
    storage.compute_returns(torch.zeros(2,4), .9, .8)
    torch.testing.assert_close(storage.returns[..., [0,2,3]], before[..., [0,2,3]])


def make_policy(modules):
    obs = TensorDict({'robot_proprio': torch.randn(2,462)}, batch_size=[2])
    policy = modules.network.MultiCriticSimBaActorCritic(
        obs, {'policy':['robot_proprio'], 'critic':['robot_proprio']}, 41, compile=False,
        actor_hidden_dim=16, critic_hidden_dim=16, latent_dim=8,
        encoder_hidden_dims={'robot_proprio':{'type':'mlp','hidden_dims':[16]}},
        actor_obs_normalization=False, critic_obs_normalization=False,
    )
    return obs, policy


def test_critics_have_independent_gradients(modules):
    obs, policy = make_policy(modules)
    assert policy.evaluate(obs).shape == (2,4)
    policy.evaluate(obs)[:, 0].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.critic[0].parameters())
    assert all(p.grad is None or p.grad.count_nonzero()==0 for c in policy.critic[1:] for p in c.parameters())
    assert all(p.grad is None for p in policy.actor.parameters())


def test_timeout_uses_terminal_value_only(modules):
    obs, policy = make_policy(modules)
    alg = modules.algorithm.MultiCriticMuonPPO(policy, device='cpu', gamma=.9)
    alg.init_storage('rl', 2, 1, obs, [41])
    alg.act(obs)
    terminal = TensorDict({'robot_proprio': torch.full((2,462), .75)}, batch_size=[2])
    expected = policy.evaluate(terminal).detach()
    rewards = torch.ones(2,4)
    alg.process_env_step(obs, torch.zeros(2), torch.ones(2), {
        'reward_groups':rewards, 'bootstrap_timeouts':torch.tensor([True,False]),
        'terminal_proprio':terminal['robot_proprio'],
    })
    torch.testing.assert_close(alg.storage.rewards[0,0], rewards[0]+.9*expected[0])
    torch.testing.assert_close(alg.storage.rewards[0,1], rewards[1])


def test_history_and_continuous_success():
    path = PACKAGE/'tasks/getup/mdp/state.py'
    spec = importlib.util.spec_from_file_location('getup_state_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = mod.GetupState.__new__(mod.GetupState)
    s.env = SimpleNamespace(common_step_counter=0, cfg=SimpleNamespace(pose=SimpleNamespace(success_height=.76)))
    s.history = torch.zeros(2,3,154)
    s.pending = torch.ones(2,dtype=torch.bool)
    s.success_steps = torch.zeros(2,dtype=torch.long)
    s.success = torch.zeros(2,dtype=torch.bool)
    s.required_steps = 50
    s.last_step = -1
    frame = torch.zeros(2,154)
    frame[:,112] = .8
    s.frame = lambda: frame
    for i in range(49):
        s.env.common_step_counter += 1
        s.begin_step()
    assert not s.success.any()
    frame[1,112] = .7  # break the second environment's consecutive streak
    s.env.common_step_counter += 1
    s.begin_step()
    assert s.success.tolist() == [True,False]
    assert s.success_steps.tolist() == [50,0]
    before = s.history.clone()
    s.observation(); s.observation()
    torch.testing.assert_close(before, s.history)
    s.pending[1] = True
    frame[1] = 3
    result = s.observation().reshape(2,3,154)
    assert torch.equal(result[0], before[0])
    assert (result[1] == 3).all()


def test_group_time_scaling_and_terminal_snapshot():
    path = PACKAGE/'tasks/getup/reward_manager.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    class Base:
        @property
        def num_envs(self): return 2
        @property
        def device(self): return 'cpu'
    history = torch.randn(2,462)
    state = SimpleNamespace(begin_step=lambda: None, end_step=lambda: None,
                            observation=lambda: history.clone())
    terms = {f'{g}_test': torch.full((2,), float(i+1))
             for i,g in enumerate(('task','style','regu','post'))}
    ns = dict(torch=torch, RewardManager=Base, get_state=lambda env: state,
              compute_terms=lambda env: terms, GROUPS=('task','style','regu','post'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'), ns)
    manager = ns['GetupRewardManager'].__new__(ns['GetupRewardManager'])
    manager._env = SimpleNamespace(cfg=SimpleNamespace(advantage_weights=(2.5,1,.1,1)), extras={},
                                   reset_time_outs=torch.tensor([True,True]),
                                   reset_terminated=torch.tensor([False,True]))
    manager._term_names = list(terms)
    manager._term_cfgs = [SimpleNamespace(weight=1.) for _ in terms]
    manager._episode_sums = {k:torch.zeros(2) for k in terms}
    manager._step_reward = torch.zeros(2,4)
    manager._reward_buf = torch.zeros(2)
    manager.group_sums = torch.zeros(2,4)
    scalar = manager.compute(.02)
    expected = torch.tensor([[1.,.04,.06,.08]]).expand(2,-1)
    torch.testing.assert_close(manager._env.extras['reward_groups'],expected)
    torch.testing.assert_close(scalar,(expected*torch.tensor([2.5,1,.1,1])).sum(-1))
    assert manager._env.extras['bootstrap_timeouts'].tolist()==[True,False]
    history.zero_()
    assert manager._env.extras['terminal_proprio'].abs().sum()>0


def test_normalizer_checkpoint_after_inference_rollout(modules):
    obs = TensorDict({'robot_proprio':torch.randn(2,462)},batch_size=[2])
    policy = modules.network.MultiCriticSimBaActorCritic(
        obs, {'policy':['robot_proprio'],'critic':['robot_proprio']},41,compile=False,
        actor_hidden_dim=16,critic_hidden_dim=16,latent_dim=8,
        encoder_hidden_dims={'robot_proprio':{'type':'mlp','hidden_dims':[16]}},
    )
    with torch.inference_mode():
        policy.update_normalization(obs)
    state = {k:v.clone() for k,v in policy.state_dict().items()}
    policy.load_state_dict(state)
    assert not torch.is_inference(policy.actor_obs_normalizer._std)
