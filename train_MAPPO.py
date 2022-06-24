# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import argparse
import math
import random
import sys
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Process
from pathlib import Path
from torch.distributions import Categorical
# Prevent numpy from using up all cpu
import os

from MAPPOPolicy import MAPPOPolicy

os.environ['MKL_NUM_THREADS'] = '6'  # pylint: disable=wrong-import-position

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', (
'agent_n', 'state', 'action_log_prob', 'value_pred', 'action', 'reward', 'next_state', 'cent_obs'))


class ReplayBuffer:
    def __init__(self, cfg, capacity=25):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self._use_gae = cfg.use_gae
        self._use_popart = cfg.use_popart
        self._use_valuenorm = cfg.use_valuenorm
        self._use_proper_time_limits = cfg.use_proper_time_limits

    def reset(self, done):
        self.position = 0
        self.buffer = []
        self.returns=[]
    def push(self, *args):
        # if len(self.buffer) < self.capacity:
        #     self.buffer.append(None)
        # self.buffer[self.position] = Transition(*args)
        # self.position = (self.position + 1) % self.capacity
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size) if len(self.buffer) > batch_size else self.buffer
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)

    def compute_returns(self, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if not self.buffer:
            return
        self.returns = [0 for _ in range(len(self.buffer)-1)]
        if self._use_gae:
            gae = 0
            for step in reversed(range(len(self.buffer)-1)):
                if self._use_popart or self._use_valuenorm:
                    delta = self.buffer[step].reward + self.gamma * value_normalizer.denormalize(
                        self.buffer[step + 1].value_pred) \
                            - value_normalizer.denormalize(self.buffer[step].value_pred)
                    gae = delta + self.gamma * self.gae_lambda * gae
                    self.returns[step] = gae + value_normalizer.denormalize(self.buffer[step].value_pred)
                else:
                    delta = self.buffer[step].reward + self.gamma * self.buffer[step + 1].value_pred - \
                            self.buffer[step].value_pred
                    gae = delta + self.gamma * self.gae_lambda * gae
                    self.returns[step] = gae + self.buffer[step].value_pred
        else:
            # self.returns[-1] = self.buffer[-1].value_pred
            for step in reversed(range(len(self.buffer) - 1)):
                self.returns[step] = self.returns[step + 1] * self.gamma + self.buffer[step].reward


class TransitionTracker:
    def __init__(self, initial_state):
        self.num_buffers = len(initial_state)
        self.prev_state = initial_state
        self.prev_action = [[None for _ in g] for g in self.prev_state]
        self.action_log_probs = [[None for _ in g] for g in self.prev_state]
        self.values = [[None for _ in g] for g in self.prev_state]
        self.prev_cent_obs = [[None for _ in g] for g in self.prev_state]

    def update_action(self, action, action_log_probs, values, cent_obs):
        for i, g in enumerate(action):
            for j, a in enumerate(g):
                if a is not None:
                    self.prev_action[i][j] = a
                    self.action_log_probs[i][j] = action_log_probs[i][j]
                    self.values[i][j] = values[i][j]
                    self.prev_cent_obs[i][j] = cent_obs[i]

    def update_step_completed(self, reward, state, done):
        transitions_per_buffer = [[] for _ in range(self.num_buffers)]
        for i, g in enumerate(state):
            for j, s in enumerate(g):
                if s is not None or done:
                    if self.prev_state[i][j] is not None:
                        transition = (j, self.prev_state[i][j], self.action_log_probs[i][j], self.values[i][j],
                                      self.prev_action[i][j], reward[i][j], s, self.prev_cent_obs[i][j])
                        transitions_per_buffer[i].append(transition)
                    self.prev_state[i][j] = s
        return transitions_per_buffer


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Meters:
    def __init__(self):
        self.meters = {}

    def get_names(self):
        return self.meters.keys()

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, name, val):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val)

    def avg(self, name):
        return self.meters[name].avg


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2
def cal_value_loss(all_args, values, value_preds_batch, return_batch,value_normalizer=None):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-all_args.clip_param,
                                                                                        all_args.clip_param)
        if all_args.use_popart or all_args.use_valuenorm:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if all_args.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, all_args.huber_delta)
            value_loss_original = huber_loss(error_original, all_args.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if all_args.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss
def process(share_obs_batch,critic_net,values):
    for cent_obs in share_obs_batch:#需要进行加速(多进程or?)
        tmp=critic_net(cent_obs)
        values.append(tmp)
def train(all_args, policy_net, optimizer, batch, transform_fn,
          critic_net, optimizers_critic, value_normalizer=None):
    share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ = [], [], [], [], [], [], []
    for buffer in batch:
        for i, return_ in enumerate(buffer.returns):  # 最后一组数据丢弃，用于当作next_value的计算（因本环境的特殊情况而设置）
            if all_args.use_popart or all_args.use_valuenorm:
                adv_targ.append(return_ - value_normalizer.denormalize(buffer.buffer[i].value_pred))
            else:
                adv_targ.append(return_ - buffer.buffer[i].value_pred)

            share_obs_batch.append(buffer.buffer[i].cent_obs)
            obs_batch.append(buffer.buffer[i].state)
            actions_batch.append(buffer.buffer[i].action)
            value_preds_batch.append(buffer.buffer[i].value_pred)
            return_batch.append(buffer.returns[i])
            old_action_log_probs_batch.append(buffer.buffer[i].action_log_prob)
    share_obs_batch=torch.cat([s.unsqueeze(0) for s in share_obs_batch]).to(device)
    obs_batch=torch.cat([transform_fn(s) for s in obs_batch if s is not None]).to(device,non_blocking=True)
    actions_batch=torch.tensor(actions_batch, dtype=torch.long).to(device)
    value_preds_batch=torch.tensor(value_preds_batch, dtype=torch.float32).view(-1,1).to(device)
    return_batch=torch.tensor(return_batch, dtype=torch.float32).view(-1,1).to(device)
    old_action_log_probs_batch=torch.tensor(old_action_log_probs_batch, dtype=torch.float32).view(-1,1).to(device)
    adv_targ=torch.tensor(adv_targ, dtype=torch.float32).view(-1,1).to(device)
    sample = share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ

    # Compute log probability and values and entropy of given actions
    action_logits=policy_net(obs_batch).view(1,-1)
    action_logits=FixedCategorical(logits=action_logits)
    action_log_probs=action_logits.log_probs(actions_batch)
    dist_entropy=action_logits.entropy().mean()

    # manager=Manager()
    # values=manager.list()
    values=[]
    # pool=ThreadPoolExecutor(max_workers=all_args.cpu_number)
    # lock=threading.Lock()
    p1=threading.Thread(target=process,args=(share_obs_batch,critic_net,values))
    p1.start()


    # values = torch.tensor(values, dtype=torch.float32,requires_grad=True).view(-1,1).to(device)

    # actor update
    imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

    surr1 = imp_weights * adv_targ
    surr2 = torch.clamp(imp_weights, 1.0 - all_args.clip_param, 1.0 + all_args.clip_param) * adv_targ

    policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

    policy_loss = policy_action_loss
    optimizer.zero_grad()

    (policy_loss - dist_entropy * all_args.entropy_coef).backward()

    if all_args.use_max_grad_norm:
        actor_grad_norm = nn.utils.clip_grad_norm_(policy_net.parameters(), all_args.max_grad_norm)
    else:
        actor_grad_norm = get_gard_norm(policy_net.parameters())
    optimizer.step()


    # critic update
    p1.join()
    values = torch.tensor(values, dtype=torch.float32,requires_grad=True).view(-1,1).to(device)

    value_loss = cal_value_loss(all_args, values, value_preds_batch, return_batch)

    optimizers_critic.zero_grad()

    (value_loss * all_args.value_loss_coef).backward()

    if all_args.use_max_grad_norm:
        critic_grad_norm = nn.utils.clip_grad_norm_(critic_net.parameters(), all_args.max_grad_norm)
    else:
        critic_grad_norm = get_gard_norm(critic_net.parameters())

    optimizers_critic.step()

    return dict()


def train_intention(intention_net, optimizer, batch, transform_fn):
    # Expects last channel of the state representation to be the ground truth intention map
    state_batch = torch.cat([transform_fn(s[:, :, :-1]) for s in batch.state]).to(device)  # (32, 4 or 5, 96, 96)
    target_batch = torch.cat([transform_fn(s[:, :, -1:]) for s in batch.state]).to(device)  # (32, 1, 96, 96)

    output = intention_net(state_batch)  # (32, 2, 96, 96)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(output, target_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_info = {}
    train_info['loss_intention'] = loss.item()

    return train_info

def main(cfg):
    from config.config import get_config
    parser = get_config()
    all_args = parser.parse_known_args()[0]

    # Set up logging and checkpointing
    log_dir = Path(cfg.log_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    print('log_dir: {}'.format(log_dir))
    print('checkpoint_dir: {}'.format(checkpoint_dir))

    # Create environment
    kwargs = {}
    if cfg.show_gui:
        import matplotlib  # pylint: disable=import-outside-toplevel
        matplotlib.use('agg')
    if cfg.use_predicted_intention:  # Enable ground truth intention map during training only
        kwargs['use_intention_map'] = True
        kwargs['intention_map_encoding'] = 'ramp'
    env = utils.get_env_from_cfg(cfg, **kwargs)

    robot_group_types = env.get_robot_group_types()
    num_robot_groups = len(robot_group_types)

    # Policy
    policy = MAPPOPolicy(cfg, train=True)

    # Optimizers
    optimizers = []
    for i in range(num_robot_groups):
        optimizers.append(optim.SGD(policy.policy_nets[i].parameters(), lr=cfg.learning_rate, momentum=0.9,
                                    weight_decay=cfg.weight_decay))

    optimizers_critic = []
    for i in range(num_robot_groups):
        optimizers_critic.append(optim.SGD(policy.critic_nets[i].parameters(), lr=cfg.learning_rate, momentum=0.9,
                                           weight_decay=cfg.weight_decay))
    if cfg.use_predicted_intention:
        optimizers_intention = []
        for i in range(num_robot_groups):
            optimizers_intention.append(
                optim.SGD(policy.intention_nets[i].parameters(), lr=cfg.learning_rate, momentum=0.9,
                          weight_decay=cfg.weight_decay))

    # Replay buffers
    replay_buffers = []
    for _ in range(num_robot_groups):
        replay_buffers.append([ReplayBuffer(all_args, 25) for _ in range(4 // num_robot_groups)])

    # Resume if applicable
    start_timestep = 0
    episode = 0
    if cfg.checkpoint_path is not None:
        checkpoint = torch.load(cfg.checkpoint_path)
        start_timestep = checkpoint['timestep']
        episode = checkpoint['episode']
        for i in range(num_robot_groups):
            optimizers[i].load_state_dict(checkpoint['optimizers'][i])
            optimizers_critic[i].load_state_dict(checkpoint['optimizers_critic'][i])
            replay_buffers[i] = checkpoint['replay_buffers'][i]
        if cfg.use_predicted_intention:
            for i in range(num_robot_groups):
                optimizers_intention[i].load_state_dict(checkpoint['optimizers_intention'][i])
        print("=> loaded checkpoint '{}' (timestep {})".format(cfg.checkpoint_path, start_timestep))

    # Logging
    train_summary_writer = SummaryWriter(log_dir=str(log_dir / 'train'))

    state = env.reset()
    transition_tracker = TransitionTracker(state)
    learning_starts = np.round(cfg.learning_starts_frac * cfg.total_timesteps).astype(np.uint32)
    total_timesteps_with_warm_up = learning_starts + cfg.total_timesteps
    for timestep in tqdm(range(start_timestep, total_timesteps_with_warm_up), initial=start_timestep,
                         total=total_timesteps_with_warm_up, file=sys.stdout):
        # Select an action for each robot

        cent_obs = env.get_state(all_robots=True)

        exploration_eps = 1 - (1 - cfg.final_exploration) * min(1, max(0, timestep - learning_starts) / (
                    cfg.exploration_frac * cfg.total_timesteps))
        if cfg.use_predicted_intention:
            use_ground_truth_intention = max(0,
                                             timestep - learning_starts) / cfg.total_timesteps <= cfg.use_predicted_intention_frac
            action, action_log_probs, values = policy.step(state, cent_obs, exploration_eps=exploration_eps,
                                                           use_ground_truth_intention=use_ground_truth_intention)
        else:
            action, action_log_probs, values = policy.step(state, cent_obs, exploration_eps=exploration_eps)
        transition_tracker.update_action(action, action_log_probs, values, cent_obs)

        # Step the simulation
        state, reward, done, info = env.step(action)

        # Store in buffers
        transitions_per_buffer = transition_tracker.update_step_completed(reward, state, done)
        for i, transitions in enumerate(transitions_per_buffer):
            for transition in transitions:
                replay_buffers[i][transition[0]].push(*transition)

        # Reset if episode ended
        if done:
            state = env.reset()
            transition_tracker = TransitionTracker(state)
            episode += 1

        # Train networks
        learning_starts = 49
        cfg.train_freq = 50
        if (timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0) or done:
            # next_value=computer(state, cent_obs, policy.critic_nets, policy.apply_transform)
            for i in range(num_robot_groups):
                for buffer in replay_buffers[i]:
                    buffer.compute_returns()
            all_train_info = {}
            for i in range(num_robot_groups):
                # batch = replay_buffers[i].sample(cfg.batch_size)
                batch = replay_buffers[i]
                policy.policy_nets[i].train()
                policy.critic_nets[i].train()
                for _ in range(all_args.ppo_epoch):
                    train_info = train(all_args, policy.policy_nets[i], optimizers[i], batch,
                                       policy.apply_transform,
                                       policy.critic_nets[i], optimizers_critic[i])

                if cfg.use_predicted_intention:
                    train_info_intention = train_intention(policy.intention_nets[i], optimizers_intention[i], batch,
                                                           policy.apply_transform)
                    train_info.update(train_info_intention)

                for name, val in train_info.items():
                    all_train_info['{}/robot_group_{:02}'.format(name, i + 1)] = val
        if done:
            train_summary_writer.add_scalar('steps', info['steps'], timestep + 1)
            train_summary_writer.add_scalar('total_cubes', info['total_cubes'], timestep + 1)
            train_summary_writer.add_scalar('episodes', episode, timestep + 1)

            for i in range(num_robot_groups):
                for name in ['cumulative_cubes', 'cumulative_distance', 'cumulative_reward',
                             'cumulative_robot_collisions']:
                    train_summary_writer.add_scalar('{}/robot_group_{:02}'.format(name, i + 1), np.mean(info[name][i]),
                                                    timestep + 1)
        ################################################################################
        # Checkpointing

        if (timestep + 1) % cfg.checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warm_up:
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save policy
            policy_filename = 'policy_{:08d}.pth.tar'.format(timestep + 1)
            policy_path = checkpoint_dir / policy_filename
            policy_checkpoint = {
                'timestep': timestep + 1,
                'state_dicts': [policy.policy_nets[i].state_dict() for i in range(num_robot_groups)],
            }
            if cfg.use_predicted_intention:
                policy_checkpoint['state_dicts_intention'] = [policy.intention_nets[i].state_dict() for i in
                                                              range(num_robot_groups)]
            torch.save(policy_checkpoint, str(policy_path))
            # Save critic
            critic_filename = 'critic_{:08d}.pth.tar'.format(timestep + 1)
            critic_path = checkpoint_dir / critic_filename
            critic_checkpoint = {
                'timestep': timestep + 1,
                'state_dicts': [policy.critic_nets[i].state_dict() for i in range(num_robot_groups)],
            }
            torch.save(critic_checkpoint, str(critic_path))
            # Save checkpoint
            checkpoint_filename = 'checkpoint_{:08d}.pth.tar'.format(timestep + 1)
            checkpoint_path = checkpoint_dir / checkpoint_filename
            checkpoint = {
                'timestep': timestep + 1,
                'episode': episode,
                'optimizers': [optimizers[i].state_dict() for i in range(num_robot_groups)],
                'optimizers_critic': [optimizers_critic[i].state_dict() for i in range(num_robot_groups)],
                'replay_buffers': [replay_buffers[i] for i in range(num_robot_groups)],
            }
            if cfg.use_predicted_intention:
                checkpoint['optimizers_intention'] = [optimizers_intention[i].state_dict() for i in
                                                      range(num_robot_groups)]
            torch.save(checkpoint, str(checkpoint_path))

            # Save updated config file
            cfg.policy_path = str(policy_path)
            cfg.critic_path = str(critic_path)
            cfg.checkpoint_path = str(checkpoint_path)
            utils.save_config(log_dir / 'config.yml', cfg)

            # Remove old checkpoint
            checkpoint_paths = list(checkpoint_dir.glob('checkpoint_*.pth.tar'))
            checkpoint_paths.remove(checkpoint_path)
            for old_checkpoint_path in checkpoint_paths:
                old_checkpoint_path.unlink()

        # 每次训练完都会将buffer的position指针以及内部经验池重置
        if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0 or done:
            for i in range(num_robot_groups):
                for buffer in replay_buffers[i]:
                    buffer.reset(done)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', default='config/experiments/ours/lifting_4-large_empty-ours.yml')
    config_path = parser.parse_args().config_path
    if config_path is None:
        if sys.platform == 'darwin':
            config_path = 'config/local/lifting_4-small_empty-local.yml'
        else:
            config_path = utils.select_run()
    if config_path is not None:
        config_path = utils.setup_run(config_path)
        main(utils.load_config(config_path))
