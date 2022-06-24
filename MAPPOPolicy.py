import copy

import torch
# from mappo_actor_critic import R_Actor, R_Critic
import random
import numpy as np
import torch
from torchvision import transforms
import networks
from envs import VectorEnv

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
def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class MAPPOPolicy:
    def __init__(self, cfg, train=False, random_seed=None):
        self.cfg = cfg
        self.robot_group_types = [next(iter(g.keys())) for g in self.cfg.robot_config]
        self.train = train
        if random_seed is not None:
            random.seed(random_seed)

        self.num_robot_groups = len(self.robot_group_types)
        self.transform = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_nets = self.build_policy_nets()

        self.critic_nets = self.build_critic_nets()
        # Resume if applicable
        if self.cfg.checkpoint_path is not None:
            self.policy_checkpoint = torch.load(self.cfg.policy_path, map_location=self.device)
            for i in range(self.num_robot_groups):
                self.policy_nets[i].load_state_dict(self.policy_checkpoint['state_dicts'][i])
                if self.train:
                    self.policy_nets[i].train()
                else:
                    self.policy_nets[i].eval()
            print("=> loaded policy '{}'".format(self.cfg.policy_path))

    def build_policy_nets(self):
        policy_nets = []
        for robot_type in self.robot_group_types:
            num_output_channels = VectorEnv.get_num_output_channels(robot_type)
            policy_nets.append(torch.nn.DataParallel(
                networks.FCN(num_input_channels=self.cfg.num_input_channels, num_output_channels=num_output_channels)
            ).to(self.device))
        return policy_nets

    def build_critic_nets(self):
        policy_nets = []
        for robot_type in self.robot_group_types:
            num_output_channels = VectorEnv.get_num_output_channels(robot_type)
            policy_nets.append(torch.nn.DataParallel(
                networks.critic_FCN(num_input_channels=self.cfg.num_input_channels, num_output_channels=num_output_channels)
            ).to(self.device))
        return policy_nets
    def apply_transform(self, s):
        return self.transform(s).unsqueeze(0)

    def step(self, state, cent_obs, exploration_eps=None, debug=False):
        if exploration_eps is None:
            exploration_eps = self.cfg.final_exploration

        action = [[None for _ in g] for g in state]
        output = [[None for _ in g] for g in state]
        action_log_probs=[[None for _ in g] for g in state]
        values=[[None for _ in g] for g in state]
        with torch.no_grad():
            for i, g in enumerate(state):
                robot_type = self.robot_group_types[i]
                self.policy_nets[i].eval()
                for j, s in enumerate(g):
                    if s is not None:
                        s = self.apply_transform(s).to(self.device)
                        o = self.policy_nets[i](s).squeeze(0)
                        # action_logits=copy.deepcopy(o).view(1,-1)
                        action_logits=FixedCategorical(logits=o.view(1,-1))
                        if random.random() < exploration_eps:
                            a = random.randrange(VectorEnv.get_action_space(robot_type))
                        else:
                            # a=torch.multinomial(action_logits,1)
                            a=action_logits.sample().item()
                        # action_log_probs[i][j]=action_logits[0,a]
                        action_log_probs[i][j] = action_logits.log_probs(torch.LongTensor([[a]]).to(self.device))
                        action[i][j] = a
                        output[i][j] = o.cpu().numpy()
                        cent_obs[i]=torch.cat([self.apply_transform(c) for c in cent_obs[i]]).to(self.device)
                        values[i][j] =self.critic_nets[i](cent_obs[i])

                if self.train:
                    self.policy_nets[i].train()

        if debug:
            info = {'output': output}
            return action, info

        return action,action_log_probs,values
class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
