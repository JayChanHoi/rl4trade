import torch
import torch.nn.functional as F

import ray
import numpy as np

from copy import deepcopy
import os
import random
from itertools import count

from ..utils import rescale, inv_rescale

@ray.remote(num_cpus=1, num_gpus=1)
class Learner(object):
    def __init__(self,
                 eval_env,
                 eval_frequency,
                 memory_server,
                 parameter_server,
                 device,
                 batch_size,
                 agent_core_net,
                 memory_size_bound,
                 optimizer,
                 gamma,
                 nstep,
                 update_lambda,
                 target_net_update_frequency,
                 priority_alpha=0.6,
                 priority_beta=0.4):
        self.memory_server = memory_server
        self.parameter_server = parameter_server
        self.agent_core_net = agent_core_net
        self.target_net = deepcopy(self.agent_core_net)
        self.target_net.load_state_dict(self.agent_core_net.state_dict())
        self.priority_alpha = priority_alpha
        self.batch_size = batch_size
        self.device = device
        self.train_count = 0
        self.priority_beta = priority_beta
        self.memory_size_bound = memory_size_bound
        self.target_net_update_frequency = target_net_update_frequency
        self.gamma = gamma
        self.nsteps = nstep
        self.update_lambda = update_lambda
        self.optimizer = optimizer
        self.eval_frequency = eval_frequency
        self.eval_env = eval_env
        if torch.cuda.is_available():
            self.agent_core_net.cuda()
            self.target_net.cuda()

        self.agent_core_net.train()
        self.target_net.eval()

    def eval(self, qnet):
        obs = self.eval_env.reset()

        reward_list = []
        episode_length = 0

        for iter in count():
            state = torch.cat(
                [
                    torch.from_numpy(obs).float(),
                    torch.from_numpy(self.eval_env.get_action_mask()).float().reshape(1, -1).repeat(obs.shape[0], 1)
                ],
                dim=1
            )

            if qnet is not None:
                action_value = qnet(state.unsqueeze(0).cuda())
                action = action_value.argmax(dim=2).squeeze().item()
            else:
                # sample action
                action = torch.from_numpy(self.eval_env.get_action_mask()).float().multinomial(1).item()

            obs, reward, done, _ = self.eval_env.step(action)
            reward_list.append(reward)
            episode_length += 1

            if done:
                episode_reward = np.sum(reward_list) / (episode_length)
                torch.cuda.empty_cache()
                return [episode_reward, (self.eval_env.total_capital_history[-1] - self.eval_env.total_capital_history[0])/ 10000]

    def run(self):
        learner_expected_reward = None
        learner_episodic_investment_return = None
        random_agent_expected_reward = None
        random_agent_episodic_investment_return = None

        # if ray.get(memory_size) >= self.learner_start_update_memory_size:
        batch_memory, sample_indices, priority_prob = ray.get(self.memory_server.send_sample_to_learner.remote(
            alpha=self.priority_alpha,
            batch_size=self.batch_size
        ))
        # eg job_obs batch item has shape -> (b, sequence_length, hist_length, job_num, job_feature_dim)
        batch_memory = [item.to(self.device) for item in batch_memory]
        # priority_alpha = torch.tensor(priority, dtype=torch.float32).to(self.device)**self.priority_alpha
        # priority_prob = (priority_alpha / priority_alpha.sum())

        is_weight = ((priority_prob.to(self.device) * self.memory_size_bound) ** self.priority_beta).reciprocal_()
        normalized_is_weight = is_weight / is_weight.max()

        action_value = self.agent_core_net(batch_memory[2])

        with torch.no_grad():
            next_state_action_value = self.target_net(batch_memory[3])
            learner_next_state_max_action_value = self.agent_core_net(batch_memory[3])
            learner_next_state_max_action = learner_next_state_max_action_value.argmax(dim=1, keepdim=True)

        rewards_ = batch_memory[1]
        dones_ = batch_memory[-1]
        non_terminal_mask = 1 - dones_.float()
        terminal_mask = dones_.float()
        action_value_target = rescale((rewards_ + (self.gamma ** (self.nsteps + 1))*(inv_rescale(next_state_action_value.gather(1, learner_next_state_max_action)).squeeze())) * non_terminal_mask + rewards_ * terminal_mask)

        td_error = (action_value_target - action_value.gather(1, batch_memory[0].view(-1, 1)).view(-1))
        with torch.no_grad():
            self.memory_server.update_priority.remote((td_error.abs() + 0.01).cpu(), sample_indices)

        loss = (normalized_is_weight * (td_error**2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_core_net.parameters(), 1)
        self.optimizer.step()

        if self.train_count > 0 and self.train_count % 10:
            self.parameter_server.update_ps_state_dict.remote({k:v.cpu() for k, v in self.agent_core_net.state_dict().items()})

        if self.train_count % self.target_net_update_frequency == 0 and self.train_count > 0:
            self.target_net.load_state_dict({k: (1 - self.update_lambda)*v1 + self.update_lambda*v2 for k, v1, v2 in zip(self.agent_core_net.state_dict().keys(), self.target_net.state_dict().values(), self.agent_core_net.state_dict().values())})

        if self.train_count % self.eval_frequency == 0 and self.train_count > 0:
            with torch.no_grad():
                self.agent_core_net.eval()
                learner_expected_reward, learner_episodic_investment_return = self.eval(self.agent_core_net)
                random_agent_expected_reward, random_agent_episodic_investment_return = self.eval(None)

        self.agent_core_net.train()
        self.train_count += 1

        return loss.item(), learner_expected_reward, learner_episodic_investment_return, random_agent_expected_reward, \
               random_agent_episodic_investment_return, self.train_count

@ray.remote(num_cpus=2, num_gpus=1)
class LearnerR2D2(object):
    def __init__(self,
                 memory_server,
                 parameter_server,
                 device,
                 batch_size,
                 agent_core_net,
                 memory_size_bound,
                 optimizer,
                 gamma,
                 nstep,
                 update_lambda,
                 target_net_update_frequency,
                 model_name,
                 num_layer,
                 priority_alpha=0.6,
                 priority_beta=0.4,
                 hidden_state_dim=512,
                 sequence_length=30,
                 burn_in_length=1,
                 gradient_norm_clip=1.0):
        self.memory_server = memory_server
        self.parameter_server = parameter_server
        self.agent_core_net = agent_core_net
        self.target_net = deepcopy(self.agent_core_net)
        self.target_net.load_state_dict(self.agent_core_net.state_dict())
        self.priority_alpha = priority_alpha
        self.batch_size = batch_size
        self.device = device
        self.train_count = 0
        self.priority_beta = priority_beta
        self.memory_size_bound = memory_size_bound
        self.target_net_update_frequency = target_net_update_frequency
        self.gamma = gamma
        self.nsteps = nstep
        self.update_lambda = update_lambda
        self.optimizer = optimizer
        self.model_name = model_name
        self.hidden_state_dim = hidden_state_dim
        self.num_layer = num_layer
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.gradient_norm_clip = gradient_norm_clip
        if torch.cuda.is_available():
            self.agent_core_net.cuda()
            self.target_net.cuda()

        self.agent_core_net.train()
        self.target_net.eval()

    def run(self):
        batch_memory, sample_indices, priority_prob = ray.get(self.memory_server.send_sample_to_learner.remote(
            alpha=self.priority_alpha,
            batch_size=self.batch_size
        ))
        # eg job_obs batch item has shape -> (b, sequence_length, hist_length, job_num, job_feature_dim)
        batch_memory = [item.to(self.device) for item in batch_memory]
        priority_prob = priority_prob.to(self.device)
        priority_beta = self.priority_beta

        is_weight = ((priority_prob * self.memory_size_bound) ** priority_beta).reciprocal_()
        normalized_is_weight = is_weight / is_weight.max()

        update_state = batch_memory[2][:, self.burn_in_length:-(self.nsteps+1), :, :]
        core_net_hns = batch_memory[4][self.burn_in_length:-(self.nsteps+1), :, :, :]
        core_net_cns = batch_memory[5][self.burn_in_length:-(self.nsteps+1), :, :, :]
        action_value, final_hidden_state = self.agent_core_net(update_state, (core_net_hns, core_net_cns))

        with torch.no_grad():
            traget_net_action_value, _ = self.target_net(
                batch_memory[2][:, :, :, :],
                (batch_memory[4], batch_memory[5])
            )
            self.agent_core_net.eval()
            action_value_extra, _ = self.agent_core_net(
                batch_memory[2][:, -(self.nsteps+1):, :, :],
                (batch_memory[4][-(self.nsteps+1):, :, :, :], batch_memory[5][-(self.nsteps+1):, :, :, :])
            )

            learner_action_value_for_max_action = torch.cat([action_value, action_value_extra], dim=1)[:, self.nsteps+1:, :]
            learner_max_action = learner_action_value_for_max_action.argmax(dim=2, keepdim=True)

            self.agent_core_net.train()

        rewards_ = batch_memory[1][:, self.burn_in_length:-(self.nsteps+1)]
        dones_ = batch_memory[-3][:, self.burn_in_length+(self.nsteps+1):]
        # dones_ -> shape :(b. sequence_length)
        non_terminal_mask = 1 - dones_.float()
        terminal_mask = dones_.float()
        action_value_target = rescale((rewards_ + (self.gamma ** (self.nsteps + 1))*(inv_rescale(traget_net_action_value.gather(2, learner_max_action)).squeeze())) * non_terminal_mask + rewards_ * terminal_mask)
        td_error = (action_value_target - action_value.gather(2, batch_memory[0][:, self.burn_in_length:-(self.nsteps+1)].unsqueeze(-1)).squeeze())

        with torch.no_grad():
            priority = (0.9 * td_error.abs().max(dim=1)[0] + (1 - 0.9) * td_error.abs().mean(dim=1)).cpu()
            self.memory_server.update_priority.remote(priority, sample_indices)
            self.memory_server.trim_global_memory.remote()

        self.optimizer.zero_grad()
        loss = (normalized_is_weight * (td_error**2).mean(dim=1)).mean()
        l = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_core_net.parameters(), self.gradient_norm_clip)
        self.optimizer.step()

        if self.train_count > 0 and self.train_count % 10:
            self.parameter_server.update_ps_state_dict.remote({k:v.cpu() for k, v in self.agent_core_net.state_dict().items()})

        if self.train_count % self.target_net_update_frequency == 0 and self.train_count > 0:
            self.target_net.load_state_dict({k: (1 - self.update_lambda)*v1 + self.update_lambda*v2 for k, v1, v2 in zip(self.agent_core_net.state_dict().keys(), self.target_net.state_dict().values(), self.agent_core_net.state_dict().values())})

        self.agent_core_net.train()
        self.train_count += 1

        return l, self.train_count