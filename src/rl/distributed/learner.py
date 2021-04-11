import torch
import torch.nn.functional as F

import ray
import numpy as np

from copy import deepcopy
from itertools import count
import os
import random

from ..utils import rescale, inv_rescale

class LearnerR2D2(object):
    def __init__(self,
                 eval_env,
                 eval_frequency,
                 memory_server,
                 parameter_server,
                 writer,
                 device,
                 batch_size,
                 agent_core_net,
                 memory_size_bound,
                 optimizer,
                 gamma,
                 nstep,
                 update_lambda,
                 target_net_update_frequency,
                 learner_start_update_memory_size,
                 model_name,
                 priority_alpha=0.6,
                 priority_beta=0.4,
                 hidden_state_dim=512,
                 sequence_length=30):
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
        self.learner_start_update_memory_size = learner_start_update_memory_size
        self.gamma = gamma
        self.nsteps = nstep
        self.update_lambda = update_lambda
        self.optimizer = optimizer
        self.writer = writer
        self.model_name = model_name
        self.eval_frequency = eval_frequency
        self.eval_env = eval_env
        self.hidden_state_dim = hidden_state_dim
        self.sequence_length = sequence_length

    def write_log(self, writer, value, tag, global_step):
        writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)

    def eval(self, qnet):
        state = self.eval_env.reset()

        reward_list = []
        episode_length = 0
        if qnet is not None:
            hidden_state = (torch.zeros(1, 1, self.hidden_state_dim, device=self.device), torch.zeros(1, 1, self.hidden_state_dim, device=self.device))

        for iter in count():
            if qnet is not None:
                action_value, hidden_state = qnet(state.unsqueeze(0).unsqueeze(0), hidden_state)
                action = action_value.argmax(dim=2).squeeze()
            else:
                # sample action
                action = random.randint(0, 3)

            state, reward, done, _ = self.eval_env.step(action)
            reward_list.append(reward)
            episode_length += 1

            if done:
                episode_reward = np.sum(reward_list) / (episode_length)
                torch.cuda.empty_cache()
                return [episode_reward, (self.eval_env.total_capital_history[-1] - self.eval_env.total_capital_history[0])/ 10000]

    def run(self):
        self.agent_core_net.train()
        self.target_net.eval()

        for iter in count():
            memory_size = self.memory_server.memory_size.remote()
            if ray.get(memory_size) >= self.learner_start_update_memory_size:
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

                with torch.no_grad():
                    burn_in_state = [
                        batch_memory[2][:, :20, :, :, :],
                        batch_memory[3][:, :20, :, :, :],
                        batch_memory[4][:, :20, :, :, :],
                        batch_memory[5][:, :20, :, :]
                    ]
                    _, core_net_burn_in_final_hidden_state = self.agent_core_net(
                        burn_in_state,
                        (batch_memory[7], batch_memory[8])
                    )

                update_state = [
                    batch_memory[2][:, 20:-(self.nsteps+1), :, :, :],
                    batch_memory[3][:, 20:-(self.nsteps+1), :, :, :],
                    batch_memory[4][:, 20:-(self.nsteps+1), :, :, :],
                    batch_memory[5][:, 20:-(self.nsteps+1), :, :]
                ]
                action_value, final_hidden_state = self.agent_core_net(update_state, core_net_burn_in_final_hidden_state)

                with torch.no_grad():
                    traget_net_action_value, _ = self.target_net(
                        [
                            batch_memory[2][:, 20+self.nsteps+1:, :, :, :],
                            batch_memory[3][:, 20+self.nsteps+1:, :, :, :],
                            batch_memory[4][:, 20+self.nsteps+1:, :, :, :],
                            batch_memory[5][:, 20+self.nsteps+1:, :, :]
                        ],
                        (batch_memory[7], batch_memory[8])
                    )
                    action_value_extra, _ = self.agent_core_net(
                        [
                            batch_memory[2][:, -(self.nsteps+1):, :, :, :],
                            batch_memory[3][:, -(self.nsteps+1):, :, :, :],
                            batch_memory[4][:, -(self.nsteps+1):, :, :, :],
                            batch_memory[5][:, -(self.nsteps+1):, :, :]
                        ],
                        final_hidden_state
                    )

                    learner_action_value_for_max_action = torch.cat([action_value, action_value_extra], dim=1)[:, self.nsteps+1:, :]
                    learner_max_action = learner_action_value_for_max_action.argmax(dim=2, keepdim=True)

                rewards_ = batch_memory[1][:, 20:-(self.nsteps+1)]
                dones_ = batch_memory[-3][:, 20+(self.nsteps+1):]
                # dones_ -> shape :(b. sequence_length)
                non_terminal_mask = 1 - dones_.float()
                terminal_mask = dones_.float()
                action_value_target = rescale((rewards_ + (self.gamma ** (self.nsteps + 1))*(inv_rescale(traget_net_action_value.gather(2, learner_max_action)).squeeze())) * non_terminal_mask + rewards_ * terminal_mask)
                td_error = (action_value_target - action_value.gather(2, batch_memory[0][:, 20:-(self.nsteps+1)].unsqueeze(-1)).squeeze())

                with torch.no_grad():
                    priority = (0.9 * td_error.abs().max(dim=1)[0] + (1 - 0.9) * td_error.abs().mean(dim=1)).cpu()
                    self.memory_server.update_priority.remote(priority, sample_indices)
                    self.memory_server.trim_excessive_sample.remote()

                loss = (normalized_is_weight * (td_error**2).mean(dim=1)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.write_log(self.writer, loss.item(), 'Loss', self.train_count)

                if self.train_count > 0 and self.train_count % 10:
                    self.parameter_server.update_ps_state_dict.remote({k:v.cpu() for k, v in self.agent_core_net.state_dict().items()})

                if self.train_count % self.target_net_update_frequency == 0 and self.train_count > 0:
                    self.target_net.load_state_dict({k: (1 - self.update_lambda)*v1 + self.update_lambda*v2 for k, v1, v2 in zip(self.agent_core_net.state_dict().keys(), self.target_net.state_dict().values(), self.agent_core_net.state_dict().values())})

                if self.train_count % self.eval_frequency == 0 and self.train_count > 0:
                    with torch.no_grad():
                        self.agent_core_net.eval()
                        expected_reward, episodic_investment_return = self.eval(self.agent_core_net)
                        self.writer.add_scalars(main_tag='expected reward', tag_scalar_dict={'learner':expected_reward}, global_step=self.train_count)
                        self.writer.add_scalars(main_tag='episodic investment return ', tag_scalar_dict={'learner':episodic_investment_return}, global_step=self.train_count)
                        self.writer.add_scalars(main_tag='memory size', tag_scalar_dict={'learner':ray.get(memory_size)}, global_step=iter)

                        expected_reward, episodic_investment_return = self.eval(None)
                        self.writer.add_scalars(main_tag='expected reward', tag_scalar_dict={'random agent':expected_reward}, global_step=self.train_count)
                        self.writer.add_scalars(main_tag='episodic investment return ', tag_scalar_dict={'random agent':episodic_investment_return}, global_step=self.train_count)

                self.agent_core_net.train()
                if self.train_count % 500 == 0 and self.train_count > 0:
                    if torch.cuda.device_count() > 1:
                        checkpoint_dict = {'iter': iter + 1, 'state_dict': self.agent_core_net.module.state_dict()}
                        torch.save(
                            checkpoint_dict,
                            os.path.join(
                                'checkpoint/{}'.format(self.model_name),
                                '{}_checkpoint_episode_{}.pth'.format('distributed_dqn', self.train_count + 1)
                            )
                        )
                    else:
                        checkpoint_dict = {'iter': iter + 1, 'state_dict': self.agent_core_net.state_dict()}
                        torch.save(
                            checkpoint_dict,
                            os.path.join(
                                'checkpoint/{}'.format(self.model_name),
                                '{}_checkpoint_episode_{}.pth'.format('distributed_dqn', self.train_count + 1)
                            )
                        )

                self.train_count += 1

                print('===============================train count : {}================================'.format(self.train_count))
                print(loss.item())

            else:
                if iter % 10000 == 0:
                    self.writer.add_scalars(main_tag='memory size', tag_scalar_dict={'learner':ray.get(memory_size)}, global_step=iter)