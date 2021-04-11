import ray
import torch

from copy import deepcopy
from itertools import count

import random

from ...env.env import BitcoinTradeEnv
from ..utils import inv_rescale, rescale

class LocalMemoryR2D2(object):
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.obs = []
        self.hidden_state_buffer = []
        self.dones = []

    def __len__(self):
        return len(self.dones)

    def reset(self, num_instance_to_keep=0):
        if num_instance_to_keep == 0:
            self.actions = []
            self.rewards = []
            self.obs = []
            self.hidden_state_buffer = []
            self.dones = []
        else:
            del self.actions[:-num_instance_to_keep]
            del self.rewards[:-num_instance_to_keep]
            del self.obs[:-num_instance_to_keep]
            del self.hidden_state_buffer[:-num_instance_to_keep]
            del self.dones[:-num_instance_to_keep]

@ray.remote(num_cpus=1)
class ActorR2D2():
    def __init__(self,
                 agent_core_net,
                 actor_id,
                 actor_total_num,
                 gamma,
                 nstep,
                 actor_update_frequency,
                 memory_size_bound,
                 device,
                 memory_server,
                 actor_epsilon,
                 actor_alpha,
                 parameter_server,
                 update_lambda,
                 sequence_length,
                 hidden_state_dim,
                 env_config,
                 trade_data_path):
        self.actor_net = deepcopy(agent_core_net).cpu()
        self.actor_net.load_state_dict({k: v.cpu() for k, v in agent_core_net.state_dict().items()})
        self.actor_net.eval()
        self.actor_id = actor_id
        self.actor_total_num = actor_total_num
        self.gamma = gamma
        self.nstep = nstep
        self.actor_update_frequency = actor_update_frequency
        self.memory_size_bound = memory_size_bound
        self.device = device
        self.memory_server = memory_server
        self.parameter_server = parameter_server
        self.update_lambda = update_lambda
        self.sequence_length = sequence_length
        self.hidden_state_dim = hidden_state_dim
        self.epsilon = actor_epsilon ** (1 + (actor_id * actor_alpha) / (self.actor_total_num - 1))
        self.env = BitcoinTradeEnv(trade_data_path, env_config)
        self.local_memory = LocalMemoryR2D2()
        self.episode_count = 0
        self.act_count = 0

    def epsilon_greedy_policy(self, state, hidden_state):
        with torch.no_grad():
            action_value, hidden_state_output = self.actor_net(state, hidden_state)

        if random.random() < self.epsilon:
            action = random.randint(0,2)
        else:
            action = action_value.argmax(dim=2).squeeze().item()

        return action, hidden_state_output

    def update_agent_from_learner(self):
        learner_state_dict = ray.get(self.parameter_server.send_latest_parameter_to_actor.remote())
        self.actor_net.load_state_dict({k: (1 - self.update_lambda)*v1 + self.update_lambda*v2 for k, v1, v2 in zip(learner_state_dict.keys(), self.actor_net.state_dict().values(), learner_state_dict.values())})

    def transit_to_nstep_return(self, start_index, end_index):
        sum_rewards = 0
        dones = False
        for i, index in enumerate(range(start_index, end_index + 1)):
            sum_rewards += (self.gamma**i) * self.local_memory.rewards[index]

            if self.local_memory.dones[index]:
                dones = True
                break

        return [sum_rewards, dones]

    def compute_local_priority(self, reward, dones, actions, sequential_state_input, sequential_initial_hidden_state):
        action_value, _ = self.actor_net(sequential_state_input, sequential_initial_hidden_state)
        action_value = action_value.squeeze()

        dones_ = dones[20+self.nstep+1:]
        reward_ = reward[20:-self.nstep-1]
        non_terminal_mask = 1 - dones_.float()
        terminal_mask = dones_.float()
        action_value_target = rescale((reward_ + (self.gamma ** (self.nstep + 1))*inv_rescale(action_value[20+self.nstep+1:, :].max(dim=1)[0])) * non_terminal_mask + reward_ * terminal_mask)

        td_error = (action_value_target - action_value[20+self.nstep+1:, :].gather(1, actions[20+self.nstep+1:].view(-1, 1)).view(-1)).abs() + 0.01
        priority = (0.9 * td_error.max() + 0.1 * td_error.mean()).view(1)

        return priority


    def run(self):
        obs = self.env.reset()
        hidden_state = (torch.zeros(1, 1, self.hidden_state_dim), torch.zeros(1, 1, self.hidden_state_dim))
        self.local_memory.hidden_state_buffer.append(hidden_state)
        step_count = 0

        for _ in count():
            with torch.no_grad():
                self.local_memory.obs.append(torch.from_numpy(obs).float())

                # sample action
                action, hidden_state = self.epsilon_greedy_policy(torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0), hidden_state)
                self.local_memory.actions.append(action)
                self.local_memory.hidden_state_buffer.append(hidden_state)

                obs, reward, done, _ = self.env.step(action)

                self.local_memory.rewards.append(reward)
                self.local_memory.dones.append(done)

                if done:
                    obs = self.env.reset()
                    self.episode_count += 1
                    step_count = 0

                    hidden_state = (torch.zeros(1, 1, self.hidden_state_dim), torch.zeros(1, 1, self.hidden_state_dim))
                    self.local_memory.hidden_state_buffer.append(hidden_state)
                    self.local_memory.reset(0)

                else:
                    if self.local_memory.__len__() == self.sequence_length + 20 + self.nstep + 1:
                        # transit to multi-step output
                        sequential_action = []
                        sequential_reward = []
                        sequential_state = []
                        sequential_dones = []
                        for start_index in [i for i in range(self.sequence_length + 20 + self.nstep + 1)]:
                            if  20 <= start_index < self.sequence_length + 20:
                                end_index = start_index + self.nstep
                                nstep_return_output = self.transit_to_nstep_return(start_index, end_index)
                                sequential_reward.append(nstep_return_output[0])
                                sequential_dones.append(nstep_return_output[1])
                            else:
                                sequential_reward.append(self.local_memory.rewards[start_index])
                                sequential_dones.append(self.local_memory.dones[start_index])

                            sequential_action.append(self.local_memory.actions[start_index])
                            sequential_state.append(self.local_memory.obs[start_index])

                        # compute priority
                        sequential_priority = self.compute_local_priority(
                            reward=torch.tensor(sequential_reward, dtype=torch.float),
                            dones=torch.tensor(sequential_dones, dtype=torch.bool),
                            actions=torch.tensor(sequential_action).long(),
                            sequential_state_input=torch.stack(sequential_state, dim=0).unsqueeze(0),
                            sequential_initial_hidden_state=self.local_memory.hidden_state_buffer[0]
                        )
                        self.memory_server.receive_sample_from_actor.remote([
                            [torch.tensor(sequential_action).long()],
                            [torch.tensor(sequential_reward, dtype=torch.float)],
                            [torch.stack(sequential_state, dim=0)],
                            [torch.tensor(sequential_dones, dtype=torch.bool)],
                            [self.local_memory.hidden_state_buffer[0]],
                            [sequential_priority]
                        ])

                        self.local_memory.reset(self.nstep+self.sequence_length)

                if self.act_count % self.actor_update_frequency == 0 and self.act_count > 0:
                    self.update_agent_from_learner()

                step_count += 1
                self.act_count += 1