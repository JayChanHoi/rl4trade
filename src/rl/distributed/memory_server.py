import torch

import ray

import random

from .sum_tree import SumTree

class GlobalMemory(object):
    def __init__(self, memory_size_bound):
        # self.memory_tree = SumTree(memory_size_bound)
        # self.memory_size_bound = memory_size_bound
        self.memory_size_bound = memory_size_bound

        self.actions = []
        self.rewards = []
        self.states = []
        self.next_states = []
        self.dones = []
        self.priority = []

    def __len__(self):
        # return self.memory_tree.n_entries
        return self.dones.__len__()

    def sample(self, alpha, batch_size):
        priority = torch.cat(self.priority, dim=0)
        priority_alpha = priority**alpha
        priority_prob = priority_alpha / priority_alpha.sum()
        sample_indices = priority_prob.multinomial(batch_size).tolist()

        actions = []
        rewards = []
        states = []
        next_states = []
        dones = []

        # idxs = []
        # segment = self.memory_tree.total() / batch_size
        # priorities = []

        for index in sample_indices:
            # a = segment * i
            # b = segment * (i + 1)
            # s = random.uniform(a, b)
            # idx, p, data = self.memory_tree.get(s)
            #
            # state, action, reward, next_state, done = data

            states.append(self.states[index])
            actions.append(self.actions[index])
            rewards.append(self.rewards[index])
            next_states.append(self.next_states[index])
            dones.append(self.dones[index])

        batch_memory = [
            torch.tensor(actions, dtype=torch.long).detach(),
            torch.tensor(rewards, dtype=torch.float).detach(),
            torch.stack(states, dim=0).detach(),
            torch.stack(next_states, dim=0).detach(),
            torch.tensor(dones, dtype=torch.bool).detach()
        ]

        return batch_memory, sample_indices, priority_prob

    def add_to_global_memory(self, sample):
        self.actions += sample[0]
        self.rewards += sample[1]
        self.states += sample[2]
        self.next_states += sample[3]
        self.dones += sample[4]
        self.priority += sample[5]

        # for i in range(len(dones)):
        #     data = (states[i], actions[i], rewards[i], next_states[i], dones[i])
        #     p = priorities[i].numpy()
        #     self.memory_tree.add(p, data)

    def trim_global_memory(self):
        excessive_size = self.__len__() - self.memory_size_bound
        if excessive_size > 0:
            del self.actions[:excessive_size]
            del self.rewards[:excessive_size]
            del self.states[:excessive_size]
            del self.dones[:excessive_size]
            del self.next_states[:excessive_size]
            del self.priority[:excessive_size]

    def update_priority(self, priority, sample_indices):
        p = priority.cpu().numpy()
        for i, index in enumerate(sample_indices):
            self.priority[index] = priority[i].view(1)

class GlobalMemoryR2D2(object):
    def __init__(self, memory_size_bound):
        self.memory_size_bound = memory_size_bound

        self.actions = []
        self.rewards = []
        self.states = []
        self.initial_hidden_state = []
        self.dones = []
        self.priority = []

    def __len__(self):
        return len(self.dones)

    def sample(self, alpha, batch_size):
        priority = torch.cat(self.priority, dim=0)
        priority_alpha = priority**alpha
        priority_prob = priority_alpha / priority_alpha.sum()
        sample_indices = priority_prob.multinomial(batch_size).tolist()

        batch_actions = []
        batch_rewards = []
        batch_state = []
        batch_dones = []
        batch_initial_hidden_state_h = []
        batch_initial_hidden_state_c = []
        for index in sample_indices:
            batch_actions.append(self.actions[index])
            batch_rewards.append(self.rewards[index])
            batch_state.append(self.states[index])
            batch_dones.append(self.dones[index])
            batch_initial_hidden_state_h.append(self.initial_hidden_state[index][0])
            batch_initial_hidden_state_c.append(self.initial_hidden_state[index][1])

        batch_memory = [
            torch.stack(batch_actions, dim=0).detach(),
            torch.stack(batch_rewards, dim=0).detach(),
            torch.stack(batch_state, dim=0).detach(),
            torch.stack(batch_dones, dim=0).detach(),
            torch.cat(batch_initial_hidden_state_h, dim=1).detach(),
            torch.cat(batch_initial_hidden_state_c, dim=1).detach()
        ]

        return [batch_memory, sample_indices, priority_prob[sample_indices]]

    def add_to_global_memory(self, sample):
        self.actions += sample[0]
        self.rewards += sample[1]
        self.states += sample[2]
        self.dones += sample[3]
        self.initial_hidden_state += sample[4]
        self.priority += sample[5]

    def trim_global_memory(self):
        excessive_size = self.__len__() - self.memory_size_bound
        if excessive_size > 0:
            del self.actions[:excessive_size]
            del self.rewards[:excessive_size]
            del self.states[:excessive_size]
            del self.dones[:excessive_size]
            del self.initial_hidden_state[:excessive_size]
            del self.priority[:excessive_size]

    def update_priority(self, priority, sample_indices):
        for i, index in enumerate(sample_indices):
            self.priority[index] = priority[i].view(1)

@ray.remote(num_cpus=3)
class MemoryServer(object):
    def __init__(self, memory_size_bound):
        self.global_memory = GlobalMemory(memory_size_bound)

    def memory_size(self):
        return self.global_memory.__len__()

    def send_sample_to_learner(self, alpha, batch_size):
        return self.global_memory.sample(alpha, batch_size)

    def update_priority(self, priority, sample_indices):
        self.global_memory.update_priority(priority, sample_indices)

    def receive_sample_from_actor(self, sample_from_actor):
        self.global_memory.add_to_global_memory(sample_from_actor)