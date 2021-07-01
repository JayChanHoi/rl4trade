import torch

import ray
import numpy as np

from itertools import count
import os

from ..utils import rescale, inv_rescale

@ray.remote(num_cpus=1, num_gpus=1)
class EvaluatorR2D2(object):
    def __init__(self,
                 num_layer,
                 parameters_server,
                 device,
                 agent_core_net,
                 env,
                 hidden_state_dim):
        self.num_layer = num_layer
        self.parameters_server = parameters_server
        self.device = device
        self.hidden_state_dim = hidden_state_dim
        self.evaluation_agent = agent_core_net
        self.evaluation_agent.eval()
        self.eval_env = env

    def eval(self, qnet):
        obs = self.eval_env.reset()

        reward_list = []
        episode_length = 0
        if qnet is not None:
            hn, cn = (torch.zeros(self.num_layer, 1, self.hidden_state_dim, device=self.device), torch.zeros(self.num_layer, 1, self.hidden_state_dim, device=self.device))

        for iter in count():
            state = torch.cat(
                [
                    torch.from_numpy(obs).float(),
                    torch.from_numpy(self.eval_env.get_action_mask()).float().reshape(1, -1).repeat(obs.shape[0], 1)
                ],
                dim=1
            )

            if qnet is not None:
                action_value, hidden_state = qnet(state.unsqueeze(0).unsqueeze(0).cuda(), (hn.unsqueeze(0), cn.unsqueeze(0)))
                action = action_value.argmax(dim=2).squeeze().item()
            else:
                # sample action
                action = torch.from_numpy(self.eval_env.get_action_mask()).float().multinomial(1).item()
                # action = random.randint(0,2)

            obs, reward, done, _ = self.eval_env.step(action)
            reward_list.append(reward)
            episode_length += 1

            if done:
                episode_reward = np.sum(reward_list) / (episode_length)
                torch.cuda.empty_cache()
                return [episode_reward, (self.eval_env.total_capital_history[-1] - self.eval_env.total_capital_history[0])/ 10000]

    def update_evaluator_agent_from_parameters_server(self):
        parameters_state_dict = ray.get(self.parameters_server.send_latest_parameter_to_actor.remote())
        self.evaluation_agent.load_state_dict(parameters_state_dict)

    def run(self):
        self.update_evaluator_agent_from_parameters_server()
        with torch.no_grad():
            learner_expected_reward, learner_episodic_investment_return = self.eval(self.evaluation_agent)
            random_agent_expected_reward, random_agent_episodic_investment_return = self.eval(None)

        return learner_expected_reward, \
               learner_episodic_investment_return, \
               random_agent_expected_reward, \
               random_agent_episodic_investment_return