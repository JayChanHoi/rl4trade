import torch

import ray

from tensorboardX import SummaryWriter

from .rl.distributed.actor import ActorR2D2, Actor
from .rl.distributed.learner import LearnerR2D2, Learner
from .rl.distributed.parameter_server import ParameterServer
from .rl.distributed.memory_server import MemoryServer
from .rl.utils import write_config, resume
from .agc_utils.agc import AGC
from .rl.model.qnet import LSTMQNet, QNet
from .env.env import BitcoinTradeEnv

import os
from collections import namedtuple
import yaml
import shutil
from itertools import count

def distributed_train(train_config):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    data_path = os.path.join(os.getcwd(), 'data/Binance_BTCUSDT_minute.csv')
    env_config_path = os.path.join(os.getcwd(), 'src/config/env_config.yml')
    env_config_dict = yaml.load(open(env_config_path, 'r'))
    env_config = namedtuple('env_config', env_config_dict.keys())(**env_config_dict)
    eval_env = BitcoinTradeEnv(data_path, env_config)
    eval_env.reset()

    if os.path.isdir('tensorboard/{}'.format(train_config.model_name)):
        shutil.rmtree('tensorboard/{}'.format(train_config.model_name))
        os.makedirs('tensorboard/{}'.format(train_config.model_name))
    else:
        os.makedirs('tensorboard/{}'.format(train_config.model_name))

    if not os.path.isdir('checkpoint/{}'.format(train_config.model_name)):
        os.makedirs('checkpoint/{}'.format(train_config.model_name))

    writer = SummaryWriter('tensorboard/{}'.format(train_config.model_name))
    write_config(writer, train_config, eval_env.env_config)
    # agent_core_net = LSTMQNet(
    #     train_config.dropout_p,
    #     hist_length=4,
    #     num_layer=train_config.num_layer
    # )

    agent_core_net = QNet(train_config.dropout_p)

    if train_config.resume != '':
        resume(agent_core_net, device, train_config.resume)

    optimizer = torch.optim.Adam(agent_core_net.parameters(), train_config.lr, eps=1.5e-4)
    optimizer = AGC(optim=optimizer, clipping=train_config.agc_clipping)

    parameter_server = ParameterServer.remote({k: v.cpu() for k, v in agent_core_net.state_dict().items()})
    memory_server = MemoryServer.remote(train_config.memory_size_bound)

    actors = []
    for i in range(train_config.actor_total_num):
        # actor = ActorR2D2.remote(agent_core_net=agent_core_net,
        #                          memory_server=memory_server,
        #                          parameter_server=parameter_server,
        #                          actor_id=i,
        #                          actor_total_num=train_config.actor_total_num,
        #                          memory_size_bound=train_config.memory_size_bound,
        #                          gamma=train_config.gamma,
        #                          update_lambda=train_config.update_lambda,
        #                          actor_update_frequency=train_config.actor_update_frequency,
        #                          actor_epsilon=train_config.actor_epsilon,
        #                          actor_alpha=train_config.actor_alpha,
        #                          device='cpu',
        #                          nstep=train_config.nstep,
        #                          env_config=env_config,
        #                          trade_data_path=data_path,
        #                          sequence_length=train_config.sequence_length,
        #                          hidden_state_dim=train_config.hidden_state_dim,
        #                          num_layer=train_config.num_layer)
        actor = Actor.remote(agent_core_net=agent_core_net,
                             memory_server=memory_server,
                             parameter_server=parameter_server,
                             actor_id=i,
                             actor_total_num=train_config.actor_total_num,
                             memory_size_bound=train_config.memory_size_bound,
                             gamma=train_config.gamma,
                             update_lambda=train_config.update_lambda,
                             actor_update_frequency=train_config.actor_update_frequency,
                             actor_epsilon=train_config.actor_epsilon,
                             actor_alpha=train_config.actor_alpha,
                             device='cpu',
                             nstep=train_config.nstep,
                             env_config=env_config,
                             trade_data_path=data_path)
        actors.append(actor)

    # learner = LearnerR2D2.remote(eval_env=eval_env,
    #                              eval_frequency=train_config.eval_frequency,
    #                              memory_server=memory_server,
    #                              parameter_server=parameter_server,
    #                              device=device,
    #                              batch_size=train_config.batch_size,
    #                              agent_core_net=agent_core_net,
    #                              memory_size_bound=train_config.memory_size_bound,
    #                              optimizer=optimizer,
    #                              gamma=train_config.gamma,
    #                              update_lambda=train_config.update_lambda,
    #                              target_net_update_frequency=train_config.target_net_update_frequency,
    #                              model_name=train_config.model_name,
    #                              priority_alpha=train_config.priority_alpha,
    #                              priority_beta=train_config.priority_beta,
    #                              nstep=train_config.nstep,
    #                              hidden_state_dim=train_config.hidden_state_dim,
    #                              sequence_length=train_config.sequence_length,
    #                              num_layer=train_config.num_layer)

    learner = Learner.remote(eval_env=eval_env,
                             eval_frequency=train_config.eval_frequency,
                             memory_server=memory_server,
                             parameter_server=parameter_server,
                             device=device,
                             batch_size=train_config.batch_size,
                             agent_core_net=agent_core_net,
                             memory_size_bound=train_config.memory_size_bound,
                             optimizer=optimizer,
                             gamma=train_config.gamma,
                             update_lambda=train_config.update_lambda,
                             target_net_update_frequency=train_config.target_net_update_frequency,
                             priority_alpha=train_config.priority_alpha,
                             priority_beta=train_config.priority_beta,
                             nstep=train_config.nstep)

    for actor in actors:
        actor.run.remote()

    for _ in count():
        memory_size = ray.get(memory_server.memory_size.remote())
        if memory_size >= train_config.learner_start_update_memory_size:
            loss, learner_expected_reward, learner_episodic_investment_return, random_agent_expected_reward, \
            random_agent_episodic_investment_return, train_count = ray.get(learner.run.remote())
            if learner_expected_reward is not None:
                writer.add_scalars(main_tag='expected reward', tag_scalar_dict={'learner':learner_expected_reward}, global_step=train_count)
            if learner_episodic_investment_return is not None:
                writer.add_scalars(main_tag='episodic investment return ', tag_scalar_dict={'learner':learner_episodic_investment_return}, global_step=train_count)
            if random_agent_expected_reward is not None:
                writer.add_scalars(main_tag='expected reward', tag_scalar_dict={'random agent':random_agent_expected_reward}, global_step=train_count)
            if random_agent_episodic_investment_return is not None:
                writer.add_scalars(main_tag='episodic investment return ', tag_scalar_dict={'random agent':random_agent_episodic_investment_return}, global_step=train_count)

            writer.add_scalar(tag='memory size', scalar_value=memory_size, global_step=train_count)
            writer.add_scalar(tag='loss', scalar_value=loss, global_step=train_count)

            if train_count % 5000 == 0 and train_count > 0:
                checkpoint_dict = {'train_iter': train_count, 'state_dict': ray.get(parameter_server.send_latest_parameter_to_actor.remote())}
                torch.save(
                    checkpoint_dict,
                    os.path.join(
                        'checkpoint/{}'.format(train_config.model_name),
                        '{}_checkpoint_episode_{}.pth'.format('distributed_dqn', train_count + 1)
                    )
                )

            print('===============================train count : {}================================'.format(train_count))
            print(loss)

if __name__ == '__main__':
    train_config_dict = yaml.load(open(os.path.join(os.getcwd(), 'src/config/train_config.yml'), "r"))
    train_config = namedtuple('train_config', train_config_dict.keys())(**train_config_dict)

    ray.init(num_cpus=int(train_config.actor_total_num * 1.0) + 12, num_gpus=1)
    distributed_train(train_config)