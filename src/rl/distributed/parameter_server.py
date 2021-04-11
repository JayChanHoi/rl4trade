import ray

@ray.remote(num_cpus=1)
class ParameterServer(object):
    def __init__(self, learner_state_dict):
        self.state_dict = learner_state_dict

    def update_ps_state_dict(self, new_learner_state_dict):
        self.state_dict = new_learner_state_dict

    def send_latest_parameter_to_actor(self):
        return self.state_dict