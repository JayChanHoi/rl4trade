import torch

def rescale(x, eps=1.0e-3):
    return x.sign() * ((x.abs() + 1.0).sqrt() - 1.0) + eps * x

def inv_rescale(x, eps=1.0e-3):
    if eps == 0:
        return x.sign() * (x * x + 2.0 * x.abs())
    else:
        return x.sign() * ((((1.0 + 4.0 * eps * (x.abs() + 1.0 + eps)).sqrt() - 1.0) / (2.0 * eps)).pow(2) - 1.0)

def resume(model, device, resume_checkpoint):
    print('=> loading checkpoint : "{}"'.format(resume_checkpoint))
    if str(device) == 'cpu':
        model.load_state_dict(torch.load(resume_checkpoint, map_location=torch.device('cpu'))['state_dict'])
    else:
        model.load_state_dict(torch.load(resume_checkpoint)['state_dict'])

def write_config(writer, train_config, env_config):
    for config in [train_config, env_config]:
        for key in config._asdict().keys():
            writer.add_text(key, str(config._asdict()[key]), global_step=0)
