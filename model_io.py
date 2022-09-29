import os

import torch


def save_weights(model, filename, path="./saved_models"):
    if not os.path.isdir(path):
        os.makedirs(path)

    fpath = os.path.join(path, filename)
    torch.save(model.state_dict(), fpath)
    return


def save_checkpoint(model, optimizer, epoch, step, best_loss, filename, root="./checkpoints"):
    if not os.path.isdir(root):
        os.makedirs(root)

    fpath = os.path.join(root, filename)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_loss": best_loss
        }
        , fpath)


def load_weights(model, filename, path="./saved_models"):
    fpath = os.path.join(path, filename)
    state_dict = torch.load(fpath)
    model.load_state_dict(state_dict)
    return model


def load_checkpoint(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']
    best_loss = ckpt['best_loss']
    step = ckpt['step']
    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model, optimizer, epoch, step, best_loss
