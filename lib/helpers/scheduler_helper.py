import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import math


def build_lr_scheduler(cfg, optimizer, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return cur_decay

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    warmup_lr_scheduler = None
    if cfg['warmup']:
        warmup_lr_scheduler = CosineWarmupLR(optimizer, num_epoch=5, init_lr=0.00001)
    return lr_scheduler, warmup_lr_scheduler


def build_bnm_scheduler(cfg, model, last_epoch):
    if not cfg['enabled']:
        return None

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return max(cfg['momentum']*cur_decay, cfg['clip'])

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.init_lr + (base_lr - self.init_lr) *
                (1 - math.cos(math.pi * self.last_epoch / self.num_epoch)) / 2
                for base_lr in self.base_lrs]


class LinearWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.init_lr + (base_lr - self.init_lr) * self.last_epoch / self.num_epoch
                for base_lr in self.base_lrs]



if __name__ == '__main__':
    # testing
    import torch.optim as optim
    from lib.models.centernet3d import CenterNet3D
    import matplotlib.pyplot as plt

    net = CenterNet3D()
    optimizer = optim.Adam(net.parameters(), 0.01)
    lr_warmup_scheduler_cosine = CosineWarmupLR(optimizer, 1000, init_lr=0.00001, last_epoch=-1)
    lr_warmup_scheduler_linear = LinearWarmupLR(optimizer, 1000, init_lr=0.00001, last_epoch=-1)

    batch_cosine, lr_cosine = [], []
    batch_linear, lr_linear = [], []

    for i in range(1000):
        batch_cosine.append(i)
        lr_cosine.append(lr_warmup_scheduler_cosine.get_lr())
        batch_linear.append(i)
        lr_linear.append(lr_warmup_scheduler_linear.get_lr())
        lr_warmup_scheduler_cosine.step()
        lr_warmup_scheduler_linear.step()

    # vis
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.scatter(batch_cosine, lr_cosine, c = 'r',marker = 'o')
    ax2 = fig.add_subplot(122)
    ax2.scatter(batch_linear, lr_linear, c = 'r',marker = 'o')
    plt.show()



