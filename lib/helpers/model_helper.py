from lib.models.gupnet import GUPNet


def build_model(cfg,mean_size):
    if cfg['model']['type'] == 'gupnet':
        return GUPNet(backbone=cfg['model']['backbone'], neck=cfg['model']['neck'], mean_size=mean_size, cfg= cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
