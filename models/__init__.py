from .trans_vg_lt import TransVG


def build_model(args,config):
    return TransVG(args,config)


