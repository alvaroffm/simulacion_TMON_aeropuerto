class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

colorines=dotdict(dict(
blue='#1A2732',
green='#96CE00',
blue_light='#3793FF'
))