import torch.nn as nn

MODELS = {}
LMBDAS = {}

def register_model(name, lmbdas):
    def _register(model_cls):
        MODELS[name] = model_cls
        LMBDAS[name] = lmbdas
        return model_cls
    return _register

def load_model(name, parameter_set=None) -> nn.Module:
    return MODELS[name](parameter_set)

def load_lmbda(name, parameter_set) -> float:
    return LMBDAS[name][parameter_set]