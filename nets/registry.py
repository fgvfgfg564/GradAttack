MODELS = {}

def register_model(name):
    def _register(model_cls):
        MODELS[name] = model_cls
        return model_cls
    return _register

def load_model(name, *args, **kwargs):
    return MODELS[name](*args, **kwargs)