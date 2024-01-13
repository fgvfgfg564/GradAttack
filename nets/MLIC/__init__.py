from .config.config import model_config
from .models.mlicpp import MLICPlusPlus
from ..registry import register_model


@register_model('mlicpp', None)
def mlicpp(_):
    return MLICPlusPlus(model_config())