import compressai.zoo as czoo
from .registry import register_model

@register_model('bmshj2018_factorized')
def bmshj2018_factorized(quality):
    return czoo.bmshj2018_factorized(quality, pretrained=True)

@register_model('bmshj2018_hyperprior')
def bmshj2018_hyperprior(quality):
    return czoo.bmshj2018_hyperprior(quality, pretrained=True)

@register_model('mbt2018')
def mbt2018(quality):
    return czoo.mbt2018(quality, pretrained=True)

@register_model('cheng2020_attn')
def cheng2020_attn(quality):
    return czoo.cheng2020_attn(quality, pretrained=True)