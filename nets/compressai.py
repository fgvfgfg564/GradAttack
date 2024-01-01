import compressai.zoo as czoo
from .registry import register_model

COMPRESSAI_LAMBDAS = {
    "MSE": [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800],
    "MS-SSIM": [2.40,4.58,8.73,16.64,31.73,60.50,115.37,220.00]
}

def translate_lambda(lmbda_cai):
    return 1. / (lmbda_cai * 255 ** 2)

LMBDAS_MSE_CAI = [0] + list([translate_lambda(x) for x in COMPRESSAI_LAMBDAS['MSE']])

@register_model('bmshj2018_factorized', LMBDAS_MSE_CAI)
def bmshj2018_factorized(quality):
    return czoo.bmshj2018_factorized(quality, pretrained=True)

@register_model('bmshj2018_hyperprior', LMBDAS_MSE_CAI)
def bmshj2018_hyperprior(quality):
    return czoo.bmshj2018_hyperprior(quality, pretrained=True)

@register_model('mbt2018', LMBDAS_MSE_CAI)
def mbt2018(quality):
    return czoo.mbt2018(quality, pretrained=True)

@register_model('cheng2020_attn', LMBDAS_MSE_CAI)
def cheng2020_attn(quality):
    return czoo.cheng2020_attn(quality, pretrained=True)