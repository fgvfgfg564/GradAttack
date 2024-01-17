import sys

from train_adversarial import parse_args, generate_adv_exp_name
from lib.test import calc_performance
from lib.path import *

ROOTDIR = os.path.split(__file__)[0]

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    lmbdas = [0.0018, 0.0067, 0.0250, 0.0932]
    result_base = {}
    result_adv = {}
    for lmbda in lmbdas:
        args.lmbda = lmbda
        expname = generate_adv_exp_name(args)
        jsonfile_base = os.path.join(ROOTDIR, expname, 'baseline.json')
        jsonfile_adv = os.path.join(ROOTDIR, expname, 'adversarial.json')

        result_base[lmbda] = load_json(jsonfile_base)
        result_adv[lmbda] = load_json(jsonfile_adv)
    lends = len(result_base[lmbdas[0]].keys())
    print(calc_performance(result_base, result_adv, lends))
