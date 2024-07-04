import sys

from test_on_dataset import parse_args, generate_test_exp_name
from lib.test import calc_performance
from lib.path import *
from lib.utils import AverageMeter
from lib.plotter import plot

ROOTDIR = os.path.split(__file__)[0]


def generate_adv_plot_name(args):
    return (
        f"plots/{args.model}/{args.test_dataset}--{args.dataset}-{args.epochs}x{args.steps_per_epoch}"
        f"-{args.steps}-{args.adv_steps}-{args.adv_optimizer}-{args.adv_lr}-{args.adv_epsilon}"
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    lmbdas = [0.0018, 0.0067, 0.0250, 0.0932]
    result_base = {}
    result_adv = {}
    for lmbda in lmbdas:
        args.lmbda = lmbda
        expname = generate_test_exp_name(args)
        jsonfile_base = os.path.join(ROOTDIR, expname, "baseline.json")
        jsonfile_adv = os.path.join(ROOTDIR, expname, "adversarial.json")

        result_base[lmbda] = load_json(jsonfile_base)
        result_adv[lmbda] = load_json(jsonfile_adv)
    lends = len(result_base[lmbdas[0]].keys())
    print(calc_performance(result_base, result_adv, lends))

    # show R-D curve
    def average_on_key(inputs):
        results = {}
        for item, data in inputs.items():
            for key, value in data.items():
                results.setdefault(key, AverageMeter())
                results[key].update(value)

        for key, value in results.items():
            results[key] = value.avg

        return results

    bpp_base = []
    psnr_base = []
    bpp_adv = []
    psnr_adv = []
    for lmbda in lmbdas:
        result_base_avg = average_on_key(result_base[lmbda])
        bpp_base.append(result_base_avg["bpp"])
        psnr_base.append(result_base_avg["psnr"])
        result_adv_avg = average_on_key(result_adv[lmbda])
        bpp_adv.append(result_adv_avg["bpp"])
        psnr_adv.append(result_adv_avg["psnr"])

    name = generate_adv_plot_name(args)
    plot(
        [bpp_base, bpp_adv],
        [psnr_base, psnr_adv],
        [args.model, args.model + "+adv."],
        title=args.test_dataset,
        save_path=os.path.join(ROOTDIR, name),
    )
