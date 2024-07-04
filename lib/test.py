import torch
import torch.nn as nn
import compressai
from compressai.utils.eval_model.__main__ import compute_metrics, inference
from .bdrate import BD_RATE, BD_PSNR


def mean(x):
    return sum(x) / len(x)


@torch.no_grad()
def calc_performance(
    result: dict, result_anchor: dict, dataset_size: int, typ="bd-rate", reduce=True
):
    """
    result1 & result2: {
        lmbda: <result from test_on_dataset_function>
    }
    """
    typ = typ.lower()
    if typ == "bd-rate":
        metricfunc = BD_RATE
    elif typ == "bd-psnr":
        metricfunc = BD_PSNR
    else:
        raise ValueError("Invalid reduction type:", typ)

    lmbdas = result.keys()
    results = {}

    # BD-rate over images
    for img_id in range(dataset_size):
        metrics1 = {}
        metrics2 = {}
        for lmbda in lmbdas:
            result1_sample = result[lmbda][str(img_id)]
            result2_sample = result_anchor[lmbda][str(img_id)]
            for metric_name, value in result1_sample.items():
                metrics1.setdefault(metric_name, [])
                metrics1[metric_name].append(value)
            for metric_name, value in result2_sample.items():
                metrics2.setdefault(metric_name, [])
                metrics2[metric_name].append(value)

        for metric_name in result2_sample.keys():
            if metric_name == "bpp":
                continue
            if "time" in metric_name.lower():
                # Time metric
                t1 = mean(metrics1[metric_name])
                t2 = mean(metrics2[metric_name])
                delta = (t1 - t2) / t2
                results.setdefault("delta_" + metric_name, [])
                results["delta_" + metric_name].append(delta)
            else:
                # Distortion
                bd = metricfunc(
                    metrics1["bpp"],
                    metrics1[metric_name],
                    metrics2["bpp"],
                    metrics2[metric_name],
                    piecewise=1,
                )
                if metric_name == "psnr":
                    print(img_id, bd)
                results.setdefault(f"{typ}({metric_name})", [])
                results[f"{typ}({metric_name})"].append(bd)

    if reduce:
        for key, value in results.items():
            results[key] = mean(value)

    return results


@torch.no_grad()
def test_on_dataset(net: compressai.models.CompressionModel, dataloader):
    net.update()
    net.eval()
    device = next(net.parameters()).device
    loss_dict = {}

    for i, d in enumerate(dataloader):
        print(f"Testing image #{i}", flush=True)
        if isinstance(d, torch.Tensor):
            d = d.to(device)
        else:
            d_new = []
            for each in d:
                d_new.append(each.to(device))
            d = tuple(d_new)

        out_criterion = inference(net, d[0])

        loss_dict[i] = out_criterion

        sample_log = ""
        for k, v in out_criterion.items():
            sample_log += f"{k}={v:.5f}\n"
        print(sample_log, flush=True)

    return loss_dict
