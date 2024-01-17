import torch
import torch.nn as nn
import math
import time
import torch.nn.functional as F
from lib.trainer import AverageMeter
from criterion import RateDistortionLoss
from compressai.utils.eval_model.__main__ import compute_metrics

# def normalize_l2(x, delta):
#     mx = x.square().mean()
#     if mx > 0.:
#         mx2 = torch.minimum(mx, torch.tensor(delta))
#         x = x * (mx2 / mx).sqrt()
#     return x

def normalize_l0(x, delta):
    x = x.clamp(torch.tensor(-delta,device=x.device), torch.tensor(delta,device=x.device))
    return x

def cosine_decay(init_lr, step, num_steps, alpha):
    t = step / num_steps
    a = (1. + math.cos(t * math.pi)) * init_lr / 2
    return alpha * init_lr + (1-alpha) * a

@torch.enable_grad()
def attack(net: nn.Module, img: torch.Tensor, criterion, optimizer_class=torch.optim.Adam, num_steps=100, init_lr=1e-3, epsilon=.01, inverse=False, show_info=False) -> torch.Tensor:
    """
    Adversarial attack using Projected ADAM optimizer
    """
    device = img.device
    if len(img.shape) == 3:
        no_batch = True
        img = img.unsqueeze(0)
    else:
        no_batch = False
    img = img.detach().cuda()
    if inverse:
        noise = nn.Parameter(torch.zeros_like(img))
    else:
        noise = nn.Parameter(epsilon*(torch.rand_like(img)-0.5))
    # print(input)
    net = net.cuda()
    optimizer = optimizer_class((noise,), init_lr)
    net.train()
    for i in range(num_steps):
        optimizer.param_groups[0]['lr'] = cosine_decay(init_lr, i, num_steps, 0.01)

        noise_normed = normalize_l0(noise, epsilon)
        input2 = torch.clamp(noise_normed + img, 0., 1.)
        out = net(input2)
        out["x_hat"] = torch.clamp(out["x_hat"], 0., 1.)
        if inverse:
            out_criterion = criterion(out, img)
            loss_rd = out_criterion['loss']
        else:
            out_criterion = criterion(out, input2)
            loss_rd = - out_criterion['loss']
        loss = loss_rd

        if show_info:
            print(f"Step: {i} "
              f"Loss: {loss.detach().cpu()} "
              f"PSNR_rd: {out_criterion['psnr_loss'].detach().cpu()} " 
              f"BPP_rd: {out_criterion['bpp_loss'].detach().cpu()} ")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    noise_normed = normalize_l0(noise, epsilon)
    output = (noise_normed + img).clamp(0., 1.).detach().to(device)
    output = torch.round(output * 255) / 255.
    if no_batch:
        output = output[0]
    return output


@torch.no_grad()
def inference(model, x, x_ref=None):
    if x_ref is None:
        x_ref = x
    x = x.unsqueeze(0)
    x_ref = x_ref.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    x_ref_padded = F.pad(
        x_ref,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    # input images are 8bit RGB for now
    metrics = compute_metrics(x_ref_padded, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    return {
        "psnr": metrics["psnr"],
        "ms-ssim": metrics["ms-ssim"],
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def adversarial_test(net: nn.Module, dataloader, adversarial=False, lmbda=1., epsilon=0.01, steps=100):
    net.eval()
    device = next(net.parameters()).device
    loss_dict = {}
    loss_ideal_dict = {}

    for i, d in enumerate(dataloader):
        print(f"Testing image #{i}", flush=True)
        if isinstance(d, torch.Tensor):
            d = d.to(device)
        else:
            d_new = []
            for each in d:
                d_new.append(each.to(device))
            d = tuple(d_new)
        
        if adversarial:
            criterion = RateDistortionLoss(lmbda)
            d_origin = d
            d = attack(net, d, criterion, steps, epsilon=epsilon).to(device)

        out_criterion = inference(net, d[0])

        for key, value in out_criterion.items():
            loss_dict.setdefault(key, AverageMeter())
            loss_dict[key].update(value)
        
        if adversarial:
            out_criterion = inference(net, d_origin[0], d[0])

            for key, value in out_criterion.items():
                loss_ideal_dict.setdefault(key, AverageMeter())
                loss_ideal_dict[key].update(value)

        sample_log = ""
        for k, v in out_criterion.items():
            sample_log += f"{k}={v:.5f}\n"
        print(sample_log, flush=True)

    result = {k:v.avg for k,v in loss_dict.items()}
    if adversarial:
        result_ideal = {k:v.avg for k,v in loss_ideal_dict.items()}
    else:
        result_ideal = None
    return result, result_ideal

@torch.no_grad()
def adversarial_tests(net: nn.Module, dataloader, lmbda=1., epsilon=0.01, steps=100):
    result, _ = adversarial_test(net, dataloader)
    result_adv, result_ideal = adversarial_test(net, dataloader, True, lmbda, epsilon, steps)
    return result, result_adv, result_ideal