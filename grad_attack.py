import torch
import torch.nn as nn

from criterion import RateDistortionLoss

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
    a = (1. + torch.cos(torch.tensor(t * torch.pi))) * init_lr / 2
    return alpha * init_lr + (1-alpha) * a

def attack(net: nn.Module, img: torch.Tensor, criterion, num_steps=100, init_lr=1e-3, epsilon=.01) -> torch.Tensor:
    """
    Adversarial attack using Projected ADAM optimizer
    """
    if len(img.shape) == 3:
        no_batch = True
        img = img.unsqueeze(0)
    else:
        no_batch = False
    img = img.detach().cuda()
    noise = nn.Parameter(epsilon*(torch.rand_like(img)-0.5))
    # print(input)
    net = net.cuda()
    optimizer = torch.optim.Adam([noise], init_lr)
    net.train()
    for i in range(num_steps):
        optimizer.param_groups[0]['lr'] = cosine_decay(init_lr, i, num_steps, 0.01)

        noise_normed = normalize_l0(noise, epsilon)
        input2 = torch.clamp(noise_normed + img, 0., 1.)
        out = net(input2)
        out["x_hat"] = torch.clamp(out["x_hat"], 0., 1.)
        out_criterion = criterion(out, input2)
        loss_rd = - out_criterion['loss']
        loss = loss_rd

        if i % 10 == 0:
            print(f"Step: {i} "
              f"Loss: {loss.detach().cpu()} "
              f"PSNR_rd: {out_criterion['psnr_loss'].detach().cpu()} " 
              f"BPP_rd: {out_criterion['bpp_loss'].detach().cpu()} ")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    noise_normed = normalize_l0(noise, epsilon)
    output = (noise_normed + img).clamp(0., 1.).detach().cpu()
    output = torch.round(output * 255) / 255.
    if no_batch:
        output = output[0]
    return output