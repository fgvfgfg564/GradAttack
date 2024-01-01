import torch
import torch.nn as nn

def PSNR(x, y):
    mse = torch.mean((x-y)**2)
    psnr = -10 * torch.log10(mse)
    return psnr


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter. L = D + \lambda R"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        likelihoods = output['likelihoods']
        bits_y = -torch.log2(likelihoods['y'])
        bits_z = -torch.log2(likelihoods['z']) if 'z' in likelihoods else torch.tensor(0.)

        out['bpp_y_loss'] = torch.sum(bits_y) / num_pixels
        out['bpp_z_loss'] = torch.sum(bits_z) / num_pixels
        out['bpp_loss'] = out['bpp_y_loss'] + out["bpp_z_loss"]
        out["mse_loss"] = torch.mean((output['x_hat'] - target) ** 2)
        out["loss"] = out["mse_loss"] + lmbda * out["bpp_loss"]
        out["psnr_loss"] = -10 * torch.log10(out["mse_loss"])

        return out