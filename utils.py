import torch
import cv2
import torch.nn as nn
import numpy as np

from criterion import RateDistortionLoss


def get_recon_bpp(net: nn.Module, img, training=False):
    net.train(training).cuda()
    input = img.cuda().unsqueeze(0).clamp(0., 1.)
    out_net = net(input)
    out_net['x_hat'] = out_net['x_hat'].clamp(0., 1.)
    recon_img = out_net['x_hat'].detach().cpu()[0]
    criterion = RateDistortionLoss()
    out_criterion = criterion(out_net, input, 0)
    bpp = out_criterion['bpp_loss'].detach().cpu()
    psnr = out_criterion['psnr_loss'].detach().cpu()
    return recon_img, bpp, psnr

def dump_torch_image(x):
    x = torch.round(torch.clamp(x, 0., 1.)*255).to(torch.uint8)
    x = x.permute(1, 2, 0)
    x = x.detach().cpu().numpy()
    return x

def jpeg_bpp_psnr_curve(image_tensor: torch.tensor, max_bpp):
    # Ensure image_tensor has shape (C, H, W) with C=3 (for RGB images)
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError("Input image_tensor should have shape (3, H, W) for RGB image.")

    # Convert torch tensor to NumPy array
    image_np = dump_torch_image(image_tensor)

    # Initialize lists to store bitrates and PSNR values
    bitrates = []
    psnr_values = []

    # Loop through different QP values (quantization parameters)
    for qp in range(1, 100):  # You can adjust the range as needed
        # Compress the RGB image using JPEG with the current QP value
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qp]
        _, enc_image = cv2.imencode('.jpg', image_np, encode_param)

        # Calculate the bitrate in bits per pixel (BPP)
        bpp = (len(enc_image) * 8) / (image_np.shape[0] * image_np.shape[1])
        if bpp > max_bpp:
            break

        # Decode the compressed image
        dec_image = cv2.imdecode(enc_image, 1)  # 1 indicates loading the image as a color image (RGB)

        # Calculate the Mean Squared Error (MSE) in RGB space
        mse = np.mean((image_np.astype(float) - dec_image.astype(float)) ** 2)
        
        # Calculate PSNR using the MSE
        psnr = -10 * np.log10(mse / (255**2))

        # Append BPP and PSNR values to the lists
        bitrates.append(bpp)
        psnr_values.append(psnr)

    return bitrates, psnr_values