"""Author: Hyung-Kwon Ko (hyungkwonko@gmail.com)"""
import os
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import argparse
from torchvision import transforms

from fc_layer import FC_Model
import vggface2.senet as SENet
from datasets.vggface2_sg2 import StyleGAN2_Data as SDV
from datasets.sg2 import StyleGAN2_Data as SDF


data_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


def load_networks(args, device):
    # load generator
    with dnnlib.util.open_url(args.G_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    
    # load auxiliary mapping
    AUX = FC_Model().to(device)
    AUX.load_state_dict(torch.load(args.AUX_path))

    # load classifier
    SE = SENet.senet50(num_classes=136, include_top=True).to(device)  # forward output w/ FC layer (dim: 138=NUM_OUT_FT)
    SE.load_state_dict(torch.load(args.SE_path))

    return G, AUX, SE


def sgf():

    parser = argparse.ArgumentParser("SGF")

    parser.add_argument('--G_path', type=str, default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl')
    parser.add_argument('--SE_path', type=str, default='ckpt/senet_ckpt_0.001_out.pth')
    parser.add_argument('--AUX_path', type=str, default='ckpt/adam_model_0.0002_8_6_0.0.pth')
    parser.add_argument('--val_x', type=int, default=0)
    parser.add_argument('--val_y', type=int, default=20)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--step_size', type=float, default=1.0)
    parser.add_argument('--cutoff', type=float, default=1e-8)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--truncation_psi', type=float, default=0.8)
    parser.add_argument('--noise_mode', type=str, default='const', choices=['const', 'random', 'none'])
    parser.add_argument('--outdir', type=str, default='out/sgf')
    parser.add_argument('--save_result', type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda')

    print('Loading G, AUX, SE networks ...')
    G, AUX, SE = load_networks(args, device)

    print(f'Manipulating image for seed --> ({args.seed}) ...')
    z0 = torch.from_numpy(np.random.RandomState(args.seed).randn(1, G.z_dim)).to(device).to(torch.float)
    label = torch.zeros([1, G.c_dim], device=device)
    img = G(z0, label, truncation_psi=args.truncation_psi, noise_mode=args.noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.float)

    if args.save_result:
        fname = f'{args.outdir}/seed{args.seed:05d}_{args.step_size}.png'
        PIL.Image.fromarray(img.to(torch.uint8)[0].detach().cpu().numpy(), 'RGB').resize((args.resize, args.resize), PIL.Image.ANTIALIAS).save(fname)

    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').resize((args.resize, args.resize), PIL.Image.ANTIALIAS)
    img = data_transforms(img).unsqueeze(0).to(device)

    print(f'Load data pre-processing scalers ...')
    data_v = SDV(split='test', transform=data_transforms, scale_size=256)
    data_f = SDF(split='test')
 
    c0 = SE(img)
    c0 = c0.cpu().detach().numpy()
    c0 = data_v.inv_scale_label(c0)
    c1 = c0.copy()

    print(f'Manipulate labels (c) ...')
    c1[0, ::2] += args.val_x
    c1[0, 1::2] += args.val_y

    c0 = data_f.scale_val_label(c0)
    c0 = torch.tensor(c0).to(device).to(torch.float)
    c1 = data_f.scale_val_label(c1)
    c1 = torch.tensor(c1).to(device).to(torch.float)

    delta_c = args.step_size * (c1 - c0)  # Algo1: line5

    for i in range(args.n):

        z_out0, _ = AUX(z0, c0)
        z_out1, _ = AUX(z0, c0 + delta_c)
        delta_z = z_out1 - z_out0  # Algo1: line9

        z_out1, _ = AUX(z0 + delta_z, c0)
        delta_z += (z_out1 - z_out0)  # Algo1: line12

        z0 += delta_z  # Algo1: line14

        img = G(z0, label, truncation_psi=args.truncation_psi, noise_mode=args.noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.float)

        if args.save_result:
            fname = f'{args.outdir}/seed{args.seed:05d}_{args.step_size}_{i}.png'
            PIL.Image.fromarray(img.to(torch.uint8)[0].detach().cpu().numpy(), 'RGB').resize((args.resize, args.resize), PIL.Image.ANTIALIAS).save(fname)

        img = PIL.Image.fromarray(img[0].detach().cpu().numpy(), 'RGB').resize((args.resize, args.resize), PIL.Image.ANTIALIAS)
        img = data_transforms(img).unsqueeze(0).to(device)

        c_out = SE(img)  # Algo1: line15
        c_out = c_out.cpu().detach().numpy()
        c_out = data_v.inv_scale_label(c_out)
        c_out = data_f.scale_val_label(c_out)
        c_out = torch.tensor(c_out).to(device).to(torch.float)

        loss = (c_out - c0).mean().item()
        c0 = c_out

        print(f"[INFO] {i}-th iteration: loss: {loss} ...")

        if abs(loss) < args.cutoff:  # Algo1: line16
            break
   


if __name__ == "__main__":
    sgf()