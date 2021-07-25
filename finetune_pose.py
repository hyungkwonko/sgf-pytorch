"""Author: Hyung-Kwon Ko (hyungkwonko@gmail.com)"""
import os
import time
import copy
from tqdm import tqdm
import logging
import argparse
import collections

from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from vggface2.utils import load_state_dict
import vggface2.senet as SENet
from datasets.vggface2_sg2 import StyleGAN2_Data


def save_image(input, outdir, pred):
    plot_style = dict(marker='o', markersize=4, linestyle='-', lw=2)
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                }

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(input)

    for pred_type in pred_types.values():
        ax.plot(pred[pred_type.slice, 0],
                pred[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    plt.savefig(outdir)


def train(args):

    logging.info('Initializing Datasets and Data_loader...')

    data_transforms = transforms.Compose([
                        transforms.Resize(args.pretrained_size),
                        transforms.CenterCrop(args.pretrained_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    data = {
        'train': StyleGAN2_Data(root=args.root, split='train', lname=args.lname, transform=data_transforms, scale_size=args.image_size),
        'val': StyleGAN2_Data(root=args.root, split='val', lname=args.lname, transform=data_transforms, scale_size=args.image_size)
        }

    data_loader = {
        'train': torch.utils.data.DataLoader(data['train'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False),
        'val': torch.utils.data.DataLoader(data['val'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = initialize_model(args)
    model = model.to(device)
    logging.info(model)

    logging.info('Params to learn:')
    if args.feature_extract:
        params_to_update = []
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad == True:
                params_to_update.append(param)
                logging.info(f'{i}. param.name: {name}')
                # logging.info(f'{i}. param.shape: {param.shape}')
    else:
        params_to_update = model.parameters()
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad == True:
                logging.info(f'{i}: {name}')

    optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=1e-5)  # l2 norm
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.MSELoss()

    logging.info('Training starts!')

    since = time.time()

    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(args.num_epochs):

        logging.info('-' * 10)
        logging.info(f'Epoch {epoch}/{args.num_epochs - 1}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                if epoch % 1 == 0:
                    model.eval()
                else:
                    continue

            running_loss = 0.0

            for batch in tqdm(data_loader[phase]):

                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                indices = batch['meta']['index']

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.to(float)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(data_loader[phase].dataset)

            logging.info(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                val_loss_history.append(epoch_loss)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(args.ckpt_dir, f'senet_ckpt_{args.lr}_{args.weight_decay}_{args.feature_extract}_{args.out_features}.pth'))

                if args.save_fig and epoch % args.save_epoch == 0:
                    outputs = outputs.cpu().numpy()          

                    for i, (index, output) in enumerate(zip(indices, outputs)):
                        if i > 3:
                            break
                        output = output.reshape(1, -1)
                        pred = data[phase].inv_scale_label(output)
                        pred = pred.reshape(-1, 2)

                        input = data[phase].get_image(index, os.path.join(args.root, phase))
                        save_image(input, os.path.join(args.val_save_dir, f'out_{index}_{epoch}.png'), pred)

                        if epoch == 0:
                            answ = data[phase].labels_original[index]
                            answ = answ.reshape(-1, 2)
                            save_image(input, os.path.join(args.val_save_dir, f'out_{index}_ans.png'), answ)
        scheduler.step()

    time_elapsed = time.time() - since

    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Loss: {best_loss:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_loss_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(args):
    model = SENet.senet50(num_classes=args.n_identity, include_top=True)  # forward output w/ FC layer (dim: 138=NUM_OUT_FT)
    # model = SENet.senet50(num_classes=N_IDENTITY, include_top=False)  # forward output w/o FC layer (dim: 2048)

    if args.mode == 'train':
        load_state_dict(model, args.pretrained_path)
        set_parameter_requires_grad(model, args.feature_extract)
        num_in_ft = model.fc.in_features  # 2048
        model.fc = nn.Linear(num_in_ft, args.out_features)
        return model
    else:
        num_in_ft = model.fc.in_features  # 2048
        model.fc = nn.Linear(num_in_ft, args.out_features)
        model.load_state_dict(torch.load(args.model_path))
        return model


def test(args):

    model = initialize_model(args)

    data_transforms = transforms.Compose([
                        transforms.Resize(args.pretrained_size),
                        transforms.CenterCrop(args.pretrained_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    data = StyleGAN2_Data(root=args.root, split='test', fname=args.lname, transform=data_transforms, scale_size=args.image_size)
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()

    running_loss = 0.0

    criterion = nn.MSELoss()

    for batch in tqdm(data_loader):

        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        outputs = model(inputs)

        with torch.no_grad():
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)

    print(f'Loss: {epoch_loss:.4f}')

    # for i in range(5):
    #     print(f"Sample latents {labels[i].cpu().detach().numpy()}, Sample output: {outputs[i].cpu().detach().numpy()}")



def main():
    parser = argparse.ArgumentParser("PyTorch SE-Net Fine-tuning Code")

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train/test')

    parser.add_argument('--image_size', type=int, default=256, help='training data size == (x.size[0])')
    parser.add_argument('--out_features', type=int, default=3, help='number of classes == y')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to run')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='l2 norm')
    parser.add_argument('--feature_extract', type=int, choices=[0, 1], default=0, help='finetune fc layer only (True, 1) / all layers (False, 0)')

    parser.add_argument('--root', type=str, default='data', help='training data dir')
    parser.add_argument('--lname', type=str, default='pose', help='label file name')
    parser.add_argument('--pretrained_path', type=str, default='./pretrained/senet50_scratch_weight.pkl', help='pretrained SENet model path')
    parser.add_argument('--model_path', type=str, default='./ckpt/senet_ckpt_0.001.pth', help='pretrained SENet model path')

    parser.add_argument('--save_epoch', type=int, default=1, help='epoch to save images for validation')
    parser.add_argument('--val_save_dir', type=str, default='./out/finetune', help='save directory for validation file')
    parser.add_argument('--log_dir', type=str, default='log', help='save directory for log file')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='save directory for checkpoint file')

    parser.add_argument('--n_identity', type=int, default=8631, help='number of classes pretrained in SENet')
    parser.add_argument('--pretrained_size', type=int, default=224, help='image size pretrained in SENet')
    parser.add_argument('--save_fig', type=int, default=0, help='save predicted result as figures')

    args = parser.parse_args()

    os.makedirs(args.val_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, f'senet_finetune_{datetime.now().time()}.log')),
            logging.StreamHandler()
        ]
    )

    logging.info(f'PyTorch Version: {torch.__version__}')
    logging.info(f'Torchvision Version: {torchvision.__version__}')
    logging.info(f'args: {args}')

    if args.mode == 'train':
        train(args)
    else:
        test(args)



if __name__ == '__main__':
    main()