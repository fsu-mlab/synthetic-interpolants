# 12/07/2021
# Architecture search across depth

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast 
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from model import Inverter
from loss import InversionLoss

import os
import click
import dnnlib
import legacy
from PIL import Image

def loader(path):
    img = Image.open(path).convert('RGB')
    return img

class Dataset(DatasetFolder):
    """
    Simple dataset for loading in numerical directory images properly
    """
    def __init__(self, root: str, loader: callable, transform=None):
        super().__init__(root, loader=loader, transform=transform, extensions=('png', 'jpg'))
    
    def find_classes(self, path):
        folders = [p for p in os.listdir(path) if os.path.isdir(f'{path}/{p}')]
        return (folders, {f: int(f) for f in folders})

@click.command()

# Required
@click.option('--outdir',       help='Where to save the results',       metavar='DIR', required=True)
@click.option('--data',         help='Training data',                   metavar='[ZIP|DIR]', type=str, required=True)
@click.option('--batch',        help='Total batch size',                metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--pkl',          help='Network pickle file',             metavar='STR', type=str, required=True)

# Optional 
@click.option('--workers',      help='DataLoader worker processes',     metavar='INT', type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--epochs',       help='Total training epochs',           metavar='INT', type=click.IntRange(min=1), default=300, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision',         metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking',      metavar='BOOL',type=bool, default=False, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU',        metavar='INT', type=click.IntRange(min=1))
@click.option('--layers',       help='Layers to test in the encoder',   metavar='INT', type=click.IntRange(min=3))


def search(**kwargs):
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda')
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.

    # Initialize SummaryWriter
    runs = 0
    if (os.path.exists(opts['outdir'])):
        for folders in os.listdir(opts['outdir']):
            runs += 1  

    writer = SummaryWriter(log_dir=f"{opts['outdir']}/{str(runs).zfill(5)}_bs{opts['batch']}")

    # Determine the number of classes
    num_classes = 0  

    for folders in os.listdir(opts['data']):
        num_classes += 1  

    # Define transforms 
    tsfms = transforms.Compose([
        transforms.Resize(256, interpolation=1),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Calculate number of steps to accumulate gradient 
    accumulate_steps = (opts['batch'] // opts['batch_gpu']) if opts.get('batch_gpu') else 1

    dataset = Dataset(opts['data'], loader, transform=tsfms)
    dataloader = DataLoader(
        dataset, 
        shuffle=True, 
        batch_size=opts['batch_gpu'] if opts.get('batch_gpu') else opts['batch'], 
        drop_last=True,
        num_workers=opts['workers']
    )

    iters_per_epoch = len(dataloader)

    inversion_loss = InversionLoss()

    # Load Generator in evaluation mode
    with dnnlib.util.open_url(opts['pkl']) as f:
        G = legacy.load_network_pkl(f)['G_ema'].synthesis
        G.to(device)
    G.eval()
    print("Generator network: ")
    print(G)

    with dnnlib.util.open_url(opts['pkl']) as f:
        D = legacy.load_network_pkl(f)['D'].to(device)
    D.train()
    print("Discriminator network: ")
    print(D)

    I = Inverter(7, 4, 3, G.w_dim)
    I.to(device)
    I.train()
    print("Inversion network: ")
    print(I)

    # Initialize optimizers
    optim_d = AdamW(D.parameters())
    optim_i = AdamW(I.parameters(), lr=0.01)
    scheduler_i = CosineAnnealingWarmRestarts(optim_i, 10)

    print(f"\nTraining for {opts['epochs']} epochs with batch size {str(opts['batch_gpu']) + ' and total batch ' + str(opts['batch']) if opts.get('batch_gpu') else int(opts['batch'])} on {num_classes} classes...\n")

    for epoch in range(opts['epochs']):
        running_loss_d = 0.
        running_loss_i = 0.

        # Iterate over dataset 
        iters = 0
        imgs, reconsts = 0, 0

        for i, (imgs, labels) in enumerate(dataloader, 0):
            with autocast():
                imgs = imgs.cuda()

                # Encode the batch
                w_pred = I(imgs)
                labels = F.one_hot(torch.tensor(labels), num_classes).float().cuda()

            # Pass the batch through the generator
            reconsts = torch.tanh(G(w_pred)) 

            with autocast():

                # Pass the images through the discriminator
                fake_score = D(reconsts, labels)
                real_score = D(imgs, labels)

                # Calculate the loss 
                loss_i = inversion_loss(imgs, reconsts, fake_score)
                loss_d = fake_score.mean() - real_score.mean()

                running_loss_i += loss_i.item()
                running_loss_d += loss_d.item()

                # Determine whether or not to back prop
                if (iters + 1) % accumulate_steps == 0:
                    iters = 0
                    loss_i.backward(retain_graph=True)
                    loss_d.backward()

                    optim_i.step()
                    optim_d.step()

                    optim_i.zero_grad(set_to_none=True)  
                    optim_d.zero_grad(set_to_none=True)      
            
            iters += 1

        print(f"Running loss of the discriminator at epoch {epoch + 1}: {running_loss_d/ iters_per_epoch}")
        print(f"Running loss of the inverter at epoch {epoch + 1}: {running_loss_i/ iters_per_epoch}")

        writer.add_scalar('Loss/discriminator', running_loss_d/ iters_per_epoch, epoch + 1)
        writer.add_scalar('Loss/inverter', running_loss_i/ iters_per_epoch, epoch + 1)
        writer.add_scalar('LR/inverter', scheduler_i.get_last_lr()[0], epoch + 1)
        writer.add_image('Image/real', make_grid(imgs, normalize=True), epoch + 1)
        writer.add_image('Image/reconstruction', make_grid(reconsts, normalize=True), epoch + 1)

        scheduler_i.step()   

        # TODO: Save the discriminator and encoder

if __name__ == "__main__":
    search()