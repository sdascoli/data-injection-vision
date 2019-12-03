import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils import *
from model import *

device = 'gpu' if torch.cuda.is_available() else 'cpu'

def train(model, data_loader, num_epochs, C_z_fin = 100, beta=1, alpha=1, print_every=100):
    nmi_scores = []
    model.train(True)
    history = {'reconst':[], 'kl':[], 'label_loss':[]}
    num_steps = num_epochs * len(data_loader)
    step = 0
    
    for epoch in range(num_epochs):
        
        for i, (x, labels) in enumerate(data_loader):
            step+=1
            C_z = C_z_fin * step/num_steps
            
            # Forward pass
            x = x.to(device).view(x.size(0), -1)
            labels = labels.to(device)
            x_reconst, mu, log_var, log_p = model(x) #, labels)
            pred_labels = log_p.max(1)[1]
            p = torch.exp(log_p)
                          
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='none').sum(-1)          
            kl_div = - 0.5 * torch.sum((1 + log_var - mu.pow(2) - log_var.exp()), dim=-1)
            label_loss = F.nll_loss(log_p, labels, reduction='none')          
            
            history['reconst'].append(reconst_loss.mean().item())
            history['kl'].append(kl_div.mean().item())
            history['label_loss'].append(label_loss.mean().item())
            
            # Backprop and optimize
            reservoir_mask = (labels==reservoir_id)
            original_mask = (labels!=reservoir_id)
                        
            losses = reconst_loss + C_z * kl_div + beta * label_loss
            original_part = reconst_loss[original_mask].sum() + C_z * kl_div[original_mask].sum() + beta * label_loss[original_mask].sum()
            reservoir_part = reconst_loss[reservoir_mask].sum() + C_z * kl_div[reservoir_mask].sum() + beta * alpha * label_loss[reservoir_mask].sum()
            
            loss = reservoir_part + original_part
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % print_every == 0:
                print ("Epoch[{}/{}] - Reconst Loss: {:.4f}, KL Div: {:.4f}, Label Loss: {:.4f}" 
                            .format(epoch+1, num_epochs, reconst_loss.mean()/x.size(0),
                                    kl_div.mean()/x.size(0), label_loss.mean()/x.size(0)))
            
    return history

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./', type=str)
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--dataset_type', default='MNIST', type=str)
    parser.add_argument('--batch_size', default=128, type=int) 
    parser.add_argument('--alpha', default=0.1, type=float) 
    parser.add_argument('--num_original', default=50, type=int) 
    parser.add_argument('--num_reservoir', default=500, type=int) 
    parser.add_argument('--num_epochs', default=200, type=int) 
    parser.add_argument('--print_every', default=50, type=int) 
    parser.add_argument('--z_dim', default=32, type=int) 
    parser.add_argument('--learning_rate', default=0.01, type=float) 
    parser.add_argument('--C_z_fin', default=10, type=float) 
    parser.add_argument('--beta', default=10, type=float) 
    args = parser.parse_args()

    dataset = getattr(torchvision.datasets, args.dataset_type)(root=args.data_dir,
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

    original_labels=[0,1,2,3,4,5]
    reservoir_labels = [0,1,2,3,4,5,6,7,8,9]
    reservoir_id = len(original_labels)+1
    
    new_dataset = create_new_dataset(dataset,
                                     num_original=args.num_original,
                                     num_reservoir=args.num_reservoir,
                                     original_labels=original_labels,
                                     reservoir_labels=reservoir_labels,
                                     reservoir_id=reservoir_id)
    new_data_loader = torch.utils.data.DataLoader(dataset=new_dataset,
                                                  batch_size=args.batch_size, 
                                                  shuffle=True)

    model = VAE_Gumbel(z_dim=args.z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print('Training for alpha = {}'.format(args.alpha))
    hist = train(model, new_data_loader, num_epochs=args.num_epochs, C_z_fin=args.C_z_fin, beta=args.beta, alpha=args.alpha, print_every=args.print_every)

    generated_name = args.save_dir+'generated_{0}_{1}_{2}_{3:.1f}.png'.format(args.dataset_type, args.num_original, args.num_reservoir, args.alpha)
    hist_name = args.save_dir+'hist_{0}_{1}_{2}_{3:.1f}.png'.format(args.dataset_type, args.num_original, args.num_reservoir, args.alpha)

    plot_conditional_generation(model, save_as = generated_name, labels=original_labels)
    plot_hist(hist, save_as = hist_name)
