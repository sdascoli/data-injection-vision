import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import copy

device = 'gpu' if torch.cuda.is_available() else 'cpu'
   
def plot_conditional_generation(model, save_as=None, labels=range(10), n_per_label=8, n_classes=10, fix_number=None):

    image_format = (-1, 1, 28, 28)
  
    plt.figure(figsize=(10,10))
  
    with torch.no_grad():
        matrix = np.zeros((n_per_label,n_classes))
        matrix[:,0] = 1

    if fix_number is None:
        final = np.roll(matrix,labels[0])
        for i in labels[1:]:
            final = np.vstack((final,np.roll(matrix,i)))
        z = torch.randn(n_per_label*len(labels), model.z_dim).to(device)
        y_onehot = torch.tensor(final).type(torch.FloatTensor).to(device)
        out = model.decode(z,y_onehot).view(*image_format)
    else:
        z = torch.randn(n_per_label, model.z_dim).to(device)
        y_onehot = torch.tensor(np.roll(matrix, fix_number)).type(torch.FloatTensor).to(device)
        out = model.decode(z,y_onehot).view(*image_format)
        
    out_grid = torchvision.utils.make_grid(out).cpu()
    npimg = out_grid.detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

    plt.axis('off')
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    
def plot_hist(hist, save_as = None):
    fig, axarr = plt.subplots(1,3, figsize=(15,4))
    steps = range(len(hist['reconst']))
    axarr[0].plot(hist['reconst'])
    axarr[1].plot(hist['kl'])
    axarr[2].plot(hist['label_loss'])
    axarr[0].set_ylabel('reconst')
    axarr[1].set_ylabel('kl')
    axarr[2].set_ylabel('label_loss')
    for i in range(3):
        axarr[i].set_xlabel('Iteration')
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)

def create_new_dataset(dataset, num_original, num_reservoir, original_labels, reservoir_labels, reservoir_id, batch_size=64, shuffle=True):
  
    new_dataset = copy.deepcopy(dataset)
    
    original_idx = [False]*len(dataset)
    reservoir_idx = [False]*len(dataset)
    originals = 0
    reservoirs = 0
    for i, (data, label) in enumerate(dataset):
        if originals==num_original and reservoirs==num_reservoir:
            break
        if label in original_labels and originals<num_original:
            original_idx[i] = True
            originals+=1
        if label in reservoir_labels and reservoirs<num_reservoir:
            reservoir_idx[i] = True
            reservoirs+=1
    
    new_dataset.targets = torch.cat((torch.LongTensor(dataset.targets)[original_idx], torch.LongTensor([reservoir_id]*sum(reservoir_idx))))
    new_dataset.data = torch.cat((dataset.data[original_idx], dataset.data[reservoir_idx]))
    print(new_dataset)
    return new_dataset
