import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Gumbel(nn.Module):
    def __init__(self, h_dim=500, z_dim=20, n_classes = 10, image_size=784):
        super(VAE_Gumbel, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(h_dim, n_classes)
        self.fc5 = nn.Linear(z_dim + n_classes, h_dim//10)
        self.fc6 = nn.Linear(h_dim//10, image_size)    
        self.n_classes = n_classes
        self.image_size = image_size
        self.z_dim = z_dim
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu, log_var, log_p = self.fc2(h), self.fc3(h), F.log_softmax(self.fc4(h), dim=1)
        return mu, log_var, log_p
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y_onehot):
        latent = torch.cat((z, y_onehot),dim=1)
        h = F.relu(self.fc5(latent))
        x_reconst = torch.sigmoid(self.fc6(h))
        return x_reconst
    
    def forward(self, x, y=None):
        x = x.view(-1, self.image_size)
        mu, log_var, log_p = self.encode(x)
        z = self.reparameterize(mu, log_var)
        if y is not None:
          indices = y.view(-1,1)
          y_onehot = torch.zeros(y.size(0), n_classes).to(device)
          y_onehot.scatter(-1, indices, 1)
        else:
          y_onehot = F.gumbel_softmax(log_p)
        x_reconst = self.decode(z, y_onehot)
        return x_reconst, mu, log_var, log_p
