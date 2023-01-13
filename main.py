import torch
import torch.nn as nn
import numpy as np
import random
random_seed = 777

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

model = Net()

def get_param(model):
    for name, param in model.named_parameters():
        np.savetxt("build/"+str(name)+".txt", param.detach())

        print(param.shape)


get_param(model)

print(model(torch.ones(2, 10)))