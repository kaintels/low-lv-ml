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


model = nn.Linear(10, 1)


a = dict(model.state_dict())

aw = a["weight"].numpy()
ab = a["bias"].numpy()

# np.savetxt("aw.txt", aw)
# np.savetxt("ab.txt", ab)
print(model(torch.ones(10)))