import torch
import torch.nn as nn
import torchvision.models as models

arch = 'squeezenet1_0'
model_file = 'checkpoints/%s_best.pth.tar' % arch
save_pt_path = 'weights/%s_places365.pt' % arch

# https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/8
model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file)
model = nn.DataParallel(model)
model.load_state_dict(checkpoint['state_dict'])


torch.save(model.module.state_dict(), save_pt_path)
print("Finish saving pytorch model")
