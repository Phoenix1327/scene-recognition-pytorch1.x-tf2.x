import torch
import torch.nn as nn
import torchvision.models as models


arch = 'resnet50'
save_pt_path = 'weights/%s_places365.pt' % arch
convert_pt_path = 'weights/%s_places365_torchscript.pt' % arch
# device = torch.device("cuda:2")

model = models.__dict__[arch](num_classes=365)
model.load_state_dict(torch.load(save_pt_path))
# model.to(device)
model.eval()

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save(convert_pt_path)
print("Finish saving torchscript model")