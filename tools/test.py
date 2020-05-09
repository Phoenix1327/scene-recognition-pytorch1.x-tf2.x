import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

print(torch.__version__)

arch = 'resnet50'
# model_file = 'checkpoints/%s_best.pth.tar' % arch
save_pt_path = 'weights/%s_places365.pt' % arch
file_name = 'categories_places365.txt'
img_name = "imgs/12.jpg"
device = torch.device("cuda:2")

# https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/8
model = models.__dict__[arch](num_classes=365)

# checkpoint = torch.load(model_file)
# model = nn.DataParallel(model)
model.load_state_dict(torch.load(save_pt_path))
model.to(device)
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
my_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)
# print(classes)

img = Image.open(img_name)
tensor = my_transforms(img).unsqueeze(0).to(device)
outputs = model.forward(tensor)
_, y_hat = outputs.max(1)

predicted_idx = y_hat.item()

print(classes[predicted_idx])

percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

print(classes[predicted_idx], percentage[y_hat[0]].item())

_, indices = torch.sort(outputs, descending=True)
print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])