import torchvision
import torchvision.transforms as transforms
from utils import *


def img_save(tensor,filename):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(filename,quality=95)

transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
transform_cutout = transforms.Compose([
            transforms.ToTensor(),
            Cutout()
        ])

dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
cutoutset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_cutout)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
cutoutloader = torch.utils.data.DataLoader(cutoutset, batch_size=3, shuffle=False, num_workers=2)

for images, labels in dataloader:
    images_mixup, labels_a, labels_b, lam = mixup_data(images, labels)
    images_cutmix, labels_a, labels_b, lam = cutmix((images, labels))
    for i in range(3):
        filename = 'Sample_' + str(i) + '.png'
        img_save(images[i], filename)

        filename_cutmix = 'cutmix_' + str(i) + '.png'
        img_save(images_cutmix[i], filename_cutmix)

        filename_mixup = 'mixup_' + str(i) + '.png'
        img_save(images_mixup[i], filename_mixup)
    break

for images, labels in cutoutloader:
    for i in range(3):
        filename = 'cotout_' + str(i) + '.png'
        img_save(images[i], filename)
    break