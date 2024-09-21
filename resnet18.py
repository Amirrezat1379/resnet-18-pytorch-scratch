import numpy as np
import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Setting RESNET-18 device as {device}")

device = torch.device(device)

class ConvPoolAc(nn.Module):
    def __init__(self, chanIn, chanOut, kernel=3, stride=1, padding=1, p_ceil_mode=False):
        super(ConvPoolAc, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=kernel,
                stride=stride, padding=padding, bias=False),
            nn.MaxPool2d(2, stride=2, ceil_mode=p_ceil_mode), #ksize, stride
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, shortcut = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.shortcut = shortcut
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out
            
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()

    def make_backbone(self, block, layers, exit_place, num_classes = 10):
        self.inplanes = 64
        back_bone_layers = []
        i = 0
        self.convMax1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 3, stride = 2, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1))
        i += 1
        if i in exit_place:
            self.backbone.append(self.convMax1)
        back_bone_layers.append(self._make_layer(block, 64, layers[0], stride = 1))
        i += 1
        if i in exit_place:
            self.backbone.append(nn.Sequential(*back_bone_layers))
            back_bone_layers = []
        back_bone_layers.append(self._make_layer(block, 128, layers[1], stride = 2))
        i += 1
        if i in exit_place:
            self.backbone.append(nn.Sequential(*back_bone_layers))
            back_bone_layers = []
        back_bone_layers.append(self._make_layer(block, 256, layers[2], stride = 2))
        i += 1
        if i in exit_place:
            self.backbone.append(nn.Sequential(*back_bone_layers))
            back_bone_layers = []
        back_bone_layers.append(self._make_layer(block, 512, layers[3], stride = 2))
        back_bone_layers.append(nn.AvgPool2d(7, stride=1))
        i += 1
        if i in exit_place:
            self.backbone.append(nn.Sequential(*back_bone_layers))
            back_bone_layers = []
        # self.fc = nn.Linear(512, num_classes)
        self.make_exits(exit_place=exit_place)

    def make_exits(self, exit_place):
        early_exits = nn.ModuleList()
        early_exits.append(nn.Sequential(
            ConvPoolAc(chanIn=64, chanOut=5, kernel=3, stride=1, padding=1), #, p_ceil_mode=True),
            nn.Flatten(),
            nn.Linear(80,10)
        ))
        early_exits.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2), #ksize, stride
            nn.ReLU(True),
            ConvPoolAc(64, 16, kernel=3, stride=1, padding=1), #, p_ceil_mode=True),
            nn.Flatten(),
            nn.Linear(64,10)
        ))
        early_exits.append(nn.Sequential(
            ConvPoolAc(128, 32, kernel=3, stride=1, padding=1), #, p_ceil_mode=True),
            nn.Flatten(),
            nn.Linear(128,10)
        ))
        early_exits.append(nn.Sequential(
            ConvPoolAc(256, 64, kernel=3, stride=1, padding=1), #, p_ceil_mode=True),
            nn.Flatten(),
            nn.Linear(64,10)
        ))
        # print(early_exits[1])
        for i in exit_place:
            self.exits.append(early_exits[i - 1])
        self.exits.append(nn.Sequential(
            nn.Linear(512,10)
        ))

        
    def _make_layer(self, block, planes, blocks, stride=1):
        shortcut = None
        if stride != 1 or self.inplanes != planes:
            
            shortcut = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, shortcut))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool(x)
        # x = self.layer0(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        result = []
        for back_bone, current_exit in zip(self.backbone, self.exits):
            # print("salam")
            # print(x.shape)
            back_bone = back_bone.to(device)
            current_exit = current_exit.to(device)
            x = back_bone(x)
            # print(x.size())
            if x.size() == torch.Size([512, 512, 1, 1]):
                x = x.view(x.size(0), -1)
            q = current_exit(x)
            result.append(q)
        return result

        # return x
    
def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False,
                data_model="mnist"):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    # transform = transforms.Compose([
    #         # transforms.Resize((224,224)),
    #         transforms.ToTensor(),
    #         # transforms.Lambda(lambda x: x.repeat(3,1,1)),
    #         # normalize,
    # ])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    if test:
        if data_model == "cifar10":
            dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
            )
        elif data_model == "mnist":
            dataset = datasets.MNIST(
            root=data_dir, train=False,
            download=True, transform=transform,
            )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    if data_model == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform,
        )
    elif data_model == "mnist":
        train_dataset = datasets.MNIST(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

        valid_dataset = datasets.MNIST(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

    num_train = len(train_dataset)
    print(num_train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)