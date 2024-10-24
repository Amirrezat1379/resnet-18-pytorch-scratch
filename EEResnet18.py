import numpy as np
import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class ConvPoolAc(nn.Module):
    def __init__(self, chanIn, chanOut, kernel=3, stride=1, padding=1, p_ceil_mode=False, pool_pad=0):
        super(ConvPoolAc, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=kernel,
                stride=stride, padding=padding, bias=False),  
            nn.MaxPool2d(2, stride=2, ceil_mode=p_ceil_mode, padding=pool_pad), #ksize, stride
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
    
class OutPutBlock(nn.Module):
    def __init__(self, in_features=720, out_features=10):
        super(OutPutBlock, self).__init__()
        self.confidence = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        conf = self.confidence(x)
        pred = self.classifier(x)
        return pred
            
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.fast_inference_mode = False
        self.exit_threshold = 0.5

    def make_backbone(self, block, layers, exit_place, num_classes = 10):
        self.inplanes = 64
        back_bone_layers = []
        i = 0
        back_bone_layers.append(nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)))
        i += 1
        if i in exit_place:
            self.backbone.append(back_bone_layers[0])
            back_bone_layers = []
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
        # back_bone_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.backbone.append(nn.Sequential(*back_bone_layers))
        # self.backbone.append(*back_bone_layers)
        back_bone_layers = []
        # self.fc = nn.Linear(512, num_classes)
        self.make_exits(exit_place=exit_place, num_classes=num_classes)
        # torch.set_printoptions(threshold=10_000)
        # print(self.backbone)

    def make_exits(self, exit_place, num_classes):
        early_exits = nn.ModuleList()
        early_exits.append(nn.ModuleList([nn.Sequential(
            ConvPoolAc(chanIn=64, chanOut=32, kernel=3, stride=1, padding=0), #, p_ceil_mode=True),
            ConvPoolAc(chanIn=32, chanOut=16, kernel=3, stride=1, padding=0)), #, p_ceil_mode=True)
            nn.Sequential(nn.Linear(2304, 512),
                          OutPutBlock(512, num_classes))]
        ))
        early_exits.append(nn.ModuleList([nn.Sequential(
            ConvPoolAc(chanIn=64, chanOut=32, kernel=3, stride=1, padding=0), #, p_ceil_mode=True),
            ConvPoolAc(chanIn=32, chanOut=16, kernel=3, stride=1, padding=0)), #, p_ceil_mode=True)
            nn.Sequential(nn.Linear(2304, 512),
                          OutPutBlock(512, num_classes))]
        ))
        early_exits.append(nn.ModuleList([nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=1),
            ConvPoolAc(128, 32, kernel=3, stride=1, padding=1, pool_pad=1)), #, p_ceil_mode=True),
            nn.Sequential(
            nn.Linear(2048,512),
            OutPutBlock(512, num_classes))]
        ))
        early_exits.append(nn.ModuleList([nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=1),
            ConvPoolAc(256, 64, kernel=3, stride=1, padding=1)), #, p_ceil_mode=True),
            nn.Sequential(
            nn.Linear(1024,512),
            OutPutBlock(512, num_classes))]
        ))
        # print(early_exits[1])
        if len(early_exits) > 0:
            for i in exit_place:
                self.exits.append(early_exits[i - 1])
        self.exits.append(nn.ModuleList([nn.Sequential(
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=2, stride=2)),(nn.Sequential(
            OutPutBlock(512, num_classes)
        ))]))

        
    def _make_layer(self, block, planes, blocks, stride=1):
        shortcut = None
        if stride != 1 or self.inplanes != planes:
            
            shortcut = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        # print(f'plains = {planes}, inplanes = {self.inplanes}')
        layers = []
        layers.append(block(self.inplanes, planes, stride, shortcut))
        self.inplanes = planes
        for i in range(1, blocks):
            # print(f"i = {i}")
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def exit_criterion_top1(self, x):
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            top1 = torch.max(pk) #x)
            return top1 > self.exit_threshold

    def _forward_training(self, x):
        res = []
        for backbone, current_early_exit in zip(self.backbone, self.exits):
            x = backbone(x)
            y = current_early_exit[0](x)
            y = y.view(y.size(0), -1)
            res.append(current_early_exit[1](y))
        return res

    def forward(self, x):
        #std forward function - add var to distinguish be test and inf

        if self.fast_inference_mode:
            for backbone, current_early_exit in zip(self.backbone, self.exits):
                x = backbone(x)
                x = current_early_exit[0](x)
                x = x.view(x.size(0), -1)
                res = current_early_exit[1](x)
                if self.exit_criterion_top1(res):
                    return res, 1
            return res, 0
        else:
            return self._forward_training(x)
    
    def set_inference_parameters(self, mode=True, thresh=0.5):
        self.fast_inference_mode = mode
        self.exit_threshold = thresh
    
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
    data_model = data_model.lower()
    # define transforms
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3,1,1)),
            normalize,
    ])
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))
    #                                 ])

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
        elif data_model == "cifar100":
            dataset = datasets.CIFAR100(
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
    elif data_model == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

        valid_dataset = datasets.CIFAR100(
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