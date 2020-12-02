import torch.nn as nn
import torch
import math
import numpy as np
from torch.nn import functional as F

from al_kitti.presets import segmen_preset as segmen_preset
'''from al_kitti.presets.rat_spn import RatSpn, RatSpnConfig


from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig'''

# inspiration: https://github.com/CSAILVision/semantic-segmentation-pytorch
# also original pytorch implementation


class Upsample_reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        '''batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)'''
        # input = input.view(self.shape)
        batch = int(input.shape[0]/self.shape[1]/self.shape[2])
        input = torch.reshape(input, (batch, self.shape[1], self.shape[2], self.shape[0]))
        out = input.permute(0, 3, 1, 2)
        return out


class Upsample(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        '''batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)'''
        out = input.permute(0, 3, 1, 2)
        return out


class View_reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = input.permute(0, 2, 3, 1)
        out = torch.reshape(out, (out.shape[0]*out.shape[1]*out.shape[2], self.shape[-1]))

        return out


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = input.permute(0, 2, 3, 1)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        x += residual
        x = self.relu(x)

        return x


class ResNet_Dropout(nn.Module):
    def __init__(self, n_class=10, p_dropout=0.1):
        super(ResNet_Dropout, self).__init__()
        layers_50 = [3, 4, 6, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # (in_feature, out_feature)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.bottleneck_layer(Bottleneck, 64, layers_50[0])
        self.layer2 = self.bottleneck_layer(Bottleneck, 128, layers_50[1], stride=2)
        self.layer3 = self.bottleneck_layer(Bottleneck, 256, layers_50[2], stride=2)
        self.layer4 = self.bottleneck_layer(Bottleneck, 512, layers_50[3], stride=2)

        self.classifier1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.classifier2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1)
        self.classifier3 = nn.ReLU(inplace=True)
        self.classifier4 = nn.Dropout(p=p_dropout, inplace=False)
        self.classifier5 = nn.Conv2d(512, n_class, kernel_size=1, stride=1, bias=False)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # define a block in resnet
    def bottleneck_layer(self, block, feature, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != feature * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, feature * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(feature * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, feature, stride, downsample))
        self.inplanes = feature * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, feature))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        x = self.classifier4(x)
        x = self.classifier5(x)

        return x


def dropout_tester(model, test_dataset, device, index):

    # get example data
    # data, target = test_dataset.__getitem__(torch.randint(test_dataset.__len__(), size=(1,)))
    data, target = test_dataset.__getitem__(index)
    data, target = data.to(device), target.to(device)
    output_list = []

    # make eval model
    model.eval()
    enable_dropout(model, 0.1)
    model.to(device)

    for i_test in range(5):
        output = model(data.unsqueeze(dim=0))['out']
        output_list.append(output)
    np.all((output_list[0] == output_list[4]).detach().cpu().numpy())
    np.sum((output_list[2].argmax(dim=1) == output_list[4].argmax(dim=1)).detach().cpu().numpy()) / (256 * 512)
    segmen_preset.show_pred_result_single(model, index, test_dataset)


def restore_dropout(fcn_model, orig_prob_list):
    """fcn_model_0 = fcn_model.to('cuda:0')
    del fcn_model"""
    for each_module in fcn_model.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.eval()
            each_module.p = orig_prob_list[0]
            orig_prob_list.pop(0)

    """fcn_model_0 = fcn_model_0.cuda()
    fcn_model = nn.DataParallel(fcn_model_0)"""
    return fcn_model


# apply dropout on line 144/145/146, but later is better
# To apply dropout even during .eval()
def enable_dropout(fcn_model, p):
    prob_list = []
    """fcn_model_0 = fcn_model.to('cuda:0')
    del fcn_model"""
    for each_module in fcn_model.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
            prob_list.append(each_module.p)
            each_module.p = p
    """fcn_model_0 = fcn_model_0.cuda()
    fcn_model = nn.DataParallel(fcn_model_0)
    # print(fcn_model)"""
    return fcn_model, prob_list
