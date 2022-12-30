# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


from networks.vit_enc import EncoderTransformer
class VITEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self):
        super(VITEncoder, self).__init__()

        self.enc = EncoderTransformer()

    def forward(self, input_image):
        feats = self.enc(input_image)
        # print(feats[0].size(), feats[1].size(), feats[2].size(), feats[3].size())
        # exit()
        return feats 

from networks.vit2 import VIT2Encoder as enc2

class VIT2Encoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self):
        super(VIT2Encoder, self).__init__()

        self.enc = enc2()

    def forward(self, input_image):
        feats = self.enc(input_image)
        # print(feats[0].size(), feats[1].size(), feats[2].size(), feats[3].size())
        # exit()
        return feats


from networks.vit2_light import VIT2Encoder_light as enc2_light

class VIT2Encoder_light(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self):
        super(VIT2Encoder_light, self).__init__()

        self.enc = enc2_light()

    def forward(self, input_image):
        feats = self.enc(input_image)
        # print(feats[0].size(), feats[1].size(), feats[2].size(), feats[3].size())
        # exit()
        return feats
##################################################################
class ResnetEncoder_mod(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers = 18, pretrained = True, num_input_images=1):
        super(ResnetEncoder_mod, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        x1 = self.encoder.relu(x)
        x2 = self.encoder.layer1(self.encoder.maxpool(x1))
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)
        # self.features.append(self.encoder.relu(x))
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        # self.features.append(self.encoder.layer2(self.features[-1]))
        # self.features.append(self.encoder.layer3(self.features[-1]))
        # self.features.append(self.encoder.layer4(self.features[-1]))
        print(x.size(),x1.size(), x2.size(), x3.size(), x4.size(), x5.size())

        return [x1,x2,x3,x4,x5]
class VIT2Encoder_light_resnet(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self):
        super(VIT2Encoder_light_resnet, self).__init__()

        self.enc = enc2_light()
        self.enc_resnet = ResnetEncoder_mod()
    def forward(self, input_image):
        feats_tx = self.enc(input_image)
        feats_res = self.enc_resnet(input_image)
        print(feats_tx[0].size(), feats_tx[1].size(), feats_tx[2].size(), feats_tx[3].size())
        exit()
        feats = self.fuse(feats_tx,feats_res)
        return feats


#################################################
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size = 3, reduction = 1, bias = True, act = nn.ReLU()):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size,1,1, bias=bias))
        modules_body.append(act)
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size,1,1, bias=bias))

        # self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        # res = self.CA(res)
        res += x
        return res

class Enc_Fuse(nn.Module):
    def __init__(self):
        super(Enc_Fuse, self).__init__()
        # self.conv_res4 = nn.Conv2d(32,32,1)
        # self.atr_4 = Atr_Conv(32)
        # self.cab_4 = CAB(32)

        # self.conv_res3 = nn.Conv2d(32,32,1)
        # self.atr_3 = Atr_Conv(32)
        # self.cab_3 = CAB(32)

        # self.conv_res2 = nn.Conv2d(32,32,1)
        # self.atr_2 = Atr_Conv(32)
        # self.cab_2 = CAB(32)

        # self.conv_res1 = nn.Conv2d(32,32,1)
        # self.atr_1 = Atr_Conv(32)
        # self.cab_1 = CAB(32)

        # self.cab_4tx = CAB(32)
        # self.cab_3tx = CAB(32)
        # self.cab_2tx = CAB(32)
        # self.cab_1tx = CAB(32)
        
        # self.fuse4 = nn.Sequential(nn.Conv2d(32*2,32,1), nn.Conv2d(32,32,3,1,1))
        # self.fuse3 = nn.Sequential(nn.Conv2d(32*2,32,1), nn.Conv2d(32,32,3,1,1))
        # self.fuse2 = nn.Sequential(nn.Conv2d(32*2,32,1), nn.Conv2d(32,32,3,1,1))
        # self.fuse1 = nn.Sequential(nn.Conv2d(32*2,32,1), nn.Conv2d(32,32,3,1,1))
    def forward(self, feats_tx, feats_res):
        res4, res3, res2, res1 = feats_res[0], feats_res[2], feats_res[2], feats_res[3] 
        tx_4, tx_3, tx_2, tx_1 = feats_tx[0], feats_tx[2], feats_tx[2], feats_tx[3]
        
        res4 = self.cab_4(self.atr_4(self.conv_res4(res4)))
        tx_4 = self.cab_4tx(tx_4)
        feat_4 = self.fuse4(torch.cat((res4,tx_4),1))

        res3 = self.cab_3(self.atr_3(self.conv_res3(res3)))
        tx_3 = self.cab_3tx(tx_3)
        feat_3 = self.fuse3(torch.cat((res3,tx_3),1))

        res2 = self.cab_2(self.atr_2(self.conv_res2(res2)))
        tx_2 = self.cab_2tx(tx_2)
        feat_2 = self.fuse2(torch.cat((res2,tx_2),1))

        res1 = self.cab_1(self.atr_1(self.conv_res1(res1)))
        tx_1 = self.cab_1tx(tx_1)
        feat_1 = self.fuse1(torch.cat((res2,tx_2),1))


        return [feat_4, feat_3, feat_2, feat_1]

class Atr_Conv(nn.Module):
    def __init__(self, dim):
        super(Atr_Conv, self).__init__()
        div_factor = 4
        self.conv1 = nn.Conv2d(dim,dim//div_factor,kernel_size = 3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(dim,dim//div_factor,kernel_size = 3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(dim,dim//div_factor,kernel_size = 3, stride=1, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(dim,dim//div_factor,kernel_size = 3, stride=1, padding=6, dilation=6)

        self.conv_fuse = nn.Conv2d(4*(dim//div_factor),dim,1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x_out = torch.cat((x1,x2,x3,x4),1)
        return self.conv_fuse(x_out) + x