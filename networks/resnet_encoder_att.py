from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url


class ResNetMultiImageInput(models.ResNet):
    
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
from .networks_diffnet.test_hr_encoder import HighResolutionNet as enc3_diffnet
from .networks_diffnet.hrnet_config import MODEL_CONFIGS
    

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
        # x5 = self.encoder.layer4(x4)
        # self.features.append(self.encoder.relu(x))
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        # self.features.append(self.encoder.layer2(self.features[-1]))
        # self.features.append(self.encoder.layer3(self.features[-1]))
        # self.features.append(self.encoder.layer4(self.features[-1]))
        # print('RES:',x.size(),x1.size(), x2.size(), x3.size(), x4.size())

        return [x4,x3,x2,x1]


class VIT2Encoder_light_resnet(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self):
        super(VIT2Encoder_light_resnet, self).__init__()

        self.enc = enc2_light()
        self.enc_resnet = ResnetEncoder_mod()
        self.fuse = Enc_Fuse()
    def forward(self, input_image):
        feats_tx = self.enc(input_image) #l, x3, x2, x1
        feats_res = self.enc_resnet(input_image) #latent, enc3, enc2, enc1
        # print('TX:',feats_tx[0].size(), feats_tx[1].size(), feats_tx[2].size(), feats_tx[3].size())
        # exit()
        feats = self.fuse(feats_tx,feats_res)
        # feats = feats_tx
        # print('feeding to decoder')
        return feats

class encoder_VIT2Encoder_light_resnet_diffnet(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self):
        super(encoder_VIT2Encoder_light_resnet_diffnet, self).__init__()
        self.enc = enc2_light()
        self.enc_resnet = ResnetEncoder_mod()
        model_url = 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w'
        loaded_state_dict = load_state_dict_from_url(model_url, progress = True)
        self.enc_diffnet = enc3_diffnet(MODEL_CONFIGS['hrnet18'])
        self.enc_diffnet.load_state_dict({k: v for k,v in loaded_state_dict.items() if k in self.enc_diffnet.state_dict()}, strict = False)
        self.fuse = Enc_Fuse_diffnet()

    def forward(self, input_image):
        ############
        r = np.random.randint(20000, size=1)
        # import os
        # import matplotlib.pyplot as plt
        # path1 = './vis_masks/Input/'
        # os.makedirs(path1, exist_ok=True)
        # image_name = path1 + '/inp_' + str(r) + '.png'
        # plt.imshow(input_image[0,:,:,:].permute(1,2,0).cpu().numpy())
        # plt.savefig(image_name)
        # plt.clf()
        ############

        feats_tx = self.enc(input_image) #l, x3, x2, x1
        feats_res = self.enc_resnet(input_image) #latent, enc3, enc2, enc1
        feats_diffnet = self.enc_diffnet(input_image)
        # print('TX:',feats_tx[0].size(), feats_tx[1].size(), feats_tx[2].size(), feats_tx[3].size())
        # exit()
        feats = self.fuse(feats_tx, feats_res, feats_diffnet, r)
        # feats = feats_tx
        # print('feeding to decoder')
        return feats

################################################
from networks.vit2_light_resfuse import VIT2Encoder_light_resfuse as enc2_light_resfuse

class VIT2Encoder_light_resnet_fuse(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self):
        super(VIT2Encoder_light_resnet_fuse, self).__init__()

        self.enc = enc2_light_resfuse()
        self.enc_resnet = ResnetEncoder_mod()
        self.fuse = Enc_Fuse()
    def forward(self, input_image):
        feats_res = self.enc_resnet(input_image)
        feats_tx = self.enc(input_image, feats_res) #l, x3, x2, x1
         #latent, enc3, enc2, enc1
        # print('TX:',feats_tx[0].size(), feats_tx[1].size(), feats_tx[2].size(), feats_tx[3].size())
        # exit()
        feats = self.fuse(feats_tx,feats_res)
        # feats = feats_tx
        # print('feeding to decoder')
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

        modules_body2 = []
        modules_body2.append(nn.Conv2d(n_feat, n_feat, kernel_size,1,1, bias=bias))
        modules_body2.append(act)
        modules_body2.append(nn.Conv2d(n_feat, n_feat, kernel_size,1,1, bias=bias))

        # self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body2 = nn.Sequential(*modules_body2)
        self.norm = nn.InstanceNorm2d(n_feat, affine = True)
    def forward(self, x):
        res = self.body(x)
        res = self.norm(res)
        res = self.body2(res)
        # res = self.CA(res)
        # res += x
        return res

class Enc_Fuse(nn.Module):
    def __init__(self):
        super(Enc_Fuse, self).__init__()

        tx_dim = 48

        self.conv_res4 = nn.Conv2d(256,tx_dim*8,1, bias = True)
        self.atr_4 = Atr_Conv(tx_dim*8)
        self.cab_4 = CAB(tx_dim*8)

        self.conv_res3 = nn.Conv2d(128,tx_dim*4,1, bias = True)
        self.atr_3 = Atr_Conv(tx_dim*4)
        self.cab_3 = CAB(tx_dim*4)

        self.conv_res2 = nn.Conv2d(64,tx_dim*2,1, bias = True)
        self.atr_2 = Atr_Conv(tx_dim*2)
        self.cab_2 = CAB(tx_dim*2)

        self.conv_res1 = nn.Conv2d(64,tx_dim,1, bias = True)
        self.atr_1 = Atr_Conv(tx_dim)
        self.cab_1 = CAB(tx_dim)

        
        self.cab_4tx = CAB(tx_dim * 8)
        self.cab_3tx = CAB(tx_dim * 4)
        self.cab_2tx = CAB(tx_dim * 2)
        self.cab_1tx = CAB(tx_dim)
        
        self.fuse4 = nn.Sequential(nn.Conv2d(tx_dim*16,tx_dim*8,1), nn.Conv2d(tx_dim*8,tx_dim*8,3,1,1))
        self.fuse3 = nn.Sequential(nn.Conv2d(tx_dim*8,tx_dim*4,1), nn.Conv2d(tx_dim*4,tx_dim*4,3,1,1))
        self.fuse2 = nn.Sequential(nn.Conv2d(tx_dim*4,tx_dim*2,1), nn.Conv2d(tx_dim*2,tx_dim*2,3,1,1))
        self.fuse1 = nn.Sequential(nn.Conv2d(tx_dim*2,tx_dim,1), nn.Conv2d(tx_dim,tx_dim,3,1,1))

        
    def forward(self, feats_tx, feats_res):
        res4, res3, res2, res1 = feats_res[0], feats_res[1], feats_res[2], feats_res[3] 
        tx_4, tx_3, tx_2, tx_1 = feats_tx[0], feats_tx[1], feats_tx[2], feats_tx[3]
        
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

        

        feat_1 = self.fuse1(torch.cat((res1,tx_1),1))
        # fusing
        # print('Inside Enc Fuse')
        # return [feat_4, feat_3, feat_2, feat_1]
        return [feat_4 + tx_4, feat_3 + tx_3, feat_2 + tx_2, feat_1 + tx_1]

class Enc_Fuse_diffnet(nn.Module):
    def __init__(self):
        super(Enc_Fuse_diffnet, self).__init__()

        tx_dim = 48

        self.conv_res4 = nn.Conv2d(256,tx_dim*8,1, bias = True)
        self.atr_4 = Atr_Conv(tx_dim*8)
        self.cab_4 = CAB(tx_dim*8)

        self.conv_res3 = nn.Conv2d(128,tx_dim*4,1, bias = True)
        self.atr_3 = Atr_Conv(tx_dim*4)
        self.cab_3 = CAB(tx_dim*4)

        self.conv_res2 = nn.Conv2d(64,tx_dim*2,1, bias = True)
        self.atr_2 = Atr_Conv(tx_dim*2)
        self.cab_2 = CAB(tx_dim*2)

        self.conv_res1 = nn.Conv2d(64,tx_dim,1, bias = True)
        self.atr_1 = Atr_Conv(tx_dim)
        self.cab_1 = CAB(tx_dim)

        
        self.cab_4tx = CAB(tx_dim * 8)
        self.cab_3tx = CAB(tx_dim * 4)
        self.cab_2tx = CAB(tx_dim * 2)
        self.cab_1tx = CAB(tx_dim)

        self.cab_4tx_ = CAB(tx_dim * 8)
        self.cab_3tx_ = CAB(tx_dim * 4)
        self.cab_2tx_ = CAB(tx_dim * 2)
        self.cab_1tx_ = CAB(tx_dim)
        
        self.fuse4 = nn.Sequential(nn.Conv2d(tx_dim*16,tx_dim*8,1), nn.Conv2d(tx_dim*8,tx_dim*8,3,1,1))
        self.fuse3 = nn.Sequential(nn.Conv2d(tx_dim*8,tx_dim*4,1), nn.Conv2d(tx_dim*4,tx_dim*4,3,1,1))
        self.fuse2 = nn.Sequential(nn.Conv2d(tx_dim*4,tx_dim*2,1), nn.Conv2d(tx_dim*2,tx_dim*2,3,1,1))
        self.fuse1 = nn.Sequential(nn.Conv2d(tx_dim*2,tx_dim,1), nn.Conv2d(tx_dim,tx_dim,3,1,1))

        self.att_4_res = nn.Sequential(nn.Conv2d(tx_dim * 8, tx_dim,1), nn.Conv2d(tx_dim, tx_dim,3,1,1), nn.Conv2d(tx_dim, tx_dim*8,1), nn.Sigmoid())
        self.att_3_res = nn.Sequential(nn.Conv2d(tx_dim * 4, tx_dim,1), nn.Conv2d(tx_dim, tx_dim,3,1,1), nn.Conv2d(tx_dim, tx_dim*4,1), nn.Sigmoid())
        self.att_2_res = nn.Sequential(nn.Conv2d(tx_dim * 2, tx_dim,1), nn.Conv2d(tx_dim, tx_dim,3,1,1), nn.Conv2d(tx_dim, tx_dim*2,1), nn.Sigmoid())
        self.att_1_res = nn.Sequential(nn.Conv2d(tx_dim * 1, tx_dim,1), nn.Conv2d(tx_dim, tx_dim,3,1,1), nn.Conv2d(tx_dim, tx_dim*1,1), nn.Sigmoid())

        self.att_4_diff = nn.Sequential(nn.Conv2d(tx_dim * 8, tx_dim,1), nn.Conv2d(tx_dim, tx_dim,3,1,1), nn.Conv2d(tx_dim, tx_dim*8,1), nn.Sigmoid())
        self.att_3_diff = nn.Sequential(nn.Conv2d(tx_dim * 4, tx_dim,1), nn.Conv2d(tx_dim, tx_dim,3,1,1), nn.Conv2d(tx_dim, tx_dim*4,1), nn.Sigmoid())
        self.att_2_diff = nn.Sequential(nn.Conv2d(tx_dim * 2, tx_dim,1), nn.Conv2d(tx_dim, tx_dim,3,1,1), nn.Conv2d(tx_dim, tx_dim*2,1), nn.Sigmoid())
        self.att_1_diff = nn.Sequential(nn.Conv2d(tx_dim * 1, tx_dim,1), nn.Conv2d(tx_dim, tx_dim,3,1,1), nn.Conv2d(tx_dim, tx_dim*1,1), nn.Sigmoid())

    def forward(self, feats_tx, feats_res, feats_diffent_enc, r):
        res4, res3, res2, res1 = feats_res[0], feats_res[1], feats_res[2], feats_res[3] 
        tx_4, tx_3, tx_2, tx_1 = feats_tx[0], feats_tx[1], feats_tx[2], feats_tx[3]
        diff4, diff3, diff2, diff1 = feats_diffent_enc[0], feats_diffent_enc[1], feats_diffent_enc[2], feats_diffent_enc[3]  

        res4 = self.cab_4(self.atr_4(self.conv_res4(res4)))
        tx_4 = self.cab_4tx(tx_4)

        attmap_4_res = self.att_4_res(res4)
        res4 = res4 * attmap_4_res

        feat_4 = self.fuse4(torch.cat((res4,tx_4),1))


        res3 = self.cab_3(self.atr_3(self.conv_res3(res3)))
        tx_3 = self.cab_3tx(tx_3)

        attmap_3_res = self.att_3_res(res3)
        res3 = res3 * attmap_3_res

        feat_3 = self.fuse3(torch.cat((res3,tx_3),1))

        res2 = self.cab_2(self.atr_2(self.conv_res2(res2)))
        tx_2 = self.cab_2tx(tx_2)

        attmap_2_res = self.att_2_res(res2)
        res2 = res2 * attmap_2_res

        feat_2 = self.fuse2(torch.cat((res2,tx_2),1))

        res1 = self.cab_1(self.atr_1(self.conv_res1(res1)))
        tx_1 = self.cab_1tx(tx_1)

        attmap_1_res = self.att_1_res(res1)
        res1 = res1 * attmap_1_res

        feat_1 = self.fuse1(torch.cat((res1,tx_1),1))
        
        
        diff_4 = self.cab_4tx_(diff4)
        diff_3 = self.cab_3tx_(diff3)
        diff_2 = self.cab_2tx_(diff2)
        diff_1 = self.cab_1tx_(diff1)


        attmap_4_diff = self.att_4_diff(diff_4)
        attmap_3_diff = self.att_3_diff(diff_3)
        attmap_2_diff = self.att_2_diff(diff_2)
        attmap_1_diff = self.att_1_diff(diff_1)

        diff_4 = diff_4 * attmap_4_diff
        diff_3 = diff_3 * attmap_3_diff
        diff_2 = diff_2 * attmap_2_diff
        diff_1 = diff_1 * attmap_1_diff

        fused_feat_4 = self.fuse4(torch.cat((diff_4, feat_4),1))
        fused_feat_3 = self.fuse3(torch.cat((diff_3, feat_3),1))
        fused_feat_2 = self.fuse2(torch.cat((diff_2, feat_2),1))
        fused_feat_1 = self.fuse1(torch.cat((diff_1, feat_1),1)) 
        # if True: #Save attention maps
        #     import os
        #     import matplotlib.pyplot as plt
            
        #     path1 = './vis_masks/LRL/' + str(r)
        #     path2 = './vis_masks/HRL/' + str(r)
        #     os.makedirs(path1, exist_ok=True)
        #     os.makedirs(path2, exist_ok=True)
            
        #     # LRL
        #     for i in range(attmap_1_res.size(1)):
        #         image_name = path1 + '/res1_' + str(i) + '.png'
        #         plt.imshow(attmap_1_res[0,i,:,:].cpu().numpy())
        #         plt.savefig(image_name)
        #         plt.clf()
        #     for i in range(attmap_2_res.size(1)):
        #         image_name = path1 + '/res2_' + str(i) + '.png'
        #         plt.imshow(attmap_2_res[0,i,:,:].cpu().numpy())
        #         plt.savefig(image_name)
        #         plt.clf()
        #     for i in range(attmap_3_res.size(1)):
        #         image_name = path1 + '/res3_' + str(i) + '.png'
        #         plt.imshow(attmap_3_res[0,i,:,:].cpu().numpy())
        #         plt.savefig(image_name)
        #         plt.clf()
        #     #HRL
        #     for i in range(attmap_1_diff.size(1)):
        #         image_name = path2 + '/diff1_' + str(i) + '.png'
        #         plt.imshow(attmap_1_diff[0,i,:,:].cpu().numpy())
        #         plt.savefig(image_name)
        #         plt.clf()
        #     for i in range(attmap_2_diff.size(1)):
        #         image_name = path2 + '/diff2_' + str(i) + '.png'
        #         plt.imshow(attmap_2_diff[0,i,:,:].cpu().numpy())
        #         plt.savefig(image_name)
        #         plt.clf()
        #     for i in range(attmap_3_diff.size(1)):
        #         image_name = path2 + '/diff3_' + str(i) + '.png'
        #         plt.imshow(attmap_3_diff[0,i,:,:].cpu().numpy())
        #         plt.savefig(image_name)
        #         plt.clf()
        # # exit()
        # print('ONE DONE')
        # fusing
        # print('Inside Enc Fuse')
        # return [feat_4, feat_3, feat_2, feat_1]
        # return [feat_4 + tx_4, feat_3 + tx_3, feat_2 + tx_2, feat_1 + tx_1]
        return [fused_feat_4 + feat_4, fused_feat_3 + feat_3, fused_feat_2 + feat_2, fused_feat_1 + feat_1]

class Atr_Conv(nn.Module):
    def __init__(self, dim):
        super(Atr_Conv, self).__init__()
        div_factor = 2
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
        # print('ATR CONV')
        return self.conv_fuse(x_out) + x
        # return x