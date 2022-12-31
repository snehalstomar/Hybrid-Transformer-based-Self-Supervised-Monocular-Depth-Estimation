from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        # self.convs = OrderedDict()
        # for i in range(4, -1, -1):
        #     # upconv_0
        #     num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
        #     num_ch_out = self.num_ch_dec[i]
        #     self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

        #     # upconv_1
        #     num_ch_in = self.num_ch_dec[i]
        #     if self.use_skips and i > 0:
        #         num_ch_in += self.num_ch_enc[i - 1]
        #     num_ch_out = self.num_ch_dec[i]
        #     self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.convs_up_4_0 = ConvBlock(num_ch_enc[-1], 256)
        self.convs_up_4_1 = ConvBlock(num_ch_enc[-1], 256)

        self.convs_up_3_0 = ConvBlock(256, 128)
        self.convs_up_3_1 = ConvBlock(256, 128)

        self.convs_up_2_0 = ConvBlock(128, 64)
        self.convs_up_2_1 = ConvBlock(128, 64)

        self.convs_up_1_0 = ConvBlock(64, 32)
        self.convs_up_1_1 = ConvBlock(64+32, 32)

        self.convs_up_0_0 = ConvBlock(32, 16)
        self.convs_up_0_1 = ConvBlock(16, 16)

        # for s in self.scales:
        #     self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.convs_disp_0 = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        self.convs_disp_1 = Conv3x3(self.num_ch_dec[1], self.num_output_channels)
        self.convs_disp_2 = Conv3x3(self.num_ch_dec[2], self.num_output_channels)
        self.convs_disp_3 = Conv3x3(self.num_ch_dec[3], self.num_output_channels)
        
        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        # for i,k in self.convs.items():
        #     self.convs[i] = nn.DataParallel(self.convs[i].cuda())
    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1].to('cuda')

        x = self.convs_up_4_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [input_features[4 - 1]]
        x = torch.cat(x, 1)
        x = self.convs_up_4_1(x)


        x = self.convs_up_3_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [input_features[3 - 1]]
        x = torch.cat(x, 1)
        x = self.convs_up_3_1(x)
        self.outputs[("disp", 3)] = self.sigmoid(self.convs_disp_3(x))

        x = self.convs_up_2_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [input_features[2 - 1]]
        x = torch.cat(x, 1)
        x = self.convs_up_2_1(x)
        self.outputs[("disp", 2)] = self.sigmoid(self.convs_disp_2(x))

        x = self.convs_up_1_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [input_features[1 - 1]]
        x = torch.cat(x, 1)
        x = self.convs_up_1_1(x)
        self.outputs[("disp", 1)] = self.sigmoid(self.convs_disp_1(x))

        x = self.convs_up_0_0(x)
        x = [upsample(x)]
        # if self.use_skips:
        #     x += [input_features[1 - 1]]
        x = torch.cat(x, 1)
        x = self.convs_up_0_1(x)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs_disp_0(x))


        # for i in range(4, -1, -1):
        #     x = self.convs[("upconv", i, 0)](x)
        #     x = [upsample(x)]
        #     if self.use_skips and i > 0:
        #         x += [input_features[i - 1]]
        #     x = torch.cat(x, 1)
        #     x = self.convs[("upconv", i, 1)](x)
        #     if i in self.scales:
        #         self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs




from networks.vit_enc import DecoderTransformer, convprojection_base
class VITDecoder(nn.Module):
    def __init__(self):
        super(VITDecoder, self).__init__()
        # self.sigmoid = nn.Sigmoid()
        # self.dec = DecoderTransformer()
        self.conv = convprojection_base()
        # self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, input_features):
        feats = self.conv(input_features)
        # print('LL:',len(feats))
        # print('DEC:',feats.size())
        # exit()
        return feats

from networks.vit2 import VIT2Decoder as dec2
class VIT2Decoder(nn.Module):
    def __init__(self):
        super(VIT2Decoder, self).__init__()
        # self.sigmoid = nn.Sigmoid()
        # self.dec = DecoderTransformer()
        self.conv = dec2()
        # self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, input_features):
        feats = self.conv(input_features)
        # print('LL:',len(feats))
        # print('DEC:',feats.size())
        # exit()
        return feats

from networks.vit2_light import VIT2Decoder_light as dec2_light
class VIT2Decoder_light(nn.Module):
    def __init__(self):
        super(VIT2Decoder_light, self).__init__()
        # self.sigmoid = nn.Sigmoid()
        # self.dec = DecoderTransformer()
        self.conv = dec2_light()
        # self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, input_features):
        feats = self.conv(input_features)
        # print('LL:',len(feats))
        # print('DEC:',feats.size())
        # exit()
        return feats


from networks.vit2_light_resfuse import VIT2Decoder_light as dec2_light_ada
class VIT2Decoder_light_ada(nn.Module):
    def __init__(self):
        super(VIT2Decoder_light_ada, self).__init__()
        # self.sigmoid = nn.Sigmoid()
        # self.dec = DecoderTransformer()
        self.conv = dec2_light_ada()
        # self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, input_features):
        feats = self.conv(input_features)
        # print('LL:',len(feats))
        # print('DEC:',feats.size())
        # exit()
        return feats
