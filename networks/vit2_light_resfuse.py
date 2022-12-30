import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


#############################################
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
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
        res = res + x
        return res

from .pac import *
class AConv(nn.Module):
    def __init__(
        self, n_feat, kernel_size = 3,
        bias=True, bn=False, act=nn.ReLU(False), res_scale=1):

        super(AConv, self).__init__()
        self.actt=act
        self.res_scale = res_scale

        self.filterSize2Channel = 25
        self.ordered_embedding1 = nn.Sequential(            
            nn.Conv2d(n_feat, self.filterSize2Channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(False),
            nn.InstanceNorm2d(self.filterSize2Channel),     
            nn.Conv2d(self.filterSize2Channel, self.filterSize2Channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(False),
            nn.InstanceNorm2d(self.filterSize2Channel),            
            nn.Conv2d(self.filterSize2Channel, self.filterSize2Channel, kernel_size=3, padding=1, bias=True)
        )
        
        self.ada_conv = PacConv2d(n_feat, n_feat, 5,1,2)                            
    def forward(self, x):
        b,ch,h,w = x.size()            
        guide_k = self.ordered_embedding1(x).unsqueeze(1).view(b,1,5,5,h,w)
        res = self.ada_conv(x, None, guide_k) # alternative interface  
        res=self.actt(res)
        res = res + x

        return res

class Depth_Block(nn.Module):
    def __init__(self, n_feat, final = False):
        super(Depth_Block, self).__init__()
        self.conv_init = AConv(n_feat)
        self.block = [CAB(n_feat) for _ in range(3)]

        self.block = nn.Sequential(*self.block)
        self.conv1 = nn.Conv2d(n_feat,n_feat,3,1,1)
        # self.conv1_up = nn.Conv2d(n_feat,n_feat,3,1,1)
        self.conv2 = nn.Sequential( nn.Conv2d(n_feat,n_feat//2,3,1,1), nn.Conv2d(n_feat//2,1,1))
        self.sig = nn.Sigmoid()

        # self.up = nn.Sequential(nn.Conv2d(n_feat//2, n_feat, kernel_size=1), nn.PixelShuffle(2))
        self.up = nn.ConvTranspose2d(n_feat, n_feat, kernel_size = 4, stride=2, padding=1, output_padding=0)

        self.conv_fuse = nn.Sequential(nn.Conv2d(n_feat+1,n_feat,1), nn.Conv2d(n_feat,n_feat,3,1,1))
    def forward(self, x, up = False, ada = False, prvs = None):
        # if ada:
        #     x = self.conv_init(x)
        res = self.block(x)
        if up:
            res = (self.up(self.conv1(res)))
        else:
            res = (self.conv1(res))
        if prvs != None:
            prvs = F.interpolate(prvs, scale_factor = 2)
            res = torch.cat((res,prvs),1)
            res = self.conv_fuse(res)
        return self.sig(self.conv2(res))

class Fuse_res(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, dim):
        super(Fuse_res, self).__init__()

        self.cab = CAB(dim)
        self.norm1= nn.InstanceNorm2d(dim, affine = True)
        self.norm2 = nn.InstanceNorm2d(dim, affine = True)
        self.conv1 = nn.Conv2d(dim*2,dim,1)
        self.conv2 = nn.Conv2d(dim,dim,3,1,1)
        # self.conv3 = nn.Conv2d(dim,dim,3,1,1)

        self.conv_att = nn.Sequential(nn.Conv2d(dim,dim,3,1,1, bias = True), nn.Conv2d(dim,dim,3,1,1, bias = True), nn.Sigmoid())
    def forward(self, tx_feat, res_feat):
        res_feat = self.norm1(self.cab(res_feat))
        att_map = self.conv_att(res_feat)
        res_feat = res_feat * att_map

        fused = torch.cat((tx_feat, res_feat),1)
        fused = self.conv2(self.norm2(self.conv1(fused)))
        return fused
#############################################
class VIT2Encoder_light_resfuse(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [3,4,4,4],
        # num_blocks = [2,2,3,4],  
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias'
    ):
        super(VIT2Encoder_light_resfuse, self).__init__()

        self.patch_embed = OverlapPatchEmbed(32, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.down_img = nn.PixelUnshuffle(2)
        self.conv_init = nn.Sequential(nn.Conv2d(inp_channels*4,32,3,1,1), nn.Conv2d(32,32,3,1,1), nn.Conv2d(32,32,3,1,1), nn.Conv2d(32,32,3,1,1), nn.InstanceNorm2d(32, affine = True))

        self.fuse1 = Fuse_res(dim)
        self.fuse2 = Fuse_res(dim*2)
        # self.fuse3 = Fuse_res(dim*3)
        self.fuse3 = Fuse_res(dim*4)

        self.resconv1 = nn.Conv2d(64,dim,1)
        self.resconv2 = nn.Conv2d(64,dim*2,1)
        self.resconv3 = nn.Conv2d(128,dim*4,1)
    def forward(self, inp_img, res_feats):
        _, res3, res2, res1 = res_feats[0], res_feats[1], res_feats[2], res_feats[3]
        res1 = self.resconv1(res1)
        res2 = self.resconv2(res2)
        res3 = self.resconv3(res3)

        inp_img = (inp_img - 0.45) / 0.225
        inp_img = self.conv_init(self.down_img(inp_img))
        # print(inp_img.size())
        # exit()

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # out_enc_level1 = out_enc_level1 + self.fuse1(out_enc_level1, res1) #### deactivated residual in fusion
        out_enc_level1 = self.fuse1(out_enc_level1, res1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # print('res2.shape->', res2.shape)
        # print('out_enc_level2.shape->', out_enc_level2.shape)
        # out_enc_level2 = out_enc_level2 + self.fuse2(out_enc_level2, res2) #### deactivated residual in fusion
        out_enc_level2 = self.fuse2(out_enc_level2, res2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        # out_enc_level3 = out_enc_level3 + self.fuse3(out_enc_level3, res3) #### deactivated residual in fusion
        out_enc_level3 = self.fuse3(out_enc_level3, res3)

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)

        return [latent, out_enc_level3, out_enc_level2, out_enc_level1]

class VIT2Decoder_light(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=32, 
        dim = 48,
        num_blocks = [3,4,4,4],
        # num_blocks = [2,2,3,4], 
        num_refinement_blocks = 1,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias'
    ):
        super(VIT2Decoder_light, self).__init__()

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.block1 = Depth_Block(32)
        self.block2 = Depth_Block(dim * 2)
        self.block3 = Depth_Block(dim * 4)
    def forward(self, feats):
        latent, out_enc_level3, out_enc_level2, out_enc_level1 = feats[0], feats[1], feats[2], feats[3]        
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)# + inp_img
        # print(out_dec_level3.size(), out_dec_level2.size(), out_dec_level1.size())
        # exit()

        op = {}
        # print(res16x.size(),res8x.size(),res4x.size(),res2x.size(),x.size())
        # exit()
        depth_4 = self.block3(out_dec_level3, False, ada = True)
        depth_3 = self.block3(out_dec_level3, True, ada = False, prvs = depth_4)
        depth_2 = self.block2(out_dec_level2, True, ada = False, prvs = depth_3)
        depth_1 = self.block1(out_dec_level1, True, ada = False, prvs = depth_2)
        

        op[("disp", 0)] = depth_1
        op[("disp", 1)] = depth_2
        op[("disp", 2)] = depth_3
        op[("disp", 3)] = depth_4
        # print(op[("disp", 0)].size(), op[("disp", 1)].size(), op[("disp", 2)].size(), op[("disp", 3)].size())
        # exit()
        return op



                        
        
