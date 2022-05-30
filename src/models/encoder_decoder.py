import torch
import torch.nn as nn
from .net_blocks import Conv2dBlock, ResBlock, ResBlocks, MLP

class Decoder_j_(nn.Module):
    def __init__(self):
        super(Decoder_j_, self).__init__()

        # AdaIN residual blocks
        self.res = nn.Sequential(
            Conv2dBlock(1024, 512, 3, 1, 1, 'adain', 'relu', 'zero'),
            Conv2dBlock(512, 512, 3, 1, 1, 'adain', 'relu', 'zero'),
            ResBlock(512, 512, 'adain', 'relu', 'zero'),
            ResBlock(512, 512, 'adain', 'relu', 'zero'),
            ResBlock(512, 512, 'adain', 'relu', 'zero'),
        )
        # upsampling blocks
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(512, 512, 5, 1, 2, norm='ln', activation='relu', pad_type='zero'),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(512, 256, 5, 1, 2, norm='ln', activation='relu', pad_type='zero'),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(256, 128, 5, 1, 2, norm='ln', activation='relu', pad_type='zero'),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(128, 64, 5, 1, 2, norm='ln', activation='relu', pad_type='zero'),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.res(x)
        x = self.up(x)
        return x

class Decoder_j(nn.Module):
    def __init__(self):
        super(Decoder_j, self).__init__()

        # MLP to generate AdaIN parameters
        self.dec = Decoder_j_()
        self.mlp = MLP(7*7*512, self.get_num_adain_params(self.dec), 256, 3, norm='none', activ='relu')

    def forward(self, content, style):
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(torch.cat([content, style], dim=1))
        return images
    
    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # residual blocks
        self.res = ResBlocks(4, 'ln', 'relu', 'zero')
        # upsampling blocks
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(512, 512, 5, 1, 2, norm='ln', activation='relu', pad_type='zero'),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(512, 256, 5, 1, 2, norm='ln', activation='relu', pad_type='zero'),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(256, 128, 5, 1, 2, norm='ln', activation='relu', pad_type='zero'),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(128, 64, 5, 1, 2, norm='ln', activation='relu', pad_type='zero'),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.res(x)
        x = self.up(x)
        return x

class Encoder(nn.Module):
    def __init__(self, norm='in'):
        super(Encoder, self).__init__()
        self.norm = norm
        
        # downsampling blocks
        self.down = nn.Sequential(
            Conv2dBlock(1, 64, 7, 2, 3, norm=self.norm, activation='relu', pad_type='zero'),
            Conv2dBlock(64, 64, 4, 2, 1, norm=self.norm, activation='relu', pad_type='zero'),
            Conv2dBlock(64, 128, 4, 2, 1, norm=self.norm, activation='relu', pad_type='zero'),
            Conv2dBlock(128, 256, 4, 2, 1, norm=self.norm, activation='relu', pad_type='zero'),
            Conv2dBlock(256, 512, 4, 2, 1, norm=self.norm, activation='relu', pad_type='zero')
        )
        # residual blocks
        self.res = ResBlocks(4, self.norm, 'relu', 'zero')

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x

if __name__ == '__main__':

    import os
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    encoder = Encoder('in')
    decoder = Decoder()
    decoder_j = Decoder_j()
    encoder.cuda()
    decoder.cuda()
    decoder_j.cuda()
    x = np.zeros((1, 1, 224, 224)).astype('float32')
    x = torch.from_numpy(x)
    x = x.cuda()
    z = encoder(x)
    y = decoder(z)
    w = decoder_j(z,z)

    print(
        'x size: {}'.format(x.shape),
        'z size: {}'.format(z.shape),
        'y size: {}'.format(y.shape),
        'w size: {}'.format(w.shape),
        sep='\n'
    )