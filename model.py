import torch
from torch import nn


# data size (C, H, W) = (1, 320, 320)

# 
class GenConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True):
        super(GenConvBlock, self).__init__()

        # first /final encode layers without batchnorm
        if bn:
            self.ConvLayer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            self.ConvLayer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4,4), stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2)
            )
    
    def forward(self, x):
        return self.ConvLayer(x)


class GenDeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenDeConvBlock, self).__init__()

        self.DeConvLayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.DeConvLayer(x)


class Generater(nn.Module):
    def __init__(self):
        super(Generater, self).__init__()

        self.down1 = GenConvBlock(in_channels=1, out_channels=64, bn=False) # (64, 160, 160)
        self.down2 = GenConvBlock(in_channels=64, out_channels=128)  # (128, 80, 80)
        self.down3 = GenConvBlock(in_channels=128, out_channels=256) # (256, 40, 40)
        self.down4 = GenConvBlock(in_channels=256, out_channels=512) # (512, 20, 20)
        self.down5 = GenConvBlock(in_channels=512, out_channels=512) # (512, 10, 10)

        self.down6 = GenConvBlock(in_channels=512, out_channels=512) # (512, 5, 5)
        # self.down7 = GenConvBlock(in_channels=512, out_channels=512) # ()

        # self.down8 = GenConvBlock(in_channels=512, out_channels=512, bn=False)

        # skip connection
        # self.up7 = GenDeConvBlock(in_channels=512, out_channels=512)
        # self.up6 = GenDeConvBlock(in_channels=1024, out_channels=1024)
        self.up5 = GenDeConvBlock(in_channels=512, out_channels=512)
        self.up4 = GenDeConvBlock(in_channels=1024, out_channels=256)
        self.up3 = GenDeConvBlock(in_channels=768, out_channels=256)
        self.up2 = GenDeConvBlock(in_channels=512, out_channels=128)
        self.up1 = GenDeConvBlock(in_channels=256, out_channels=64)

        self.up0 = GenDeConvBlock(in_channels=128, out_channels=64)

        self.c0 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=1, padding=0)
        self.tanh = nn.Tanh()

    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        # u7 = self.up7(d)
        # u6 = self.up6(torch.cat((u7, d7), 1))
        u5 = self.up5(d6)
        u4 = self.up4(torch.cat((u5, d5), 1))
        u3 = self.up3(torch.cat((u4, d4), 1))
        u2 = self.up2(torch.cat((u3, d3), 1))
        u1 = self.up1(torch.cat((u2, d2), 1))
        u0 = self.up0(torch.cat((u1, d1), 1))

        output = self.c0(u0)
        output = self.tanh(output)

        return x+output


class DisConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn, ks, st, pd):
        super(DisConvBlock, self).__init__()
        if bn:
            self.ConvLayer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(ks,ks), stride=st, padding=pd),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            self.ConvLayer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(ks,ks), stride=st, padding=pd),
                nn.LeakyReLU(negative_slope=0.2)
            )
    
    def forward(self, x):
        return self.ConvLayer(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.c1 = DisConvBlock(in_channels=1, out_channels=64, bn=False, ks=4, st=2, pd=1)     # (64, 160, 160)
        self.c2 = DisConvBlock(in_channels=64, out_channels=128, bn=True, ks=4, st=2, pd=1)    # (128, 80, 80)
        self.c3 = DisConvBlock(in_channels=128, out_channels=256, bn=True, ks=4, st=2, pd=1)   # (256, 40 ,40)
        self.c4 = DisConvBlock(in_channels=256, out_channels=512, bn=True, ks=4, st=2, pd=1)   # (512, 20, 20)
        self.c5 = DisConvBlock(in_channels=512, out_channels=1024, bn=True, ks=4, st=2, pd=1)  # (1024, 10, 10)
        self.c6 = DisConvBlock(in_channels=1024, out_channels=512, bn=True, ks=4, st=2, pd=1)  # (512, 5, 5)
        # self.c7 = DisConvBlock(in_channels=2048, out_channels=1024, bn=True, ks=1, st=1, pd=0) # 
        # self.c8 = DisConvBlock(in_channels=1024, out_channels=512, bn=True, ks=1, st=1, pd=0)

        # residual block
        self.r1 = DisConvBlock(in_channels=512, out_channels=1024, bn=True, ks=1, st=1, pd=0) # (1024, 5, 5)
        self.r2 = DisConvBlock(in_channels=1024, out_channels=512, bn=True, ks=3, st=1, pd=1)  # (512, 5, 5)
        self.r3 = DisConvBlock(in_channels=512, out_channels=512, bn=True, ks=3, st=1, pd=1)  # (512, 5, 5)

        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.linear1 = nn.Linear(512*5*5, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x6 = self.c6(x5) # (512, 5, 5)
        # x7 = self.c7(x6)
        # x8 = self.c8(x7)  # size (512, H, W)

        # residual block
        rs1 = self.r1(x6)
        rs2 = self.r2(rs1)
        rs3 = self.r3(rs2) # size (512, H, W)

        # print(x6.shape, rs3.shape)
        xa = self.LeakyReLU(x6+rs3)
        xo = xa.view(xa.shape[0], -1)
        # print(xo.shape)
        xl = self.linear1(xo)
        output = self.Sigmoid(xl)

        return output


# x = torch.rand(2,1,320,320)

# Gen = Generater()
# test_gen = Gen.forward(x)
# print(test_gen.shape)

# Dis = Discriminator()

# test_dis = Dis.forward(x)
# print(test_dis.shape)










        







