import torch
import torch.nn as nn
import math

def conv_3x3_act(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_act(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_act2(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def dw_conv(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup)
    )

def dw_conv2(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def dw_conv3(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def upsample(inp, oup, scale=2):
    return nn.Sequential(
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp),
                nn.ReLU(inplace=True),
                conv_1x1_act2(inp,oup),
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))

def IRBlock(oup, hidden_dim):
    return nn.Sequential(
            nn.Conv2d(oup, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, n):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.n = n

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.conv2 = torch.nn.ModuleList()
        for i in range(n):
            self.conv2.append(IRBlock(oup, hidden_dim))

    def forward(self, x):
        x = self.conv1(x)

        for i in range(self.n):
            x = x + self.conv2[i](x)

        return x

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        #mobilenet v2

        input_channel = 32

        self.features1 = nn.Sequential(*[
                            conv_3x3_act(3, input_channel, 2),
                            dw_conv(input_channel, 16, 1),
                            InvertedResidual(16, 24, 2, 6, 1)
                        ])

        self.features2 = InvertedResidual(24, 32, 2, 6, 2)
        self.features3 = InvertedResidual(32, 64, 2, 6, 3)

        self.features4 = nn.Sequential(*[
                            InvertedResidual(64, 96, 1, 6, 2),
                            InvertedResidual(96, 160, 2, 6, 2),
                            InvertedResidual(160, 320, 1, 6, 0),
                            conv_1x1_act(320,1280),
                            nn.Conv2d(1280, 64, 1, 1, 0, bias=False),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                        ])

        self.upsample2 = upsample(64, 32)
        self.upsample1 = upsample(32, 24)

        self.conv3 = nn.Conv2d(64, 64, 1, 1, 0)
        self.conv2 = nn.Conv2d(32, 32, 1, 1, 0)
        self.conv1 = nn.Conv2d(24, 24, 1, 1, 0)

        self.conv4 = dw_conv3(24, 24, 1)

    def forward(self, x):
        x = x/127.5-1

        f1 = self.features1(x)

        f2 = self.features2(f1)

        f3 = self.features3(f2)

        f4 = self.features4(f3)
        f3 = self.conv3(f3)
        f4 += f3

        f4 = self.upsample2(f4)
        f2 = self.conv2(f2)
        f4 += f2

        f4 = self.upsample1(f4)
        f1 = self.conv1(f1)
        f4 += f1

        f4 = self.conv4(f4)

        return f4

class Header(nn.Module):
    def __init__(self, num_classes, mode='train'):
        super(Header, self).__init__()

        self.mode = mode

        self.header_heatmaps = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2d(96, num_classes, 1, 1, 0, bias=True),
                        nn.Sigmoid()
                    ])

        self.header_centers = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2d(96, 1, 1, 1, 0, bias=True),
                        nn.Sigmoid(),
                    ])

        self.header_regs = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2d(96, num_classes*2, 1, 1, 0, bias=True),
                    ])

        self.header_offsets = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2d(96, num_classes*2, 1, 1, 0, bias=True),
                    ])

    def argmax2loc(self, x, h=48, w=48):
        y0 = torch.div(x,w).long()
        x0 = torch.sub(x, y0*w).long()
        return x0,y0

    def forward(self, x):

        res = []
        if self.mode=='train':
            h1 = self.header_heatmaps(x)
            h2 = self.header_centers(x)
            h3 = self.header_regs(x)
            h4 = self.header_offsets(x)
            res = [h1,h2,h3,h4]

        elif self.mode=='test':
            pass

        elif self.mode=='all':
            pass

        else:
            print("[ERROR] wrong mode.")

        return res

class PoseKitModel(nn.Module):
    def __init__(self, num_classes=17, width_mult=1.,mode='train'):
        super(PoseKitModel, self).__init__()

        self.backbone = Backbone()

        self.header = Header(num_classes, mode)

        self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)

        x = self.header(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    from torchsummary import summary

    model = PoseKitModel.cuda()
    print(summary(model, (3, 192, 192)))

    dummy_input1 = torch.randn(1, 3, 192, 192).cuda()
    input_names = [ "input1" ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input1, "pose.onnx",
        verbose=True, input_names=input_names, output_names=output_names,
        do_constant_folding=True,opset_version=11)
