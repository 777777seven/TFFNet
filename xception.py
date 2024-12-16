""" 
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']

model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_input, key_input, value_input):
        batch_size, channels, height, width = query_input.size()

        query = self.query_conv(query_input).view(batch_size, channels, -1)  # (batch_size, channels, height*width)
        key = self.key_conv(key_input).view(batch_size, channels, -1)  # (batch_size, channels, height*width)
        value = self.value_conv(value_input).view(batch_size, channels, -1)  # (batch_size, channels, height*width)

        attn_scores = torch.bmm(query.transpose(1, 2), key)  # (batch_size, height*width, height*width)

        attn_weights = self.softmax(attn_scores)  # (batch_size, height*width, height*width)

        attn_output = torch.bmm(attn_weights, value.transpose(1, 2))
        attn_output = attn_output.transpose(1, 2).view(batch_size, channels, height, width)

        return attn_output


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, x):
        # features begin
        x = self.conv1(x) # [64, 32, 149, 149]
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x) # [64, 64, 147, 147]
        x = self.bn2(x)
        x = self.relu(x)
        
        textural_f_1 = self.block1(x) # [64, 128, 74, 74]
        textural_f_2 = self.block2(textural_f_1) # [64, 256, 37, 37]
        x = self.block3(textural_f_2) # [64, 728, 19, 19]
        x = self.block4(x) # [64, 728, 19, 19]
        x = self.block5(x) # [64, 728, 19, 19]
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x) # [64, 1024, 10, 10]
        
        x = self.conv3(x) # [64, 1536, 10, 10]
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x) # [64, 2048, 10, 10]
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1)) # [64, 2048, 1, 1]
        x = x.view(x.size(0), -1)
        # feature end

        y = self.fc(x)

        return [textural_f_1, textural_f_2, x, y]


def xception(pretrained=False,num_classes=1000):
    """
    Construct Xception.
    """

    model = Xception()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    return model

class xception_text(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000):
        super(xception_text, self).__init__()
        self.xception = Xception()
        if pretrained:
            self.xception.load_state_dict(model_zoo.load_url(model_urls['xception']))
        self.xception.fc = nn.Linear(in_features=2048, out_features=num_classes)
        self.fc_2 = nn.Linear(128, num_classes)

        self.t_down = nn.AvgPool2d(2, stride=2)
        self.t_conv_1 = nn.Sequential(
                        nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        )
        self.t_conv_concat = nn.Sequential(
                            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            )
        self.t_conv_add = nn.Sequential(
                        nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        )
        self.fc_3 = nn.Sequential(
                        nn.Linear(2048, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        )
        self.fc_head = nn.Sequential(
            nn.Linear(640, num_classes),
        )
        self.cross_attention = CrossAttention(in_channels=128)

    def fusion_forward(self, t1, t2, feat):
        t1 = self.t_down(t1)

        t_1_1 = t1[:int(t1.size(0) / 2), :]
        t_1_2 = t1[int(t1.size(0) / 2):, :]
        t_2_1 = t2[:int(t1.size(0) / 2), :]
        t_2_2 = t2[int(t1.size(0) / 2):, :]

        t_1 = torch.cat([t_1_1, t_2_1], dim=1)
        t_2 = torch.cat([t_1_2, t_2_2], dim=1)

        t_1 = self.t_conv_1(t_1)
        t_2 = self.t_conv_1(t_2)

        t_2_attention = self.cross_attention(t_1, t_2, t_2)
        t_1_attention = self.cross_attention(t_2, t_1, t_1)

        t_concat = torch.cat([t_1_attention, t_2_attention], dim=1)
        t_add = t_1_attention + t_2_attention

        t_concat = self.t_conv_concat(t_concat)
        t_add = self.t_conv_add(t_add)

        t_add_sigmoid = torch.sigmoid(t_add)
        t_concat = t_concat * t_add_sigmoid
        t_concat = F.adaptive_avg_pool2d(t_concat, (1, 1))
        t_concat = t_concat.view(t_concat.size(0), -1)

        feat = self.fc_3(feat)
        feat_1 = feat[:int(feat.size(0) / 2), :]
        feat_2 = feat[int(feat.size(0) / 2):, :]

        feat_concat = torch.cat([feat_1, feat_2, t_concat], dim=1)
        y = self.fc_head(feat_concat)
        return y

    def forward(self, x):
        t1, t2, feature, outputs_2 = self.xception(x)
        outputs_f = self.fusion_forward(t1, t2, feature)
        # return t1, t2, feature, outputs_f, outputs_2
        return outputs_2



if __name__ == '__main__':
    model = xception_text(pretrained=True, num_classes=2)

    print(model)
    # input_x = torch.randn([64,3,299,299])
    # t1, t2, feature, outputs_1, outputs_2 = model(input_x)
    # print(t1.shape, t2.shape, feature.shape, outputs_1.shape, outputs_2.shape)
