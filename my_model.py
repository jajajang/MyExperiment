import torch as th
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.modules.loss as Moddy
from nltk.corpus import wordnet as wn
from torch.autograd import Function, Variable


z_dim=100
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class myResNet(nn.Module):

    def __init__(self, block, layers, num_dims=z_dim):
        self.inplanes = 64
        super(myResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_dims)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def myResnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = myResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


eps=1e-5
class PoincareDistance(Function):
    boundary = 1 - eps

    def grad(self, x, v, sqnormx, sqnormv, sqdist):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    def forward(self, u, v):
        self.save_for_backward(u, v)
        self.squnorm = th.clamp(th.sum(u * u, dim=-1), 0, self.boundary)
        self.sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, self.boundary)
        self.sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    def backward(self, g):
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = self.grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = self.grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv


class Arcosh(Function):
    def __init__(self, eps=eps):
        super(Arcosh, self).__init__()
        self.eps = eps

    def forward(self, x):
        self.save_for_backward(x)
        self.z = th.sqrt(x * x - 1)
        return th.log(x + self.z)

    def backward(self, g):
        z = th.clamp(self.z, min=eps)
        z = g / z
        return z


class myLoss(Moddy._WeightedLoss):

    ruler=Arcosh()
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(myLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce
    def forward(self, input, target):
        Moddy._assert_no_grad(target)
        zet=((th.sqrt(th.sum(input*input, dim=-1)+1)))
        inner=(th.sum(input*position[target.data], dim=-1))
        return th.sum(self.ruler(zet*position_zet[target.data]-inner).cuda())

class whereru(Function):
    ruler=Arcosh()
    def __init__(self):
        super(whereru,self).__init__()
    
    def forward(self,input):
        resulty=Variable(th.cuda.FloatTensor(position.size()[0]))
        for j in range(0,position.size()[0]):
            zet=(th.sqrt(th.sum(input*input, dim=-1)+1))
            resulty[j]=-self.ruler(zet*position_zet[j]-th.sum(input*position[j], dim=-1)).cuda()
        return resulty

class myLossA(Moddy._WeightedLoss):
    ruler=whereru()
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(myLossA, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input):
        return self.ruler(input)


ordered_word=[]
f=open('wnids.txt','r')
while True:
    line=f.readline()
    if not line: break
    ordered_word.append(wn.synset_from_pos_and_offset(line[0],int(line[1:-1])).name())

tempy=th.load('bigger_dim.pth')
obj=tempy['objects']
position_=(tempy['model']['lt.weight']).float()

true_pos=th.zeros(200,z_dim)
for i in range(0,200):
    true_pos[i]=position_[obj.index(ordered_word[i])]

zetty=-th.sum(true_pos*true_pos, dim=-1).expand(z_dim,-1)+1
position=2*true_pos/zetty.t()
position_zet= (th.sqrt(th.sum(position*position, dim=-1)+1))
