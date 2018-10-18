import torch as th
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.modules.loss as Moddy
from nltk.corpus import wordnet as wn
from torch.autograd import Function, Variable


device=th.device('cuda:0' if th.cuda.is_available() else 'cpu')
z_dim=2
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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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


class myResNetDeeper(nn.Module):

    def __init__(self, block, layers, num_dims=z_dim):
        self.inplanes = 64
        super(myResNetDeeper, self).__init__()
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
        self.fc = nn.Linear(512 * block.expansion, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 100)
        self.fc4 = nn.Linear(100, num_dims)

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
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x


def myResnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = myResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def myResnet18deep(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = myResNetDeeper(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def myResnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = myResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def myResnet34deep(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = myResNetDeeper(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def myResnet152(pretrained=False, **kwargs):
    model = myResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
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
        return th.sum(self.ruler(th.clamp(zet*position_zet[target.data]-inner,min=1+eps).to(device)))

class myLossC(Moddy._WeightedLoss):

    ruler=Arcosh()
    def forward(self, input):
        zet=(th.sqrt(th.sum(input*input, dim=-1)+1))
        return th.mean(self.ruler(th.clamp(zet,min=1+eps).to(device)))


class myLossL(Moddy._WeightedLoss):

    ruler=Arcosh()
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(myLossL, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce
    def forward(self, input, target,level):
        Moddy._assert_no_grad(target)
        levy=categorize[level]
        zet=((th.sqrt(th.sum(input*input, dim=-1)+1)))
        inner=(th.sum(input*position_all[levy[target.data]], dim=-1))
        return th.sum(self.ruler(th.clamp(zet*position_all_zet[categorize[level][target.data]]-inner,min=1+eps).to(device)))

class myLossEuc(Moddy._WeightedLoss):

    def forward(self, input, target,level):
        Moddy._assert_no_grad(target)
        diff=input-position_all[categorize[level][target.data]]
        return th.sum(diff*diff)

class myLossV(Moddy._WeightedLoss):

    ruler=Arcosh()
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(myLossV, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce
    def forward(self, input, target):
        #all=(target.data).expand(len(input))
        all=[target]*len(input)
        zet=((th.sqrt(th.sum(input*input, dim=-1)+1)))
        inner=(th.sum(input*position_all[all], dim=-1))
        return th.sum(self.ruler(zet*position_all_zet[all]-inner).to(device))

class myLossA(Moddy._WeightedLoss):
    ruler=Arcosh()
    def forward(self, input):
        resulty=Variable(th.FloatTensor(position.size()[0],input.size()[0]))
        for j in range(0,position.size()[0]):
            zet=(th.sqrt(th.sum(input*input, dim=-1)+1))
            beep= zet*position_zet[j]-th.sum(input*position[j], dim=-1)
            resulty[j]=-self.ruler(beep).to(device)
        return resulty.t()

class myLossSoft(Moddy._WeightedLoss):
    ruler=Arcosh()
    crossy=nn.CrossEntropyLoss()
    def forward(self, input, target):
        resulty=Variable(th.FloatTensor(position.size()[0],input.size()[0]))
        for j in range(0,position.size()[0]):
            zet=(th.sqrt(th.sum(input*input, dim=-1)+1))
            beep= zet*position_zet[j]-th.sum(input*position[j], dim=-1)
            resulty[j]=-self.ruler(beep).to(device)
        return self.crossy(resulty.t(), target)


class myLossAL(Moddy._WeightedLoss):
    ruler=Arcosh()
    def forward(self, input, level):
        resulty=Variable(th.FloatTensor(position.size()[0],input.size()[0]))
        for j in range(0,position.size()[0]):
            zet=(th.sqrt(th.sum(input*input, dim=-1)+1))
            beep= zet*position_all_zet[categorize[level][j]]-th.sum(input*position_all[categorize[level][j]], dim=-1)
            #here, no minus because I don't use topk shit
            resulty[j]=self.ruler(beep).to(device)
        return resulty.t()


class myLossEucAL(Moddy._WeightedLoss):
    def forward(self, input, level):
        resulty=Variable(th.FloatTensor(position.size()[0],input.size()[0]))
        for j in range(0,position.size()[0]):
            diff=input-position_all[categorize[level][j]]
            beep=th.sum(diff*diff, dim=-1)
            #here, no minus because I don't use topk shit
            resulty[j]=beep
        return resulty.t()


indy_mix=[]
ordered_word=[]
f=open('wnids.txt','r')
while True:
    line=f.readline()
    if not line: break
    #ordered_word.append(wn.synset_from_pos_and_offset(line[0],int(line[1:-1])).name())
    indy_mix.append(int(line[1:-1]))

ordered_indy=sorted(indy_mix)
for i in range(0,len(ordered_indy)):
    ordered_word.append(wn.synset_from_pos_and_offset('n',ordered_indy[i]).name())

tempy=th.load('dim_2_200.pth')
obj=tempy['objects']
position_=(tempy['model']['lt.weight']).float()

true_pos=th.zeros(200,z_dim)
for i in range(0,200):
    true_pos[i]=position_[obj.index(ordered_word[i])]

zetty=-th.sum(true_pos*true_pos, dim=-1).expand(z_dim,-1)+1
position=Variable(2*true_pos/zetty.t(),requires_grad=False).to(device)
position_zet= (th.sqrt(th.sum(position*position, dim=-1)+1))


zetty_all=(-th.sum(position_*position_, dim=-1).expand(z_dim,-1)+1)
position_all=Variable(2*position_/zetty_all.t(), requires_grad=False).to(device)
position_all_zet= (th.sqrt(th.sum(position_all*position_all, dim=-1)+1)).to(device)


synsetss=[0]*len(obj)
for i in range(0,len(obj)):
    synsetss[i]=wn.synset(obj[i])


level=[0]*len(obj)
for i in range(0,len(obj)):
    pathy=synsetss[i].hypernym_paths()[0]
    for j in range(0,len(obj)):
        if synsetss[j] in pathy:
            level[i]+=1

bitbit=[0]*13
for i in range(0,13):
    bitbit[i]=[j for j,x in enumerate(level) if x==i]

ordy=[0]*200
for i in range(0,200):
    ordy[i]=wn.synset(ordered_word[i])

categorize=[0]*13
for i in range(0,13):
    categorize[i]=[0]*200

for s in range(0,13):
    for i in range(0,200):
        pathy=ordy[i].hypernym_paths()[0]
        tik=True
        for j in bitbit[s]:
            if synsetss[j] in pathy:
                categorize[s][i]=j
                tik=False
            if tik:
                categorize[s][i]=categorize[s-1][i]    

categorize=th.LongTensor(categorize).to(device)