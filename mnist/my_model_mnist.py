import torch as th
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.modules.loss as Moddy
from nltk.corpus import wordnet as wn
from torch.autograd import Function, Variable
import numpy as np

z_dim=2

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


eps = 1e-5


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
        return th.sum(self.ruler(zet*position_zet[target.data]-inner))


class myLossA(Moddy._WeightedLoss):
    ruler=PoincareDistance()
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(myLossA, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input):
        zet=((th.sqrt(th.sum(input*input, dim=-1)+1)).expand(z_dim,-1)).t()
        bs=input/(zet+1)
        resulty=Variable(th.FloatTensor(input.size()[0],10))
        for i in range(0,input.size()[0]):
            for j in range(0,10):
                    resulty[i,j]=self.ruler(bs[i], position[j])
        return resulty


position=np.zeros((10,2))
for i in range(0,10):
    position[i][0]=3*np.cos(np.pi*2/10*i)
    position[i][1]=3*np.sin(np.pi*2/10*i)

position=Variable(th.FloatTensor(position),requires_grad=False)
position_zet= (th.sqrt(th.sum(position*position, dim=-1)+1))