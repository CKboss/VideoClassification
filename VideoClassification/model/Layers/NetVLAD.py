import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class NetVLAD(nn.Module):

    def __init__(self,D,K):

        super(NetVLAD,self).__init__()

        self.D = D
        self.K = K

        self.filt = Parameter(torch.randn(K,1,1,D))
        self.center = Parameter(torch.randn(D,K))
        self.biases = Parameter(torch.randn(K))

    def forward(self,x):
        '''
        :param x: a tensor size is BxDxHxW
        :return: a tensor Bx(DxK)
        '''

        x = x.permute(0,2,3,1)
        x.contiguous()

        B = x.size(0)
        N = x.size(1)*x.size(2)

        x = x.view(B,N,self.D)

        x = F.normalize(x,p=2)
        descripter = x.view(B,N,self.D,1) # [BxNxDx1]

        # reshape for conv
        descripter = descripter.permute(0,3,1,2)
        conv_vlad = F.conv2d(descripter,self.filt)

        conv_vlad = conv_vlad.permute(0,2,3,1) # [BxNx1xK]
        conv_vlad += self.biases

        a_k = F.softmax(conv_vlad)
        a_k = a_k.view(a_k.size(0),a_k.size(1),a_k.size(3)) # [BxNxK]

        v1 = torch.matmul(torch.transpose(x,1,2),a_k) # [BxDxK]
        v2 = torch.sum(a_k,dim=1).view(B,1,self.K) * self.center # [BxDxK]

        v = v1-v2 #[BxDxK]

        norm = F.normalize(F.normalize(v,p=2,dim=1).view(-1,self.D*self.K),p=2,dim=1) # [Bx(DxK)]

        return norm

if __name__=='__main__':

    x = Variable(torch.randn(4,512,14,14))
    x = x.permute(0,2,3,1)
    x = x.view(x.size(0),x.size(1)*x.size(2),x.size(3))

    model = NetVLAD(D=512,K=64)

    y = model(x)
