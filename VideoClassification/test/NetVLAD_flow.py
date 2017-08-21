import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


x = torch.randn(4,14,14,512)
x = Variable(x)


B=4
D=512
K=64

# filt = torch.randn(1,D,1,K)
# 输出c,输入c,H,W
filt = torch.randn(K,1,1,D)
center = torch.randn(D,K)
biases = torch.randn(K)

filt = Variable(filt)
center = Variable(center)
biases = Variable(biases)


x = x.view(x.size(0),-1,x.size(3)) # [BxNxD]

x = F.normalize(x,p=2) # [BxNxD]

descripter = x.view(x.size(0),x.size(1),x.size(2),1) # [BxNxDx1]

descripter = descripter.permute(0,3,1,2)

conv_vlad = F.conv2d(descripter,filt)

conv_vlad = conv_vlad.permute(0,2,3,1) # [BxNx1xK]

conv_vlad += biases

a_k = F.softmax(conv_vlad)

a_k = a_k.view(a_k.size(0),a_k.size(1),a_k.size(3)) # [BxNxK]

v1 = torch.matmul(torch.transpose(x,1,2),a_k) # [BxDxK]

v2 = torch.sum(a_k,dim=1).view(B,1,K) * center # [BxDxK]

v = v1-v2 #[BxDxK]

norm = F.normalize(F.normalize(v,p=2,dim=1).view(-1,D*K),p=2,dim=1)
