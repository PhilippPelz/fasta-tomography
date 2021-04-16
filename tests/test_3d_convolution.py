import torch.nn as nn
from scipy import signal
import torch.optim as opt
from fastatomography.util import *
import torch.nn.functional as F


class MyConv3dGaussian(nn.Conv3d):
    def __init__(self, sigma, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(MyConv3dGaussian, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        ss = kernel_size
        Y, X, Z = np.mgrid[-ss[0] // 2:ss[0] // 2, -ss[1] // 2:ss[1] // 2, -ss[2] // 2:ss[2] // 2].astype(np.float32)
        self.r = nn.Parameter(data=th.as_tensor([Y, X, Z]) , requires_grad=False)
        self.sigma = nn.Parameter(data=sigma, requires_grad=True)

    def forward(self, input):
        weight = th.exp(-th.sum((self.r / self.sigma) ** 2, 0)).unsqueeze(0).unsqueeze(0)
        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride, (0, 0, 0),
                            self.dilation, self.groups)

        return F.conv3d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


th.backends.cudnn.deterministic = True

in_channels = 1
out_channels = 1
kernel_size = np.array([11, 11, 11])

d = 100
vol = th.zeros((d, d, d))
vol[50, 50, 50] = 1
vol[50, 50, 60] = 1
vol[50, 30, 60] = 1
vol = vol.cuda().unsqueeze(0).unsqueeze(0)

s = th.tensor([2.])
conv = MyConv3dGaussian(s, in_channels, out_channels, kernel_size, bias=False,padding=tuple(kernel_size//2)).cuda()
target = conv(vol).detach()
#%%
fig, ax = plt.subplots(1,2)
ax[0].imshow(vol[0,0,50, ...].detach().cpu())
ax[1].imshow(target[0,0,50, ...].detach().cpu()-vol[0,0,50, ...].detach().cpu())
plt.show()
#%%
from fastatomography.util.atom_tracing.atom_tracing import peakFind3D

p = peakFind3D(target[0,0].cpu().numpy(), 0.5)
p = Param(p)
init = th.zeros((d, d, d))
init[p.xp,p.yp,p.zp] = 1
init = th.roll(init,tuple(kernel_size//2),(0,1,2))
p2 = peakFind3D(init.numpy(), 0.5)
#%%

s = th.tensor([3.])
conv2 = MyConv3dGaussian(s, in_channels, out_channels, kernel_size, bias=False).cuda()

test_vol = init.cuda().unsqueeze(0).unsqueeze(0).requires_grad_()

loss = nn.MSELoss()
optimizer = opt.Adam([conv2.sigma], lr=2e-2)
optimizer2 = opt.Adam([test_vol], lr=2e-2)

for i in range(100):
    test_vol = test_vol.clone().requires_grad_()

    optimizer2.zero_grad()
    m = conv2(test_vol)
    l = loss(m, target)
    l.backward()
    print(f'{i:03d} Loss 1: {l}')
    optimizer2.step()

    test_vol = test_vol.detach()

    if i % 2 == 0:
        test_vol[test_vol < 0.5] = 0

    if i % 10 == 0:
        plot(test_vol[0,0,50, ...].detach().cpu(), f'test_vol it {i}')

    optimizer.zero_grad()
    l = loss(conv2(test_vol), target)
    l.backward()
    print(f'{i:03d} Loss 1: {l}, s= {conv2.sigma[0]} , sigma.grad= {conv2.sigma.grad[0]}')
    optimizer.step()


