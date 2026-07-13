import torch
import torch.nn as nn

from .masks import CentralMaskedConv2d, RowMaskedConv2d, ColMaskedConv2d, \
    SzMaskedConv2d, fSzMaskedConv2d, angle45MaskedConv2d, \
    angle135MaskedConv2d, chaMaskedConv2d, fchaMaskedConv2d, huiMaskedConv2d
from .vmamba import MambaIR

class vmamba(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9, mask_type='o_fsz'):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"

        self.mask_dict = {'central': ['branch1_1', 'branch1_2'], 'col': 'branch2', 'row': 'branch3', 'sz': 'branch4',
                     'fsz': 'branch5', 'a45': 'branch6', 'a135': 'branch7', 'cha': 'branch9',
                     'fcha': 'branch10', 'hui': 'branch11'}

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.mask_types = mask_type.split('_')
        mask_number = len(self.mask_types)

        if 'o' in self.mask_types:  # 2 3
            self.branch1_1 = DC_branchl(2, base_ch, 'central', num_module)
            # self.branch1_2 = DC_branchl(3, base_ch, 'central', num_module)


    def forward(self, x):
        mask_types = self.mask_types
        y = []

        x = self.head(x)
        if 'o' in mask_types:
            br1_1 = self.branch1_1(x)
            # br1_2 = self.branch1_2(x)
            y.append(br1_1)
            # y.append(br1_2)
        

        x = torch.cat(y, dim=1)
        return x
        # return self.tail(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, mask_type, num_module):
        super().__init__()

        ly = []
        if mask_type == 'o':
            ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]

        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        # ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        # ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [MambaIR(embed_dim=in_ch, depths=[1], stride=stride)]
        ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]

        # ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)


