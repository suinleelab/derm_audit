#!/usr/bin/env python
import torch

class CIEXYZToJzazbz(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.tensor(1.15))
        self.register_buffer('g', torch.tensor(0.66))
        self.register_buffer('c_1', torch.tensor(3424/(2**12)))
        self.register_buffer('c_2', torch.tensor(2413/(2**7)))
        self.register_buffer('c_3', torch.tensor(2392/(2**7)))
        self.register_buffer('n', torch.tensor(2610/(2**14)))
        self.register_buffer('p', torch.tensor(1.7*2523/2**5))
        self.register_buffer('d', torch.tensor(-0.56))
        self.register_buffer('d_0', torch.tensor(1.6295499532821566*(10**-11)))
        self.register_buffer(
                'xyz_to_xpypz',
                torch.tensor([
                    [self.b, 0, -1*(self.b-1)],
                    [-1*(self.g-1), self.g, 0],
                    [0, 0, 1]
                    ])
                )
        self.register_buffer(
                'xyz_to_lms', 
                torch.tensor([
                    [0.41478972, 0.579999, 0.0146480],
                    [-0.2015100, 1.120649, 0.0531008],
                    [-0.0166008, 0.264800, 0.6684799]
                    ])
                )
        self.register_buffer(
                'lms_prime_to_iab',
                torch.tensor([
                    [0.5, 0.5, 0],
                    [3.524000, -4.066708, 0.542708],
                    [0.199076, 1.096799, -1.295875]
                    ])
                )

    def forward(self, im):
        batch = im.shape[0]

        h = im.shape[2]
        w = im.shape[3]
        assert im.shape[1] == 3
        flattened = im.permute(0,2,3,1).flatten(start_dim=0, end_dim=2)

        xp_yp_z = torch.matmul(flattened, self.xyz_to_xpypz.T) #
        lms = torch.matmul(xp_yp_z, self.xyz_to_lms.T)
        lpmpsp = ((self.c_1 + self.c_2*(lms/10000)**self.n)/(1+self.c_3*(lms/10000)**self.n))**self.p
        iab = torch.matmul(lpmpsp, self.lms_prime_to_iab.T)
        jz = (1+self.d)*iab[:,0]/(1+self.d*iab[:,0])-self.d_0
        jzazbz = torch.clone(iab)
        jzazbz[:,0] = jz
        return jzazbz.reshape(batch, h, w, 3).permute(0,3,1,2)

class JzazbzToCIEXYZ(CIEXYZToJzazbz):
    def __init__(self):
        super().__init__()
        self.register_buffer(
                'iab_to_lms_prime',
                torch.inverse(self.lms_prime_to_iab)
                )
        self.register_buffer(
                'lms_to_xyz',
                torch.inverse(self.xyz_to_lms)
                )
        self.register_buffer(
                'xpypzp_to_xyz',
                torch.inverse(self.xyz_to_xpypz)
                )

    def forward(self, im):
        batch = im.shape[0]

        h = im.shape[2]
        w = im.shape[3]
        assert im.shape[1] == 3
        jzazbz = im.permute(0,2,3,1).flatten(start_dim=0, end_dim=2)

        iz = (jzazbz[:,0] + self.d_0)/(1+self.d-self.d*(jzazbz[:,0] + self.d_0))
        izazbz = torch.clone(jzazbz)
        izazbz[:,0] = iz

        lpmpsp = torch.matmul(izazbz, self.iab_to_lms_prime.T)

        lms = 10000*((self.c_1 - lpmpsp**(1/self.p))/(self.c_3*lpmpsp**(1/self.p)-self.c_2))**(1/self.n)

        xpypzp = torch.matmul(lms, self.lms_to_xyz.T)
        xyz = torch.matmul(xpypzp, self.xpypzp_to_xyz.T)

        return xyz.reshape(batch, h, w, 3).permute(0,3,1,2)

