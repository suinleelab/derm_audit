#!/usr/bin/env python
import numpy as np
import torch

M_16 = torch.tensor([
    [0.401288, 0.650173, -0.051461],
    [-0.250268, 1.204414, 0.045854],
    [-0.002079, 0.048952, 0.953127]
    ],
    dtype=torch.float32)

M_16_inv = torch.inverse(M_16)

surround_map = {
        'average': {'F': 1, 'c': 0.69, 'Nc': 1},
        'dim': {'F': 0.9, 'c': 0.59, 'Nc': 0.9},
        'dark': {'F': 0.8, 'c': 0.525, 'Nc': 0.8}
        }

def calculate_independent_parameters(xyz_w, L_A, surround='average', Y_b=100):
    RGB_w = torch.matmul(M_16, xyz_w)

    F = surround_map[surround]['F']
    c = surround_map[surround]['c']
    N_c = surround_map[surround]['Nc']

    D = F*(1-(1/3.6)*torch.exp((-1*torch.tensor(L_A, dtype=torch.float32)-42)/92))
    if D<0: D = 0
    if D>1: D = 1

    D_R = D*xyz_w[1]/RGB_w[0]+1-D
    D_G = D*xyz_w[1]/RGB_w[1]+1-D
    D_B = D*xyz_w[1]/RGB_w[2]+1-D

    k=1/(5*L_A+1)
    F_L=0.2*(k**4)*L_A*5+0.1*(1-k**4)**2*(5*L_A)**(1/3)
    n = Y_b/xyz_w[1]
    z = 1.48+np.sqrt(n)
    N_bb = 0.725*(1/n)**0.2
    N_bc = N_bb
    RGB_wc = torch.matmul(
            torch.tensor([
                [D_R, 0, 0],
                [0, D_G, 0],
                [0, 0, D_B]
                ]),
            RGB_w
            )
    RGB_aw = 400*((F_L*RGB_wc/100)**0.42/((F_L*RGB_wc/100)**0.42+27.13))+0.1
    A_w = (2*RGB_aw[0]+RGB_aw[1]+RGB_aw[2]/20-0.305)*N_bb
    return F_L, n, z, N_c, N_bb, N_bc, RGB_w, D, D_R, D_G, D_B, RGB_wc, RGB_aw, A_w

class CAM16ToXYZ(torch.nn.Module):
    '''
    Convert from CAM16 to CIEXYZ. Note this module may only work with scalar inputs currently!
    '''
    def __init__(self, xyz_w, L_A=300, surround='average', Y_b=100):
        super().__init__()
        self.pi = torch.tensor(np.pi, dtype=torch.float32)
        self.F_L, self.n, self.z, self.N_c, self.N_bb, self.N_cb, self.RGB_w, self.D, self.D_R, self.D_G, self.D_B, self.RGB_wc, self.RGB_aw, self.A_w = calculate_independent_parameters(xyz_w, L_A, surround=surround, Y_b=Y_b)

    def forward(self, J, C, h):
        J = torch.tensor(J)
        C = torch.tensor(C)
        h = torch.tensor(h)

        t = (C/(torch.sqrt(J/100)*(1.64-0.29**self.n)**0.73))**(1/0.9)
        e_t = 1/4*(np.cos(h*self.pi/180+2)+3.8)
        A = self.A_w*(J/100)**(1/(C*self.z))
        if t != 0:
            p_1 = (50000/13*self.N_c*self.N_cb)*e_t*(1/t)
        p_2 = A/self.N_bb + 0.305
        p_3 = 21/20

        if t == 0:
            a = 0
            b = 0
        else:
            if torch.abs(torch.sin(h*self.pi/180)) > torch.abs(torch.cos(h*self.pi/180)):
                p_4 = p_1/torch.sin(h*self.pi/180)
                b = p_2*(2+p_3)*(460/1403)/(p_4 + (2+p_3)*(220/1403)*(torch.cos(h*self.pi/180)/torch.sin(h*self.pi/180))-(27/1403)+p_3*(6300/1403))
                a = b*(torch.cos(h*self.pi/180)/torch.sin(h*self.pi/180))
            else:
                p_5 = p_1/torch.cos(h*self.pi/180)
                a = p_2*(2 + p_3)*(460/1403)/(p_5 + (2+p_3)*(220/1403)-((27/1403)-p_3*(6300/1403))*(torch.sin(h*self.pi/180)/torch.cos(h*self.pi/180)))
                b = a*(torch.sin(h*self.pi/180)/torch.cos(h*self.pi/180))

        R_a = 460/1403*p_2 + 451/1403*a + 288/1403*b
        G_a = 460/1403*p_2 - 891/1403*a - 261/1403*b
        B_a = 460/1403*p_2 - 220/1403*a - 6300/1403*b

        R_c = torch.sign(R_a-0.1)*(100/self.F_L)*((27.13*torch.abs(R_a-0.1))/(400-torch.abs(R_a-0.1)))**(1/0.42)
        G_c = torch.sign(G_a-0.1)*(100/self.F_L)*((27.13*torch.abs(G_a-0.1))/(400-torch.abs(G_a-0.1)))**(1/0.42)
        B_c = torch.sign(B_a-0.1)*(100/self.F_L)*((27.13*torch.abs(B_a-0.1))/(400-torch.abs(B_a-0.1)))**(1/0.42)

        R = R_c/self.D_R
        G = G_c/self.D_G
        B = B_c/self.D_B
        RGB = torch.tensor([R,G,B])
        XYZ = torch.matmul(M_16_inv, RGB)
        return XYZ

class CAM16CAT(torch.nn.Module):
    def __init__(self, xyz_w, xyz_wr, D=1, surround='average', L_A=300):
        super().__init__()
        if D is None:
            F = surround_map[surround]['F']
            c = surround_map[surround]['c']
            Nc = surround_map[surround]['Nc']

            D = F*(1-(1/3.6)*torch.exp((-1*torch.tensor(L_A, dtype=torch.float32)-42)/92))
            if D<0: D = 0
            if D>1: D = 1

        rgb_w = torch.matmul(M_16, xyz_w)
        rgb_wr = torch.matmul(M_16, xyz_wr)


        lambda_rt = torch.tensor([
            [D*xyz_w[1]/xyz_wr[1]*rgb_wr[0]/rgb_w[0] + 1 - D, 0, 0],
            [0, D*xyz_w[1]/xyz_wr[1]*rgb_wr[1]/rgb_w[1] + 1 - D, 0],
            [0, 0, D*xyz_w[1]/xyz_wr[1]*rgb_wr[2]/rgb_w[2] + 1 - D]
            ])
        self.Phi_rt = torch.matmul(torch.matmul(M_16_inv, lambda_rt), M_16)

    def forward(self, xyz):
        batch = xyz.shape[0]
        h = xyz.shape[2]
        w = xyz.shape[3]
        assert xyz.shape[1] == 3
        flattened = xyz.permute(0,2,3,1).flatten(start_dim=0, end_dim=2)
        xyz_c = torch.matmul(flattened, self.Phi_rt.T.to(flattened.device))
        return xyz_c.reshape(batch, h, w, 3).permute(0,3,1,2)
