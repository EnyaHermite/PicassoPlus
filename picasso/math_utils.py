import math
import torch
import numpy as np
from scipy.special import sph_harm

TPI = torch.tensor(math.pi)


# phi for azimuth angle, theta for polar angle
def torch_sph_harm_reorder(L, phi, theta):
    phi = phi.squeeze()
    theta = theta.squeeze()

    components = torch.zeros(size=(phi.shape[0], (L+1)**2)).to(phi.get_device())
    if L>=0:
        components[...,0] = 1/2.*torch.sqrt(1/TPI)
    if L>=1:
        components[...,1] = 1/2.*torch.sqrt(3/2/TPI)*torch.sin(theta)*torch.cos(phi)
        components[...,2] = 1/2.*torch.sqrt(3/TPI)*torch.cos(theta)
        components[...,3] = 1/2.*torch.sqrt(3/2/TPI)*torch.sin(theta)*torch.sin(phi)
    if L>=2:
        components[...,4] = 1/4.*torch.sqrt(15/2/TPI)*(torch.sin(theta)**2)*torch.cos(2*phi)
        components[...,5] = 1/2.*torch.sqrt(15/2/TPI)*torch.sin(theta)*torch.cos(theta)*torch.cos(phi)
        components[...,6] = 1/4.*torch.sqrt(5/TPI)*(3*(torch.cos(theta)**2)-1)
        components[...,7] = 1/4.*torch.sqrt(15/2/TPI)*(torch.sin(theta)**2)*torch.sin(2*phi)
        components[...,8] = 1/2.*torch.sqrt(15/2/TPI)*torch.sin(theta)*torch.cos(theta)*torch.sin(phi)
    if L>=3:
        components[...,9] = 1/8.*torch.sqrt(35/TPI)*(torch.sin(theta)**3)*torch.cos(3*phi)
        components[...,10] = 1/4.*torch.sqrt(105/2/TPI)*(torch.sin(theta)**2)*torch.cos(theta)*torch.cos(2*phi)
        components[...,11] = 1/8.*torch.sqrt(21/TPI)*torch.sin(theta)*(5*(torch.cos(theta)**2)-1)*torch.cos(phi)
        components[...,12] = 1/4.*torch.sqrt(7/TPI)*(5*(torch.cos(theta)**3)-3*torch.cos(theta))

        components[...,13] = 1/8.*torch.sqrt(35/TPI)*(torch.sin(theta)**3)*torch.sin(3*phi)
        components[...,14] = 1/4.*torch.sqrt(105/2/TPI)*(torch.sin(theta)**2)*torch.cos(theta)*torch.sin(2*phi)
        components[...,15] = 1/8.*torch.sqrt(21/TPI)*torch.sin(theta)*(5*(torch.cos(theta)**2)-1)*torch.sin(phi)
    if L>=4:
        components[...,16] = 3/16.*torch.sqrt(35/2/TPI)*(torch.sin(theta)**4)*torch.cos(4*phi)
        components[...,17] = 3/8.*torch.sqrt(35/TPI)*(torch.sin(theta)**3)*torch.cos(theta)*torch.cos(3*phi)
        components[...,18] = 3/8.*torch.sqrt(5/2/TPI)*(torch.sin(theta)**2)*(7*(torch.cos(theta)**2)-1)*torch.cos(2*phi)
        components[...,19] = 3/8.*torch.sqrt(5/TPI)*torch.sin(theta)*(7*(torch.cos(theta)**3)-3*torch.cos(theta))*torch.cos(phi)
        components[...,20] = 3/16.*torch.sqrt(1/TPI)*(35*torch.cos(theta)**4-30*(torch.cos(theta)**2)+3)

        components[...,21] = 3/16.*torch.sqrt(35/2/TPI)*(torch.sin(theta)**4)*torch.sin(4*phi)
        components[...,22] = 3/8.*torch.sqrt(35/TPI)*(torch.sin(theta)**3)*torch.cos(theta)*torch.sin(3*phi)
        components[...,23] = 3/8.*torch.sqrt(5/2/TPI)*(torch.sin(theta)**2)*(7*(torch.cos(theta)**2)-1)*torch.sin(2*phi)
        components[...,24] = 3/8.*torch.sqrt(5/TPI)*torch.sin(theta)*(7*(torch.cos(theta)**3)-3*torch.cos(theta))*torch.sin(phi)
    if L>=5:
        components[...,25] = 3/32.*torch.sqrt(77/TPI)*(torch.sin(theta)**5)*torch.cos(5*phi)
        components[...,26] = 3/16.*torch.sqrt(385/2/TPI)*(torch.sin(theta)**4)*torch.cos(theta)*torch.cos(4*phi)
        components[...,27] = 1/32.*torch.sqrt(385/TPI)*(torch.sin(theta)**3)*(9*(torch.cos(theta)**2)-1)*torch.cos(3*phi)
        components[...,28] = 1/8.*torch.sqrt(1155/2/TPI)*(torch.sin(theta)**2)*(3*(torch.cos(theta)**3)-torch.cos(theta))*torch.cos(2*phi)
        components[...,29] = 1/16.*torch.sqrt(165/2/TPI)*torch.sin(theta)*(21*torch.cos(theta)**4-14*torch.cos(theta)**2+1)*torch.cos(phi)
        components[...,30] = 1/16.*torch.sqrt(11/TPI)*(63*torch.cos(theta)**5-70*torch.cos(theta)**3+15*torch.cos(theta))
        
        components[...,31] = 3/32.*torch.sqrt(77/TPI)*(torch.sin(theta)**5)*torch.sin(5*phi)
        components[...,32] = 3/16.*torch.sqrt(385/2/TPI)*(torch.sin(theta)**4)*torch.cos(theta)*torch.sin(4*phi)
        components[...,33] = 1/32.*torch.sqrt(385/TPI)*(torch.sin(theta)**3)*(9*(torch.cos(theta)**2)-1)*torch.sin(3*phi)
        components[...,34] = 1/8.*torch.sqrt(1155/2/TPI)*(torch.sin(theta)**2)*(3*(torch.cos(theta)**3)-torch.cos(theta))*torch.sin(2*phi)
        components[...,35] = 1/16.*torch.sqrt(165/2/TPI)*torch.sin(theta)*(21*torch.cos(theta)**4-14*torch.cos(theta)**2+1)*torch.sin(phi)
    if L>=6:
        raise NotImplementedError
    return components


def sci_sph_harm_oneside(l, phi, theta):
    coeff = []
    for n in range(l + 1):
        real_coeff = []
        imag_coeff = []

        for m in range(-n, 1):
            temp = sph_harm(m, n, phi.detach().cpu(), theta.detach().cpu())
            real = torch.real(temp).to(torch.float)
            imag = torch.imag(temp).to(torch.float)

            real_coeff.append(real)
            if torch.all(imag<1e-7):
                pass
            else:
                imag_coeff.append(imag)

        real_coeff = torch.concat(real_coeff, dim=1)
        if len(imag_coeff)>0:
            imag_coeff = torch.concat(imag_coeff, dim=1)
            temp_coeff = torch.concat([real_coeff, imag_coeff], dim=1)
        else:
            temp_coeff = real_coeff
        coeff.append(temp_coeff)

    coeff = torch.concat(coeff, dim=1)
    return coeff