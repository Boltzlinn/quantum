import jit_funcs
from quantum import commutator
import bse
import numpy as np
import matplotlib.pyplot as plt

#for k-diagnoal probe and perturbs
def response(epsilon, rho_0, probe, *perturbs):
    omega = 0
    c = rho_0
    for o, a in perturbs:
        omega += o
        c = commutator(a,c) / (omega - epsilon)
    return np.sum(probe.T * c)

def shg_lg(omega, r, r_gd, r2, epsilon, epsilon_gd, f_ij):
    o_m_e = omega - epsilon
    h = r / o_m_e  #asij
    g = h * (f_ij.T) #asij
    g_gd = (r_gd * (f_ij.T) + epsilon_gd[:, None] * g) / o_m_e
    h2 = r2 / (-2.*omega - epsilon)  #asij
    res = np.tensordot(h2, commutator(r[:, None], g) + 1j * g_gd, ((1, 2, 3), (2, 4, 3))) + 1j / 2. * np.tensordot(g_gd, h, ((2, 3, 4), (1, 3, 2)))
    return -res
    
def SHG(lat, Omega, nv, nf, nc, eta=0.05, couple=False):
    lat.nv = nv
    lat.nf = nf
    lat.nc = nc
    Omega_p = Omega + 1j * eta
    
    lat.mk_hamiltonian()
    lat.solve()
    lat.mk_r_bare(nv, nf, nc)
    lat.mk_overlape_idx()

    r_bare=lat.r_bare
    r_bare_gd=lat.r_bare_gd
    epsilon=lat.epsilon
    epsilon_gd=lat.epsilon_gd
    
    W = bse.mk_W(lat)
    
    eigs, W_exc, r_abre_exc = bse.bse_solve(lat, W, couple=couple, op_cv=lat.r_bare_cv)
    print('bse solved')
    rho = np.array(np.where(np.arange(nc + nv) < nv, 1., 0.), dtype=np.float32)
    f_ij=rho[:, None] - rho
    f_E=np.multiply(rho[None,:], (1 - rho)[:, None], order='F')
    f_A=1
    if couple == False: f_A = 1 - f_E.T
    
    '''
    ipa = []
    exc = []
    aug=[]
    '''
    
    ipa=np.array([shg_lg(omega_p, r_bare, r_bare_gd, r_bare, epsilon, epsilon_gd, f_ij) for omega_p in Omega_p])
    print('ipa')

    R_E, R_E_gd = bse.renormalize(Omega_p, r_bare, W_exc, eigs, r_abre_exc, f_E, grad=True, r_bare=r_bare, op_bare_gd=r_bare_gd)
    R2_E=bse.renormalize(-2 * Omega_p, r_bare, W_exc, eigs, r_abre_exc, f_E)
    exc=np.array([shg_lg(omega_p, r_E, r_E_gd, r2_E, epsilon, epsilon_gd, f_ij) for omega_p, r_E, r_E_gd, r2_E in zip(Omega_p, R_E,R_E_gd,R2_E) ])
    print('exc')
    
    R_A, R_A_gd = bse.renormalize(Omega_p, r_bare, W_exc, eigs, r_abre_exc, f_A, grad=True, r_bare=r_bare, op_bare_gd=r_bare_gd)
    R2_A = bse.renormalize(-2 * Omega_p, r_bare, W_exc, eigs, r_abre_exc, f_A)
    aug=np.array([shg_lg(omega_p, r_A, r_A_gd, r2_A, epsilon, epsilon_gd, f_ij) for omega_p, r_A, r_A_gd, r2_A in zip(Omega_p, R_A,R_A_gd,R2_A)])
    print('aug')
    
    '''
    for omega in Omega_p:
        temp_ipa = shg_lg(omega, r_bare, r_bare_gd, r_bare, epsilon, epsilon_gd, f_ij)
        ipa += [temp_ipa]

        r_E, r_E_gd = bse.renormalize(omega, r_bare, W_exc, eigs, r_abre_exc, f_E, grad=True, r_bare=r_bare, op_bare_gd=r_bare_gd)
        r2_E = bse.renormalize(-2 * omega, r_bare, W_exc, eigs, r_abre_exc, f_E)
        temp_exc = shg_lg(omega, r_E, r_E_gd, r2_E, epsilon, epsilon_gd, f_ij)
        exc += [temp_exc]

        r_A, r_A_gd = bse.renormalize(omega, r_bare, W_exc, eigs, r_abre_exc, f_A, grad=True, r_bare=r_bare, op_bare_gd=r_bare_gd)
        r2_A = bse.renormalize(-2 * omega, r_bare, W_exc, eigs, r_abre_exc, f_A)
        temp_aug = shg_lg(omega, r_A, r_A_gd, r2_A, epsilon, epsilon_gd, f_ij)
        aug += [temp_aug]
    '''
    ipa = np.array(ipa)/lat.volume
    exc = np.array(exc)/lat.volume
    aug = np.array(aug)/lat.volume
    data = np.array([ipa, exc, aug]).ravel()
    name = lat.name + '_' + str(lat.real_shape)
    np.savetxt(name + '.txt',data)
    plt.plot(Omega, np.abs(ipa[:,0, 0, 0]) * Omega, label = 'ipa')
    plt.plot(Omega, np.abs(exc[:,0, 0, 0]) * Omega, label = 'exc')
    plt.plot(Omega, np.abs(aug[:,0, 0, 0]) * Omega, label='aug')
    plt.legend()
    plt.savefig(name + '.pdf')
    #plt.show()