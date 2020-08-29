import jit_funcs
from quantum import commutator
import bse
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile

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
    return - res

def shg_vg(omega, vel, vel2, epsilon, f_ij):
    h = vel / (omega - epsilon)
    g = h * (f_ij.T)
    h2 = vel2 / (-2 * omega - epsilon)
    res = np.tensordot(h2, commutator(vel[:, None], g), ((1, 2, 3), (2, 4, 3)))
    return -res
  
def SHG(lat, Omega, nv, nf, nc, eta=0.05, gauge='l', scheme='iea', couple=False):
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
    epsilon_gd = lat.epsilon_gd
    vel_bare = lat.vel_bare
    
    rho = np.array(np.where(np.arange(nc + nv) < nv, 1., 0.), dtype=np.float32)
    f_ij=rho[:, None] - rho

    flag = False
    for s in scheme:
        if (s == 'e') | (s == 'a'):
            flag = True
            break
    
    if flag:
        if gauge == 'l':
            W = bse.mk_W(lat)
            eigs, W, r_bare_exc = bse.bse_solve(lat, W, couple=couple, op_cv=lat.r_bare_cv)
            t_pm = bse.cal_t_mat(Omega_p, r_bare_exc, W, eigs)
            t2_pm = bse.cal_t_mat(-2 * Omega_p, r_bare_exc, W[0][None], eigs)
            eigs, W, r_bare_exc = None, None, None
        elif gauge == 'v':
            W = bse.mk_W(lat, grad=False)
            eigs, W, vel_bare_exc = bse.bse_solve(lat, W, couple=couple, op_cv=lat.vel_bare_cv)
            t_pm = bse.cal_t_mat(Omega_p, vel_bare_exc, W[0][None], eigs)
            t2_pm = bse.cal_t_mat(-2 * Omega_p, vel_bare_exc, W[0][None], eigs)
            eigs, W, vel_bare_exc = None, None, None
    
    data=[]
    for s in scheme:
        if s == 'i':
            if gauge == 'l':
                res = np.array([shg_lg(omega_p, r_bare, r_bare_gd, r_bare, epsilon, epsilon_gd, f_ij) for omega_p in Omega_p])
            elif gauge == 'v':
                res = np.array([shg_vg(omega_p, vel_bare, vel_bare, epsilon, f_ij) for omega_p in Omega_p])
            label = 'ipa'
        else:
            if s == 'e':
                f = np.multiply(rho[None,:], (1 - rho)[:, None], order='F')
                if couple == True: f += f.T
                label = 'exc'
            if s == 'a':
                f=1
                if couple == False: f = 1 - np.multiply(rho[:, None], (1 - rho)[None,:], order='F')
                label = 'aug'
            if gauge == 'l': 
                R, R_gd = bse.renormalize(t_pm, r_bare, f, grad=True, r_bare=r_bare, op_bare_gd=r_bare_gd)
                R2 = bse.renormalize(t2_pm, r_bare, f)
                res = np.array([shg_lg(omega_p, r, r_gd, r2, epsilon, epsilon_gd, f_ij) for omega_p, r, r_gd, r2 in zip(Omega_p, R, R_gd, R2)])
            elif gauge == 'v':
                Vel = bse.renormalize(t_pm, vel_bare, f)
                Vel2 = bse.renormalize(t2_pm, vel_bare, f)
                res = np.array([shg_vg(omega_p, vel, vel2, epsilon, f_ij) for omega_p, vel, vel2 in zip(Omega_p, Vel, Vel2)])
        res /= lat.volume
        data += [res]
        if gauge == 'l':
            plt.plot(Omega, np.abs(res[:, 0, 0, 0]*Omega*2), label=label)
        if gauge == 'v':
            plt.plot(Omega, np.imag(res[:, 0, 0, 0] / Omega / Omega), label=label)
        print(label)
    
    data = np.array(data)
    name = lat.name + '_' + str(lat.real_shape[0]) + '_' + str(nc+nv) + '_' + gauge + '_' + scheme
    header = 'original data shape is ' + str(data.shape)
    np.savetxt(name + '.txt', data.ravel(), header=header)    
    
    plt.legend()
    plt.savefig(name + '.pdf')
    #plt.show()