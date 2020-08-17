from quantum import commutator
from jax import vmap,jit
import numpy as np
import jax.numpy as jnp
from scipy import linalg
from operator import itemgetter
import time
import sys
from numba import njit, prange
import jit_funcs
from memory_profiler import profile

def mk_W(lat, grad = True):
    #bsij(tcv)
    nv, nc, nf = lat.nv, lat.nc, lat.nf
    nb = nv + nc
    num_d_basis = lat.num_d_basis
    Fock = jit(vmap(vmap(lat.interaction, (0, None, None)), (0, 0, None))) #st
    Hartree = jit(vmap(lat.interaction, (None, 0, 0)))  #s
    if grad:
        dim = lat.dim + 1
        Fock_grad = jit(vmap(vmap(lat.interaction_grad, (0, None, None), 1), (0, 0, None), 1)) #ast
    else:
        dim = 1
    
    d_qtm_nks = np.asarray(lat.d_qtm_array[:, lat.dim:], order='C')
    delta_ks = np.asarray(lat.d_qtm_array[:, None, 0:lat.dim] - lat.d_qtm_array[None,:, 0:lat.dim], order='C') #sta
    wv_funcs = np.transpose(lat.wv_funcs[:,:, nf - nv:nf + nc], axes=(0,2,1)).reshape((num_d_basis*nb,-1))  #sni->sin->(si)n
    W = np.zeros((dim, num_d_basis, nb, nb, num_d_basis, nc, nv), dtype=np.complex64)
    fock = np.empty((dim, num_d_basis, num_d_basis), dtype=np.float32)

    for idx,G in zip(lat.overlape_idx, lat.G_array):
        overlape = (wv_funcs[:, idx[0]].conj() @ wv_funcs[:, idx[1]].T).reshape((num_d_basis, nb, num_d_basis, nb))  #(si)n,(tj)n->(si)(tj)->sitj
        qs = np.copy(delta_ks)
        jit_funcs.add_G(qs, G)
        fock[0] = Fock(qs, d_qtm_nks, d_qtm_nks)
        hartree = np.asarray(Hartree(G, d_qtm_nks, d_qtm_nks))
        if grad:
            fock[1:] = Fock_grad(qs, d_qtm_nks, d_qtm_nks)
        jit_funcs.W_update(W, overlape, fock, hartree)
    W = W.reshape((dim, num_d_basis, nb, nb, -1))
    print('W made')
    return W
    
def bse_solve(lat, W, couple=False, op_cv=None):
    #W: bsij(tcv)
    nv, nc = lat.nv, lat.nc
    num_d_basis = lat.num_d_basis
    Dim = num_d_basis * nc * nv
    W_cvcv = W[0,:, nv:nv + nc, 0:nv].reshape((Dim, Dim))
    diag = np.array(lat.epsilon[:, nv:nv + nc, 0:nv], dtype=np.complex64).ravel()
    H_eh = np.diag(diag)
    H_eh -= W_cvcv
    if couple:
        W_vccv = np.transpose(W[0,:, 0:nv, nv:nv + nc], axes=(0, 2, 1, 3))
        K = W_vccv.reshape((Dim, Dim))
        H = np.append(np.append(H_eh, K.conj(), axis=1), np.append(-K, -H_eh.conj(), axis=1), axis=0)
        eigs, psi = linalg.eig(H, overwrite_a=True)
        shape = W.shape
        W_exc = (W.reshape(-1, Dim) @ psi[0:Dim]).reshape(shape[0:-1] + (-1,)) #bsij(tcv),(tcv)n->bsijn
    else:
        eigs, psi = linalg.eigh(H_eh, overwrite_a=True, driver='evr')
        shape = W.shape
        W_exc = (W.reshape(-1, Dim) @ psi).reshape(shape[0:-1] + (-1,))  #(bsij)(tcv),(tcv)n->(bsij)n->bsijn
    if op_cv is None:
        return eigs, W_exc, psi
    else:
        op_exc = to_exciton_config(op_cv, psi, couple)
        return eigs, W_exc, op_exc

def to_exciton_config(op_cv, psi, couple):
    if couple:
        op_exc = linalg.solve(psi, np.append(op_cv, op_cv.conj(), axis=0))  #n(tcv+tvc),(tcv+tvc)a->na
    else:
        op_exc = psi.T.conj() @ op_cv  #n(tcv),(tcv)a->na
    return op_exc

def cal_t_mat(omega, op_exc, W_exc, eigs):
    omega_p, omega_m = omega, -np.conj(omega)
    l=len(omega_p)
    Omega = np.asarray([omega_p, omega_m]).T.ravel()
    h = op_exc[:,:, None] / (Omega - eigs[:, None, None]) #nao
    
    shape = W_exc.shape
    t_pm = -np.asarray(np.transpose(np.matmul(W_exc.reshape(-1, shape[-1]), h.reshape(shape[-1], -1), order='F').reshape(shape[0:-1] + (-1, l, 2)), axes=(0, 4, 1, 2, 3, 5, 6)), order='F') #bsijn,nao->bsijao->basijo+
    return t_pm

def renormalize(t_pm, op_bare, f=1, grad=False, r_bare=None, op_bare_gd=None):
    if grad & ((len(t_pm) == 1) | (r_bare is None) | (op_bare_gd is None)):
        print('not given enough data to calculated op_rm_gd, only op_rm will be returned')
        grad = False
    temp = np.copy(t_pm, order='K')
    temp *= f[:,:,None, None]
    t = np.transpose(temp[..., 0], axes=(5, 0, 1, 2, 3, 4))  #basijo->obasij
    t += np.transpose(temp[..., 1], axes=(5, 0, 1, 2, 4, 3)).conj()
    if grad:
        for t_o in t:
            t_o[1:] += 1j * commutator(r_bare[:, None], t_o[0])
            t_o[0] += op_bare
            t_o[1:] += op_bare_gd
        return t[:,0], t[:,1:]
    else:
        t[:,0] += op_bare
        return t[:,0]


