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
    #[task(idx,G) for idx,G in zip(lat.overlape_idx, lat.G_array)]
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
        eigs, psi = linalg.eig(H)
        W_exc = np.tensordot(W, psi[0:Dim], (4, 0))  #bsij(tcv),(tcv)n->bsijn
    else:
        eigs, psi = np.linalg.eigh(H_eh)
        W_exc = np.tensordot(W, psi, (4, 0))  #(bsij)(tcv),(tcv)n->(bsij)n->bsijn
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

def renormalize(omega, op_bare, W_exc, eigs, op_exc, f=1, grad=False, r_bare=None, op_bare_gd=None):
    omega_p, omega_m = omega, -np.conj(omega)
    l=len(omega_p)
    Omega = np.array([omega_p, omega_m]).ravel()
    h = op_exc[:,:, None] / (Omega - eigs[:, None, None]) #nao
    if grad & ((len(W_exc) == 1) | (r_bare is None) | (op_bare_gd is None)):
        print('not given enough data to calculated op_rm_gd, only op_rm will be returned')
        grad = False
    if grad:
        t_pm = -np.transpose(np.array(np.transpose(np.tensordot(W_exc, h, (4, 0)), axes=(0, 4, 1, 2, 3, 5)), order='F'), axes=(5,0,1,2,3,4))#bsijn,nao->bsijao->basijo->obasij
        t_pm *= f
        t = t_pm[0:l]
        t += np.transpose(t_pm[l:], axes=(0, 1, 2, 3, 5, 4)).conj()
        for t_o in t:
            t_o[1:] += 1j * commutator(r_bare[:, None], t_o[0])
            t_o[0] += op_bare
            t_o[1:] += op_bare_gd
        #t[:, 1:] += 1j *np.asarray([ for o in range(l)])
        #t[:, 0] += op_bare
        #t[:,1:] += op_bare_gd
        return t[:,0], t[:,1:]
    else:
        t_pm = -np.transpose(np.array(np.transpose(np.tensordot(W_exc[0], h, (3, 0)), axes=(3, 0, 1, 2, 4)), order='F'), axes=(4, 0, 1, 2, 3))  #sijn,nao->sijao->asijo->oasij
        t_pm *= f
        t = t_pm[0:l]
        t += np.transpose(t_pm[l:], axes=(0, 1, 2, 4, 3)).conj()
        t += op_bare
        return t


