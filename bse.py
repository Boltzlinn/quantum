from quantum import commutator
from jax import vmap,jit
import numpy as np
from scipy import linalg
from operator import itemgetter
import time
import sys
from numba import njit, prange

def mk_W(lat, grad = True):
    #bsij(tcv)
    nv, nc, nf = lat.nv, lat.nc, lat.nf
    nb = nv + nc
    num_d_basis = lat.num_d_basis
    if grad:
        dim = lat.dim + 1
        Fock_grad = jit(vmap(vmap(lat.interaction_grad, (0, None, None), 1), (0, 0, None), 1)) #ast
    else:
        dim = 1
    d_qtm_nks = lat.d_qtm_array[:, lat.dim:]
    delta_d_qtms = lat.d_qtm_array[:, None,:] - lat.d_qtm_array[None,:,:]
    delta_ks = delta_d_qtms[:,:, 0:lat.dim]
    #wv_funcs = lat.wv_funcs[:,:, nf - nv:nf + nc]  #sni
    wv_funcs = np.transpose(lat.wv_funcs[:,:, nf - nv:nf + nc], axes=(0,2,1)).reshape((num_d_basis*nb,-1))  #sni->sin->(si,n)
    W = np.zeros((dim, num_d_basis, nb, nb, num_d_basis, nc, nv), dtype=np.complex64)
    Fock = jit(vmap(vmap(lat.interaction, (0, None, None)), (0, 0, None))) #st
    Hartree = jit(vmap(lat.interaction, (None, 0, 0)))  #s
    
    @njit(parallel=True)
    def overlape_to_fock(overlape, out):
        S,I,J,T,C,V = out.shape
        for s in prange(S): 
            for i in range(I): 
                for j in range(J): 
                    for t in prange(T): 
                        for c in range(C): 
                            for v in range(V): 
                                out[s, i, j, t, c, v] = overlape[s, i, t, c + V] * overlape[s, j, t, v].conjugate()
    @njit(parallel=True)
    def overlape_to_hartree(overlape, out):
        S,I,J,T,C,V = out.shape
        for s in prange(S): 
            for i in range(I): 
                for j in range(J): 
                    for t in prange(T): 
                        for c in range(C): 
                            for v in range(V): 
                                out[s, i, j, t, c, v] = overlape[s, i, s, j] * overlape[t, c + V, t, v].conjugate()
    
    @njit(parallel=True)
    def add_fock(W, fock, temp):
        S,I,J,T,C,V = W.shape
        for s in prange(S): 
            for i in range(I): 
                for j in range(J): 
                    for t in prange(T): 
                        for c in range(C): 
                            for v in range(V):
                                W[s, i, j, t, c, v] += (fock[s, t] * temp[s, i, j, t, c, v])
    
    @njit(parallel=True)
    def add_fock_grad(W, fock, temp):
        A,S,I,J,T,C,V = W.shape
        for s in prange(S): 
            for i in range(I): 
                for j in range(J): 
                    for t in prange(T): 
                        for c in range(C): 
                            for v in range(V):
                                for a in range(A):
                                    W[a ,s, i, j, t, c, v] += (fock[a, s, t] * temp[s, i, j, t, c, v])
    

    @njit(parallel=True)
    def minus_hartree(W, hartree, temp):
        S,I,J,T,C,V = W.shape
        for s in prange(S): 
            for i in range(I): 
                for j in range(J): 
                    for t in prange(T): 
                        for c in range(C): 
                            for v in range(V):
                                W[s, i, j, t, c, v] -= (hartree[s] * temp[s, i, j, t, c, v])
    
    @njit(parallel=True)
    def add_G(qs, G):
        S,T,A=qs.shape
        for s in prange(S):
            for t in prange(T):
                for a in range(A):
                    qs[s, t, a] += G[a]
    
    temp = np.empty((num_d_basis, nb, nb, num_d_basis, nc, nv),dtype=np.complex64)
    def task(idx, G):
        overlape = (wv_funcs[:, idx[0]].conj() @ wv_funcs[:, idx[1]].T).reshape((num_d_basis, nb, num_d_basis, nb))  #(si)n,(tj)n->(si)(tj)->sitj
        #overlape = np.tensordot(wv_funcs[:, idx[0]], wv_funcs[:, idx[1]], (1,1))
        overlape_to_fock(overlape, temp)
        #temp = overlape[:,:, None,:, nv:nv + nc, None] * overlape[:, None,:,:, None, 0:nv].conj()  #sijtcv
        qs = np.array(delta_ks)
        add_G(qs, G)
        add_fock(W[0], np.array(Fock(qs, d_qtm_nks, d_qtm_nks), dtype=np.float32), temp)
        #W[0] += np.array(Fock(qs, d_qtm_nks, d_qtm_nks))[:, None, None,:, None, None] * temp
        if grad:
            add_fock_grad(W[1:], np.array(Fock_grad(qs, d_qtm_nks, d_qtm_nks), dtype=np.float32), temp)
            #W[1:] += np.array(Fock_grad(qs, d_qtm_nks, d_qtm_nks))[:,:, None, None,:, None, None] * temp
        #overlape_diag = np.array([overlape[s,:, s,:] for s in range(num_d_basis)])  #sij
        #temp = overlape_diag[:,:,:, None, None, None] * overlape_diag[None, None, None,:, nv:nv + nc, 0:nv].conj()
        overlape_to_hartree(overlape, temp)
        minus_hartree(W[0], np.array(Hartree(G, d_qtm_nks, d_qtm_nks), dtype=np.float32), temp)
        #W[0] -= np.array(Hartree(G, d_qtm_nks, d_qtm_nks))[:, None, None, None, None, None] * temp
    [task(idx,G) for idx,G in zip(lat.overlape_idx, lat.G_array)]
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
    h_p = op_exc / (omega_p - eigs[:, None]) #na
    h_m = op_exc / (omega_m - eigs[:, None])
    if grad & ((len(W_exc) == 1) | (r_bare is None) | (op_bare_gd is None)):
        print('not given enough data to calculated op_rm_gd, only op_rm will be returned')
        grad = False
    if grad:
        def t_val_gd(h):
            t = -np.transpose(np.tensordot(W_exc, h, (4, 0)), axes=(0, 4, 1, 2, 3))  #bsijn,na->bsija->basij
            t *= f
            t[1:] += 1j * commutator(r_bare[:, None], t[0])
            return t
        t = t_val_gd(h_p)
        t += np.transpose(t_val_gd(h_m), axes=(0, 1, 2, 4, 3)).conj()
        return t[0] + op_bare, t[1:] + op_bare_gd
    else:
        def t_val(h):
            t = -np.transpose(np.tensordot(W_exc[0], h, (3, 0)), axes=(3, 0, 1, 2))
            t *= f
            return t
        t = t_val(h_p)
        t += np.transpose(t_val(h_m), axes=(0, 1, 3, 2)).conj()
        return t + op_bare


