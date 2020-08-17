from numba import njit, prange
import numpy as np

@njit(parallel=True)
def add_G(qs, G):
    S,T,A=qs.shape
    for s in prange(S):
        for t in range(T):
            for a in range(A):
                qs[s, t, a] += G[a]
    
@njit(parallel=True,fastmath=True)
def W_update(W, overlape, fock, hartree):
    A, S, I, J, T, C, V = W.shape
    for s in prange(S):
        for t in range(T):
            for v in range(V): 
                for c in range(C):
                    cv_conj = overlape[t, c + V, t, v].conjugate()
                    for j in range(J):
                        jv_conj = overlape[s, j, t, v].conjugate() 
                        for i in range(I):
                            W[0, s, i, j, t, c, v] -= overlape[s, i, s, j] * cv_conj * hartree[s]
                            for a in range(0, A):
                                W[a, s, i, j, t, c, v] += overlape[s, i, t, c + V] * jv_conj * fock[a, s, t]

@njit(parallel=True)
def get_shg_lg(h, h2, commu, g_gd, res):
    O, C, B, A = res.shape
    A, O, S, I, J = h.shape
    for o in prange(O):
        for c in range(C):
            for b in range(B):
                for a in range(A):
                    for s in prange(S):
                        for i in range(I):
                            for j in range(J):
                                res[o, c, b, a] += ((h2[c, o, s, i, j] * (commu[b, a, o, s, j, i] + 1j * g_gd[b, a, o, s, j, i])) + 1j / 2.*g_gd[c,b,o,s,i,j]*h[a,o,s,j,i])

@njit(parallel=True)                               
def get_t(W_exc, eigs, omega, op_exc, t):
    B, S, I, J, N = W_exc.shape
    N, A = op_exc.shape
    O = len(omega)
    for o in prange(O):
        for a in range(A):
            h = op_exc[:, a] / (omega[o] - eigs)
            for s in prange(S):
                for b in range(B):
                    for i in range(I):
                        for j in range(J):
                            t[b, a, o, s, i, j] -= np.dot(W_exc[b, s, i, j], h)
    return t