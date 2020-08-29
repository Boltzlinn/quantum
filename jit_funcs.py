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
