from quantum import Quantum, sigma, p_mat
import jax.numpy as jnp
from jax import grad, jacfwd, hessian, jit, vmap, custom_jvp
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from functools import partial
from timeit import timeit
from operator import itemgetter
import time
import sys


class Lattice(Quantum):

    def __init__(self, dim, avec, bvec, real_shape=3, kstart=0., bz_shape=5, special_pts={}, ifspin=0, ifmagnetic=0, hoppings=(), num_sub=1):
        self.dim = dim
        self.avec = avec
        self.bvec = bvec
        self.special_pts = special_pts
        self.diag_qtm_nums = {}
        self.ndiag_qtm_nums = {}
        self.hamiltonian = {}
        self.G_list = {}
        self.V = {}
        if np.sum(real_shape) <= 0:
            print('real_shape can not be 0 or negative, real_shape is set to 1 instead')
            real_shape = 1
        self.real_shape = np.zeros(dim, dtype=np.int32)+real_shape
        self.kstart = np.zeros(dim)+kstart
        if np.sum(bz_shape) <= 0:
            print('bz_shape can not be 0 or negative, bz_shape is set to 5 instead')
            bz_shape = 5
        self.bz_shape = np.zeros(dim, dtype=np.int32) + bz_shape
        self.hoppings = hoppings
        self.num_sub = num_sub
        klabels = ('kx', 'ky', 'kz')
        Glabels = ('Gz', 'Gy', 'Gz')
        for i in range(dim):
            self.diag_qtm_nums[klabels[i]] = [
                self.kstart[i] + n / self.real_shape[i] for n in range(self.real_shape[i])]
            self.ndiag_qtm_nums[Glabels[i]] = [
                n - self.bz_shape[i] // 2 for n in range(self.bz_shape[i])]
        if ifspin:
            if ifmagnetic:
                self.ndiag_qtm_nums['spin'] = [-0.5, 0.5]
            else:
                self.diag_qtm_nums['spin'] = [-0.5, 0.5]
    
    @staticmethod
    def kinetic(k, d_qtm_nk, nd_qtm):
        print("kinetic not set!")
        pass

    @staticmethod
    def ext_potential(d_qtm_nk, delta_nd_qtm):
        print("external potential not set")
        pass

    @staticmethod
    def interaction(q, d_qtm_nk1, d_qtm_nk2):
        print("interaction not set")
        pass
    
    @staticmethod
    def separate(container, idx):
        l = len(container)
        if idx > l:
            part1 = container
            part2 = ()
            print("index larger than container length, the second part is set ()")
        elif idx == l:
            part1 = container
            part2 = ()
        else:
            part1 = container[0:idx]
            part2 = container[idx:]
        return part1, part2

    @staticmethod
    def vec2arr(vec):
        return vec[:, None] - vec[None,:]

    @partial(jit, static_argnums=(0))
    def mk_V(self):
        d_qtm_list = list(self.Basis.keys())
        d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[1],d_qtm_list)))

        basis = self.nd_basis
        num_sub = self.num_sub
        V = {d_qtm_nk: np.zeros((self.num_nd_bands, self.num_nd_bands), dtype=np.complex64) for d_qtm_nk in d_qtm_nk_list}
        for delta_nd_qtm in self.hoppings:
            # self.hopping could be tuple, list, etc. but not a generator
            for nd_qtm1 in basis:
                i = basis.get(nd_qtm1)
                nd_qtm2 = self.qtm_minus(nd_qtm1, delta_nd_qtm)
                j = self.nd_basis.get(nd_qtm2)
                if j != None:
                    for d_qtm_nk in V:
                        V[d_qtm_nk][i * num_sub:(i + 1) * num_sub, j * num_sub:(j + 1) * num_sub] = self.ext_potential(d_qtm_nk, delta_nd_qtm)
        self.V = V
    
    @partial(jit, static_argnums=(0, 2))
    def hm(self, k, d_qtm_nk):
        data = tuple(map(lambda nd_qtm: self.kinetic(k, d_qtm_nk, nd_qtm), list(self.nd_basis.keys())))
        diag = jnp.kron(jnp.eye(self.num_nd_basis, dtype=jnp.complex64), jnp.ones((self.num_sub, self.num_sub), dtype=jnp.complex64))
        T = jnp.repeat(jnp.hstack(data).reshape((1, -1)), self.num_nd_basis, axis=0).reshape((self.num_nd_bands, self.num_nd_bands)) * diag
        return jnp.asarray(T + self.V[d_qtm_nk])

    def mk_hamiltonian(self):
        def task(d_qtm):
            k, d_qtm_nk = self.separate(d_qtm, self.dim)
            return self.hm(np.array(k), d_qtm_nk)
        hamiltonian = np.array([task(d_qtm) for d_qtm in self.Basis])
        self.hamiltonian = np.array(hamiltonian)
        print('non-interacting hamiltonian made')

    #@partial(jit, static_argnums=(0))
    def mk_r_bare(self):
        nv, nc, nf = self.nv, self.nc, self.nf
        def task(d_qtm):
            k, d_qtm_nk = self.separate(d_qtm, self.dim)
            return self.vel_opt(np.array(k), d_qtm_nk)
        vel_plane_wave = np.array([task(d_qtm) for d_qtm in self.Basis])
        vel_plane_wave = np.transpose(vel_plane_wave, axes=(3,0,1,2))
        vel = np.transpose(self.wv_funcs.conj(), axes=(0,2,1)) @ vel_plane_wave @ self.wv_funcs #asij
        #vel = jnp.einsum('slk,slma,smn->askn', self.wv_funcs.conj(), vel_plane_wave, self.wv_funcs)
        epsilon = self.energies[:,:, None] - self.energies[:, None]
        epsilon_inv = np.array(1./(epsilon + np.diag(np.inf * np.ones((self.num_nd_bands)))), dtype=np.float32)
        r_bare = -1j * vel * epsilon_inv
        energies_grad=np.einsum('asii->asi', vel)
        epsilon_gd=energies_grad[:,:,:, None] - energies_grad[:,:, None,:]  #asij
        r_bare_gd=(r_bare[:, None] @ vel - vel @ r_bare[:, None] - epsilon_gd[:, None] * r_bare) * epsilon_inv
        r_bare=np.array(np.transpose(r_bare[:,:, nf - nv:nf + nc, nf - nv:nf + nc], axes=(1, 2, 3, 0)), order='C')
        r_cv_bare=np.array(r_bare[:, nv:nv + nc, 0:nv].reshape((-1, self.dim)), order='C')
        self.r_bare=np.transpose(r_bare, axes=(3, 0, 1, 2))
        self.r_bare_gd=r_bare_gd[:,:,:, nf - nv:nf + nc, nf - nv:nf + nc]
        self.epsilon=epsilon[:, nf - nv:nf + nc, nf - nv:nf + nc]
        self.epsilon_gd=epsilon_gd[:,:, nf - nv:nf + nc, nf - nv:nf + nc]
        self.r_cv_bare=r_cv_bare
        print("r_bare solved")

    #@partial(jit, static_argnums=(0))
    def mk_overlape_idx(self):
        num_sub = self.num_sub
        eye = np.eye(self.num_nd_bands, dtype=np.int32)
        def task(G):
            idx1 = []
            idx2 = []
            for nd_qtm1 in self.nd_basis:
                i = self.nd_basis[nd_qtm1]
                l = len(nd_qtm1) - self.dim
                delta_nd_qtm = G + tuple((0 for i in range(l)))
                nd_qtm2 = self.qtm_minus(nd_qtm1, delta_nd_qtm)
                j = self.nd_basis.get(nd_qtm2)
                if j != None:
                    idx1 += list(range(i * num_sub, (i + 1) * num_sub))
                    idx2 += list(range(j * num_sub, (j + 1) * num_sub))
            return [idx1, idx2]
        self.overlape_idx = np.array([task(G) for G in self.G_list])
        self.overlape_Idx = np.array([eye[idx[0]].T @ eye[idx[1]] for idx in self.overlape_idx])
        print("overlape index made")

    @partial(jit, static_argnums=(0))
    def hartree(self, d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2):
        interaction = vmap(partial(self.interaction, d_qtm_nk1 = d_qtm_nk1, d_qtm_nk2 = d_qtm_nk1))(self.G_array)
        overlape_1 = jnp.einsum('ei,gef,fj->gij', wv_func1.conj(), self.overlape_Idx, wv_func1)
        overlape_2 = jnp.einsum('ei,gef,fj->gij', wv_func2.conj(), self.overlape_Idx, wv_func2)
        res = jnp.einsum('g,gij,gkl->ijkl', interaction, overlape_1, overlape_2.conj())
        return res
    
    @partial(jit, static_argnums=(0))
    def fock(self, delta_k, d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2):
        q_array = delta_k + self.G_array
        interaction = vmap(partial(self.interaction, d_qtm_nk1 = d_qtm_nk1, d_qtm_nk2 = d_qtm_nk2))(q_array)
        interaction_grad = vmap(partial(self.interaction_grad, d_qtm_nk1 = d_qtm_nk1, d_qtm_nk2 = d_qtm_nk2))(q_array)
        interaction_val_grad = jnp.append(interaction.reshape((-1,1)), interaction_grad, axis=1)
        overlape = jnp.einsum('ei,gef,fj->gij', wv_func1.conj(), self.overlape_Idx, wv_func2)
        res_val_grad = jnp.einsum('ga,gik,gjl->aijkl', interaction_val_grad, overlape, overlape.conj())
        return res_val_grad

    @partial(jit, static_argnums=(0))
    def W(self, delta_k, d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2):
        hartree = self.hartree(d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2)
        fock_val_grad = self.fock(delta_k, d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2)
        res = fock_val_grad - hartree[None,:,:,:,:] * jnp.append(0, jnp.ones(self.dim))[:,None, None, None, None]
        return res
    

    @partial(jit, static_argnums=(0))
    def mk_W_p(self):
        #saij(tcv)
        nv, nc, nf = self.nv, self.nc, self.nf
        nb = nv + nc
        num_d_basis = self.num_d_basis
        dim = self.dim
        d_qtm_nks = self.d_qtm_array[:, self.dim:]
        delta_d_qtms = self.d_qtm_array[:, None,:] - self.d_qtm_array[None,:,:]
        delta_ks = delta_d_qtms[:,:, 0:self.dim]
        wv_funcs = np.array(self.wv_funcs[:,:, nf - nv:nf + nc], order='C')  #sni
        W_val_grad = np.zeros((dim + 1, num_d_basis, nb, nb, num_d_basis, nc, nv), dtype=np.complex64)
        Fock = vmap(vmap(self.interaction, (0, None, None)), (0, 0, None)) #st
        Fock_grad = vmap(vmap(self.interaction_grad, (0, None, None), 1), (0, 0, None), 1) #ast
        Hartree = vmap(self.interaction, (None, 0, 0))  #s
        def task(idx, G):
            overlape = np.tensordot(wv_funcs[:,idx[0]].conj(), wv_funcs[:,idx[1]],(1,1)) #sni,tnj->sitj
            temp = overlape[:,:, None,:, nv:nv + nc, None] * overlape[:, None,:,:, None, 0:nv].conj()  #sijtcv
            qs = delta_ks + G
            W_val_grad[0] += np.array(Fock(qs, d_qtm_nks, d_qtm_nks))[:, None, None,:, None, None] * temp
            W_val_grad[1:] += np.array(Fock_grad(qs, d_qtm_nks, d_qtm_nks))[:,:, None, None,:, None, None] * temp
            overlape_diag = np.array([overlape[s,:, s,:] for s in range(num_d_basis)])  #sij
            temp = overlape_diag[:,:,:, None, None, None] * overlape_diag[None, None, None,:, nv:nv + nc, 0:nv].conj()
            W_val_grad[0] -= np.array(Hartree(G, d_qtm_nks, d_qtm_nks))[:, None, None, None, None, None] * temp

        [task(idx, G) for idx, G in zip(self.overlape_idx, self.G_array)]
        self.W_val_grad = W_val_grad.reshape((dim + 1,num_d_basis, nb, nb, -1))
        print(np.shape(self.W_val_grad))
        print("W calculated")


    @partial(jit, static_argnums=(0))
    def mk_W(self):
        nv, nc, nf = self.nv, self.nc, self.nf
        num_d_basis = self.num_d_basis
        Dim = num_d_basis * nc * nv
        d_qtm_nks = self.d_qtm_array[:,self.dim:]
        delta_d_qtms = self.d_qtm_array[:, None,:] - self.d_qtm_array[None,:,:]
        delta_ks = delta_d_qtms[:,:, 0:self.dim]
        wv_funcs = self.wv_funcs[:,:, nf - nv:nf + nc]
        W_val_grad = []
        i = 0
        def task(delta_ki,d_qtm_nki,wv_funci):
            return vmap(self.W, in_axes=(0, None, 0, None, 0), out_axes= 3)(delta_ki, d_qtm_nki, d_qtm_nks, wv_funci, wv_funcs)#aijskl

        def split(n):
            if n * n * (nv + nc)** 4 * 3 < 100000000:
                return [n]
            else:
                return split(n // 2) + split(n - n // 2)
        Idx = split(self.num_d_basis)
        Idx = list(np.cumsum(Idx))
        Idx = [0]+Idx
        Idx = [list(range(Idx[i], Idx[i + 1])) for i in range(len(Idx) - 1)]
        
        print(len(Idx))
        
        W_val_grad = []
        for idx in Idx:
            W_val_grad += [np.array(vmap(task, (0, 0, 0))(delta_ks[idx], d_qtm_nks[idx], wv_funcs[idx])[:,:,:,:,:, nv:nv + nc, 0:nv]).ravel()]#saijtcv
            print([idx[0],idx[-1]])
        W_val_grad = np.hstack(tuple(W_val_grad))
        self.W_val_grad = np.transpose(W_val_grad.ravel().reshape((num_d_basis, self.dim + 1, nv + nc, nv + nc, Dim)), axes=(1, 0, 2, 3, 4)).ravel().reshape((self.dim + 1, num_d_basis, nv + nc, nv + nc, Dim))  #asijm
        print(np.shape(self.W_val_grad))
        print("W calculated")
    
    def bse_solve(self):
        nv, nc = self.nv, self.nc
        num_d_basis = self.num_d_basis
        Dim = num_d_basis * nc * nv
        W_cvcv = self.W_val_grad[0,:, nv:nv + nc, 0:nv].reshape((Dim, Dim))
        print('cvcv')
        diag = np.array(self.epsilon[:, nv:nv + nc, 0:nv], dtype=np.complex64).ravel()
        H_eh = np.diag(diag)
        print('h_eh_0')
        H_eh -= W_cvcv
        print('h_eh')
        W_vccv = np.transpose(self.W_val_grad[0,:, 0:nv, nv:nv + nc], axes=(0, 2, 1, 3))
        print('vccv')
        K = W_vccv.reshape((Dim, Dim))
        print('K')
        H_ehhe = np.append(np.append(H_eh, K.conj(), axis = 1), np.append(-K, -H_eh.conj(), axis = 1), axis = 0)
        print('h_ehhe')
        eigs_E, psi_E = np.linalg.eigh(H_eh)
        print('exciton soloved')
        eigs_A, psi_A = linalg.eig(H_ehhe)
        print('exciton deexciton solved')
        R_E = psi_E.T.conj() @ self.r_cv_bare
        print('R_E solved')
        R_A = linalg.solve(psi_A, np.append(self.r_cv_bare, self.r_cv_bare.conj(), axis=0))
        print('R_A solved')

        self.eigs_E = np.asarray(eigs_E,dtype = np.complex64)
        self.psi_E = np.asarray(psi_E)
        self.R_E = np.asarray(R_E)
        self.eigs_A = np.asarray(eigs_A,dtype = np.complex64)
        self.psi_A = np.asarray(psi_A[:Dim])
        self.R_A = np.asarray(R_A)

    @staticmethod
    #@partial(jit,static_argnums=(1,2))
    def r_renormal(omega, r_bare, r_bare_gd, R, eigs, psi, W_val_grad, f=1, grad=True):
        #r_bare asij
        #W_val_grad bsij(tcv)
        omega_p, omega_m = omega, -np.conj(omega)
        R_p = R / (omega_p - eigs[:, None]) #(scv+svc)a
        h_cv_p = psi @ R_p #(tcv)a
        R_m = R / (omega_m - eigs[:, None])
        h_cv_m = psi @ R_m
        if grad:
            t_p_val_gd = -np.transpose(np.tensordot(W_val_grad, h_cv_p, (4, 0)), axes=(0, 4, 1, 2, 3))  #bsij(tcv),(tcv)a->bsija->basij
            t_p_val_gd *= f
            t_p = t_p_val_gd[0]
            t_p_val_gd[1:] += 1j * (r_bare[:, None] @ t_p - t_p @ r_bare[:, None])  #bsij,asjk->basik
            t_m_val_gd = -np.transpose(np.tensordot(W_val_grad, h_cv_m, (4, 0)), axes=(0, 4, 1, 2, 3))  #basij
            t_m_val_gd *= f
            t_m = t_m_val_gd[0]
            t_m_val_gd[1:] += 1j * (r_bare[:,None] @ t_m - t_m @ r_bare[:,None])
            t_p_val_gd += np.transpose(t_m_val_gd, axes=(0, 1, 2, 4, 3)).conj()
            t_p_val_gd[0] += r_bare
            t_p_val_gd[1:] += r_bare_gd
            return t_p_val_gd[0], t_p_val_gd[1:]
        else:
            t_p = -np.transpose(np.tensordot(W_val_grad[0], h_cv_p, (3, 0)), axes=(3, 0, 1, 2))
            t_p *= f
            t_m = -np.transpose(np.tensordot(W_val_grad[0], h_cv_m, (3, 0)), axes=(3, 0, 1, 2))
            t_m *= f
            t_p += np.transpose(t_m, axes=(0, 1, 3, 2)).conj()
            t_p += r_bare
            return np.array(t_p, order='C')

    @staticmethod
    #@jit
    def shg(omega, r, r_gd, r2, epsilon, epsilon_gd, f):
        h = r / (omega - epsilon)  #asij
        g = h * (f.T) #asij
        g_gd = (r_gd * (f.T) + epsilon_gd[:, None] * g) / (omega - epsilon)
        h2 = r2 / (-2.*omega - epsilon)  #asij
        #res = (np.transpose(h2, axes=(0, 1, 3, 2))[:, None, None]) * ((r[:, None] @ g - g @ r[:, None]) + 1 j * g_gd) + 1 j / 2.*np.transpose(h,axes=(0,1,3,2))[:,None]*g_gd[:,None]
        res = np.tensordot(h2, (r[:, None] @ g - g @ r[:, None]) + 1j * g_gd, ((1, 2, 3), (2, 4, 3))) + 1j / 2. * np.tensordot(g_gd, h, ((2, 3, 4), (1, 3, 2)))
        return -res

    def SHG(self, Omega, nv, nf, nc, eta=0.05):
        self.eta = eta
        self.nv = nv
        self.nf = nf
        self.nc = nc
        Omega_p = Omega + 1j * eta
        self.G_list = list(set(map(lambda nd_qtm: self.separate(nd_qtm, self.dim)[0], self.nd_basis)))
        self.G_array = np.array(self.G_list)
        self.d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[1], self.d_qtm_list)))
        self.interaction_grad = jacfwd(self.interaction, argnums=(0))
        self.vel_opt = jacfwd(self.hm,argnums=(0))
        self.mk_hamiltonian()
        self.solve()
        self.mk_r_bare()
        self.mk_overlape_idx()
        self.mk_W()
        self.bse_solve()

        f = np.where(np.arange(nc + nv) < nv, 1., 0.)
        f_ij=self.vec2arr(f)
        f_cv = f[None,:] * (1 - f)[:, None]

        r_bare=self.r_bare
        r_bare_gd=self.r_bare_gd
        epsilon=self.epsilon
        epsilon_gd = self.epsilon_gd
        ipa = []
        exc = []
        aug = []
        
        for omega in Omega_p:
            temp_ipa = self.shg(omega, r_bare, r_bare_gd, r_bare, epsilon, epsilon_gd, f_ij)
            ipa += [temp_ipa]

            r_E, r_E_gd = self.r_renormal(omega, r_bare, r_bare_gd, self.R_E, self.eigs_E, self.psi_E, self.W_val_grad, f_cv)
            r2_E = self.r_renormal(-2 * omega, r_bare, r_bare_gd, self.R_E, self.eigs_E, self.psi_E, self.W_val_grad, f_cv, grad=False)
            temp_exc = self.shg(omega, r_E, r_E_gd, r2_E, epsilon, epsilon_gd, f_ij)
            exc += [temp_exc]

            r_A, r_A_gd = self.r_renormal(omega, r_bare, r_bare_gd, self.R_A, self.eigs_A, self.psi_A, self.W_val_grad)
            r2_A = self.r_renormal(-2 * omega, r_bare, r_bare_gd, self.R_A, self.eigs_A, self.psi_A, self.W_val_grad, grad=False)
            temp_aug = self.shg(omega, r_A, r_A_gd, r2_A, epsilon, epsilon_gd, f_ij)
            aug += [temp_aug]
        ipa = np.array(ipa)
        exc = np.array(exc)
        aug = np.array(aug)
        data = np.array([ipa,exc,aug]).flatten()
        np.savetxt("hbn_64.txt",data)

        plt.plot(Omega, np.abs(ipa[:, 0, 0, 0]) / self.num_d_basis * Omega)
        plt.plot(Omega, np.abs(exc[:, 0, 0, 0]) / self.num_d_basis * Omega)
        plt.plot(Omega, np.abs( aug[:, 0, 0, 0]) / self.num_d_basis * Omega)
        plt.savefig("hbn_shg_64.pdf")
        #plt.show()

    def plot_bands(self, pts_list, num_pts=200, d_qtm_nk=None, close=True):
        if len(pts_list) == 0:
            print("No points to plot")
            pass
        if len(self.Basis) == 0:
            print("Please set basis first")
            pass
        else:
            if len(self.nd_basis) == 0:
                print("Please set basis first")
                pass

        d_qtm_nk_list = []
        if d_qtm_nk == None:
            d_qtm_list = list(self.Basis.keys())
            d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[1],d_qtm_list)))
        else:
            d_qtm_nk_list = [d_qtm_nk]

        label_list = []
        vec_list = []
        if isinstance(pts_list, dict):
            self.special_pts.update(pts_list)
            label_list = list(pts_list.keys())
            vec_list = list(pts_list.values())
        elif isinstance(pts_list, (list, tuple)):
            for label in pts_list:
                if self.special_pts.get(label) == None:
                    print("Undefined point "+label)
                    pass

            label_list = list(pts_list)
            vec_list = [self.special_pts[label] for label in label_list]
        if close:
            label_list = label_list + [label_list[0]]
            vec_list = vec_list + [vec_list[0]]

        vec_list = np.array(vec_list)
        num_nodes = len(vec_list)
        len_list = np.array([np.linalg.norm(
            (vec_list[i+1]-vec_list[i])@self.bvec) for i in range(num_nodes-1)])
        num_list = (len_list/len_list.sum()*num_pts).astype(np.int32)
        num_list[-1] = num_list[-1]+(num_pts-num_list.sum())
        node_list = np.insert(num_list.cumsum(), 0, 0)*len_list.sum()/num_pts
        kpath = []
        for i in range(num_nodes-1):
            kpath = kpath+[x/num_list[i]*vec_list[i+1] +
                           (1-x/num_list[i])*vec_list[i] for x in range(num_list[i])]

        x = np.linspace(0, len_list.sum(), num_pts)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        linestyles = ['-', '--', '-.', ':']
        for temp in zip(d_qtm_nk_list, linestyles):
            d_qtm_nk, lt = temp
            t0 = time.clock()
            hms = [self.hm(k, d_qtm_nk) for k in kpath]
            t1 = time.clock()
            engies = jnp.asarray([jnp.linalg.eigvalsh(mat) for mat in hms])
            t2 = time.clock()
            print(t1 - t0)
            print(t2 - t1)
            #engies = jnp.asarray([jnp.linalg.eigvalsh(self.hm(k, d_qtm_nk)) for k in kpath])
            [ax.plot(x, engies[:,i], lt) for i in range(self.num_nd_bands)]
        en_bot = np.min(engies[:,0])
        en_top = np.max(engies[:,-1])
        [ax.plot([node_list[i], node_list[i]], [en_bot, en_top], 'k--')
         for i in range(1, num_nodes-1)]
        ax.set_xticks(node_list)
        ax.set_xticklabels(label_list, minor=False)
        plt.savefig("temp_band.pdf")
        plt.show()


def test_Lattice():
    dim = 2
    avec = sigma[0]
    bvec = sigma[0]
    cubic = Lattice(dim, avec, bvec,
                    real_shape=1, kstart=0, bz_shape=5)
    cubic.mk_basis({}, {})
    cubic.print_basis()
    hoppings = tuple((i, j) for i in range(-1, 2) for j in range(-1, 2))
    cubic.hoppings = hoppings

    def hopping_func(k, d_qtm_nk, nd_qtm1, delta_nd_qtm):
        if delta_nd_qtm == (0., 0.):
            return jnp.linalg.norm((k+jnp.array(nd_qtm1))@bvec)**2+4./(jnp.linalg.norm(jnp.array(delta_nd_qtm))+5)
        else:
            return 4./(jnp.linalg.norm(jnp.array(delta_nd_qtm))+5)
    cubic.hopping_func = hopping_func
    cubic.mk_hamiltonian()
    cubic.print_hamiltonian()
    # cubic.solve()
    # cubic.print_eigen_energies()
    cubic.plot_bands({"G": (0., 0.), "M": (0.5, 0.), "K": (0.5, 0.5)})

def test_renormal():
    test = Lattice(2, sigma[0], sigma[0])
    dim = 2
    nc, nv = 2, 2
    nb = 4
    num_d_basis = 50*50*2
    Dim = num_d_basis * nc * nv
    r_bare = np.ones((dim, num_d_basis, nb, nb), dtype=np.complex64)
    print(sys.getsizeof(r_bare))
    r_bare_gd = np.ones((dim, dim, num_d_basis, nb, nb), dtype=np.complex64)
    print(sys.getsizeof(r_bare_gd))
    R = np.ones((Dim * 2, dim), dtype=np.complex64)
    print(sys.getsizeof(R))
    eigs = np.ones((Dim * 2), dtype=np.complex64)
    print(sys.getsizeof(eigs))
    psi = np.ones((Dim, Dim * 2), dtype=np.complex64)
    print(sys.getsizeof(psi))
    W_val_grad = np.ones((num_d_basis, dim, nb, nb, Dim), dtype=np.complex64)  #sbijm
    print(sys.getsizeof(W_val_grad))
    test.r_renormal(3+0.1*1j,r_bare,r_bare_gd,R,eigs,psi,W_val_grad)


if __name__ == "__main__":
    test_renormal()
