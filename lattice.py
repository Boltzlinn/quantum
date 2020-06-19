from my_bsr import my_bsr_mat
from quantum import Quantum, sigma
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
import numpy as np
import matplotlib.pyplot as plt


class Lattice(Quantum):

    def __init__(self, dim, avec, bvec, real_shape=3, kstart=0., bz_shape=5, special_pts={}, ifspin=0, ifmagnetic=0, hoppings=(), num_sub=1):
        self.dim = dim
        self.avec = avec
        self.bvec = bvec
        self.special_pts = special_pts
        self.diag_qtm_nums = {}
        self.ndiag_qtm_nums = {}
        if np.sum(real_shape) <= 0:
            print('real_shape can not be 0 or negative, real_shape is set to 1 instead')
            real_shape = 1
        self.real_shape = np.zeros(dim, dtype=np.int32)+real_shape
        self.kstart = np.zeros(dim)+kstart
        if np.sum(bz_shape) <= 0:
            print('bz_shape can not be 0 or negative, bz_shape is set to 5 instead')
            bz_shape = 5
        self.bz_shape = np.zeros(dim, dtype=np.int32)+bz_shape
        self.hoppings = hoppings
        self.num_sub = num_sub
        klabels = ('kx', 'ky', 'kz')
        Glabels = ('Gz', 'Gy', 'Gz')
        for i in range(dim):
            self.diag_qtm_nums[klabels[i]] = [self.kstart[i] + n / self.real_shape[i] for n in range(self.real_shape[i])]
            self.ndiag_qtm_nums[Glabels[i]] = [n - self.bz_shape[i] // 2 for n in range(self.bz_shape[i])]
        if ifspin:
            if ifmagnetic:
                self.ndiag_qtm_nums['spin'] = [-0.5, 0.5]
            else:
                self.diag_qtm_nums['spin'] = [-0.5, 0.5]

    @staticmethod
    def hopping_func(k, d_qtm_nk, nd_qtm1, delta_nd_qtm):
        pass

    def hm(self, k, d_qtm_nk):
        basis = list(self.Basis.values())[0][1]
        indices = []
        data = []
        indptr = [0]
        count = 0
        for nd_qtm1 in basis:
            #self.hopping could be tuple, list, etc. but not a generator
            for delta_nd_qtm in self.hoppings:
                nd_qtm2 = tuple(np.array(nd_qtm1)-np.array(delta_nd_qtm))
                j = basis.get(nd_qtm2)
                if j != None:
                    count = count+1
                    indices = indices+[j]
                    data = data + [self.hopping_func(k, d_qtm_nk, nd_qtm1, delta_nd_qtm)]
            indptr = indptr + [count]
        
        indptr = np.array(indptr)
        indices = np.array(indices)
        data = jnp.array(data)
        num_basis = len(basis)
        num_bands = self.num_sub*num_basis
        self.num_bands = num_bands
        return my_bsr_mat(data, indices, indptr, shape=(num_bands, num_bands))

    def mk_hamiltonian(self):
        hamiltonian = {}
        for d_qtm in self.Basis:
            k = jnp.array(d_qtm[0:self.dim])
            if len(d_qtm) == self.dim:
                d_qtm_nk = ()
            else:
                d_qtm_nk = d_qtm[self.dim:]
            hamiltonian[d_qtm] = self.hm(k, d_qtm_nk)
        self.hamiltonian = hamiltonian

    def plot_bands(self, pts_list, num_pts=200, d_qtm_nk=None):
        if len(pts_list) == 0:
            print("No points to plot")
            pass
        if len(self.Basis) == 0:
            print("Please set basis first")
            pass
        else:
            if len(list(self.Basis.values())[0][1]) == 0:
                print("Please set basis first")
                pass

        d_qtm_nk_list = []
        if d_qtm_nk == None:
            d_qtm_list = list(self.Basis.keys())
            if len(d_qtm_list[0]) == self.dim:
                d_qtm_nk_list = [()]
            else:
                d_qtm_nk_list = list(
                    set([d_qtm[self.dim:] for d_qtm in d_qtm_list]))
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
        label_list = label_list + [label_list[0]]
        vec_list = jnp.array(vec_list + [vec_list[0]])

        num_nodes = len(vec_list)
        len_list = np.array([jnp.linalg.norm(
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

        for d_qtm_nk in d_qtm_nk_list:
            engies = jnp.array([jnp.linalg.eigvalsh(
                self.hm(k, d_qtm_nk)) for k in kpath])
            [ax.plot(x, engies[:, i]) for i in range(self.num_bands)]
        en_bot = np.min(engies[:, 0])
        en_top = np.max(engies[:, -1])
        [ax.plot([node_list[i], node_list[i]], [en_bot, en_top], 'k--')
         for i in range(1, num_nodes-1)]
        ax.set_xticks(node_list)
        ax.set_xticklabels(label_list, minor=False)
        plt.savefig("temp_band.pdf")
        plt.show()


def test_Lattice():
    dim=2
    avec=sigma[0]
    bvec=sigma[0]
    cubic = Lattice(dim, avec, bvec,
                    real_shape=1, kstart=0, bz_shape=5)
    cubic.mk_basis({}, {})
    cubic.print_basis()
    hoppings = tuple((i,j) for i in range(-1,2) for j in range(-1,2))
    cubic.hoppings = hoppings
    def hopping_func(k, d_qtm_nk, nd_qtm1, delta_nd_qtm):
        if delta_nd_qtm == (0., 0.):
            return jnp.linalg.norm((k+jnp.array(nd_qtm1))@bvec)**2+4./(jnp.linalg.norm(jnp.array(delta_nd_qtm))+5)
        else:
            return 4./(jnp.linalg.norm(jnp.array(delta_nd_qtm))+5)
    cubic.hopping_func = hopping_func
    cubic.mk_hamiltonian()
    cubic.print_hamiltonian()
    #cubic.solve()
    #cubic.print_eigen_energies()
    cubic.plot_bands({"G":(0.,0.), "M":(0.5,0.), "K":(0.5,0.5)})


if __name__ == "__main__":
    test_Lattice()
