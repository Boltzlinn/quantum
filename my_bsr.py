import jax.numpy as jnp
from scipy import sparse
from jax import custom_jvp, jacfwd
from functools import partial
import numpy as np


@partial(custom_jvp, nondiff_argnums=(1, 2, 3))
def my_bsr_mat(data, indices, indptr, shape):
    if len(jnp.shape(data)) == 1:
        return jnp.array(sparse.csr_matrix((data, indices, indptr), shape).toarray())
    else:
        return jnp.array(sparse.bsr_matrix((data, indices, indptr), shape).toarray())


@my_bsr_mat.defjvp
def my_bsr_mat_jvp(indices, indptr, shape, priminals, tangents):
    data = priminals[0]
    data_dot = tangents[0]
    priminals_out = my_bsr_mat(data, indices, indptr, shape)
    data_shape = jnp.shape(data)
    data_size = jnp.size(data)
    data_dot = data_dot.flatten()
    def partial_data(d_data): return my_bsr_mat(
        d_data.reshape(data_shape), indices, indptr, shape)
    tangents_out = data_dot@jnp.transpose(jnp.array(
        [partial_data(d_data) for d_data in jnp.eye(data_size)]), axes=(1, 0, 2))
    return priminals_out, tangents_out


if __name__ == "__main__":
    indptr = jnp.array([0, 2, 3, 6])
    indices = jnp.array([0, 2, 2, 0, 1, 2])
    shape = (6, 6)
    ones = jnp.ones((2, 2))
    def H(k):
        kx,ky=k
        data = jnp.array([kx**2*ones, 2.*kx*ones, 3.*ky*ones, 4.*kx*ky*ones, 5.*ky**2*ones, 6.*ones])
        return my_bsr_mat(data,indices,indptr,shape)

    #H_str=np.array(['kx^2','0.','2.kx','0.','0.','3.ky','4.kx ky', '5.ky^2', '6.']).reshape(shape)
    k=(1.,1.)
    #print("The sparse 'Hamiltonian' is H(kx,ky)=")
    #print(H_str)
    print("The values at k="+str(k)+"is:")
    print(H(k))
    print("The derivatives respective to kx and ky are:")
    print("H_{kx}"+str(k))
    print(jacfwd(H)(k)[0])
    print("H_{ky}"+str(k))
    print(jacfwd(H)(k)[1])


