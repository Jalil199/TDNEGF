### Libraries
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.sparse import identity as eye
from quspin.operators import hamiltonian,quantum_operator # operators
from quspin.basis import spin_basis_general # spin basis constructor
import numpy as np # general math functions
import sys

### Parameters, basis and symmetries
Lx, Ly = 5, 2# linear dimension of spin 1 2d lattice
N2d = Lx*Ly  #number of sites for spin 1/2
basis = spin_basis_general(N2d,pauli=1)
no_checks = dict(check_symm=False,check_herm=False,check_pcon=False)
HBAR = 0.658211928e0
### Hamiltonian
def Kitaev_H(alpha = 1, S = np.zeros([3,3]) , Js = [1.0,1.0,1.0], J_coup = 1.0 ):
    '''This function computes the hamiltonian of the Kitaev model coupled 
    to a general localized potentials.The kitaev model by default have a
    size of 5x2, it can be generalized later
    Args:
    ----
    alpha : 1D float, optional 
    This parameter in the kitaev Hamiltonian, it determines the phase 
    S : dictionary 
    Contains the local potential to be coupled to each one of the
    spin operators. 
    Js: 1D array of floats
    Contains the coupling between each one of the spin components, e.g.
    Js=[J_x, J_y, J_z] -> H = J_xS_x S_x + J_yS_yS_y ....
    j_coup: is the strenght between between the system and the locali_
    zed potentials 
    '''
    ##print("join0 " ,S )
    # Couplings between spin 
    Jxx,Jyy,Jzz = Js                                ### Magnitude of the coupling 
    # Configugarion 5x2 npbc
    J_yy = [[Jyy,0,1],[Jyy,2,3],[Jyy,6,7],[Jyy,8,9]]
    J_xx = [[Jxx,1,2],[Jxx,3,4],[Jxx,5,6],[Jxx,7,8]]
    J_zz = [[Jzz,5,0],[Jzz,2,7],[Jzz,4,9]]
    operator_list_0 = [["zz",J_zz],["yy",J_yy],["xx",J_xx]]
    # Contain the Heinsenberg elements of each coupling 
    J_yxx = [[Jxx,0,1],[Jxx,2,3],[Jxx,6,7],[Jxx,8,9]]
    J_yzz = [[Jzz,0,1],[Jzz,2,3],[Jzz,6,7],[Jzz,8,9]]
    J_xyy = [[Jyy,1,2],[Jyy,3,4],[Jyy,5,6],[Jyy,7,8]]
    J_xzz = [[Jzz,1,2],[Jzz,3,4],[Jzz,5,6],[Jzz,7,8]]
    J_zxx = [[Jxx,5,0],[Jxx,2,7],[Jxx,4,9]]
    J_zyy = [[Jyy,5,0],[Jyy,2,7],[Jyy,4,9]]
    operator_list_1 = [["xx",J_zxx],["yy",J_zyy],
                      ["xx",J_yxx],["zz",J_yzz],
                      ["yy",J_xyy],["zz",J_xzz]]
    operator_dict = dict(H0=operator_list_0,H1 = operator_list_1)
    ### Parameter that controls the HK model 
    #print("pass until before params",np.sin(3),alpha)
    params_dict = dict(H0=np.sin(alpha)+np.cos(alpha),H1=np.cos(alpha))
    #(np.zeros([3,3]) !== 0).any()
    #[:,1]
    # print("bool_val :",(S != 0).any())
    if (S != 0.).any() :
        ##print("join " ,S )
        J_x = [ [S[0,0] , 5], [S[0,1]  , 6], [S[0,2],  7], [S[0,3], 8 ], [S[0,4],  9]  ]
        J_y = [ [S[1,0] , 5], [S[1,1]  , 6], [S[1,2],  7], [S[1,3], 8 ], [S[1,4],  9]  ]
        J_z = [ [S[2,0] , 5], [S[2,1]  , 6], [S[2,2],  7], [S[2,3], 8 ], [S[2,4],  9]  ]
        operator_list_3 = [["z", J_z  ], ['x', J_x  ], ['y', J_y ] ]
        operator_dict = dict(H0=operator_list_0,H1=operator_list_1,Hcoup=operator_list_3)
        params_dict = dict(H0=np.sin(alpha)+np.cos(alpha),H1=np.cos(alpha), Hcoup = -J_coup)

    ##print(operator_dict)
    ##print(params_dict)
    # Build the hamiltonian in quspin form 
    H = quantum_operator(operator_dict,basis=basis,dtype=np.complex128,**no_checks)
    # Put the parameters in the hamiltonian 
    H_1 = H.tohamiltonian(params_dict)
    return  H_1     

### Operators 

def spin_op(basis=basis, sites=(0,1) ): 
    site1,site2=sites
    sigma = [['zz', [[1.0, site1,site2]]] ,
             ['yy', [[1.0, site1,site2]]],
             ['xx', [[1.0, site1,site2]]]]
    return hamiltonian(sigma,[],
                       dtype=np.complex128,basis=basis,**no_checks)
def spin_op_z(basis = basis,sites = 1 ):#sites=(0,1) ):
    site1 = sites
    sigma = [['z', [[1.0, site1]]] ]
    return hamiltonian(sigma,[],
                       dtype=np.complex128,basis=basis,**no_checks)
def spin_op_x(basis=basis, sites=1 ):
    site1=sites
    sigma = [['x', [[1.0, site1]]] ]
    return hamiltonian(sigma,[],
                       dtype=np.complex128,basis=basis,**no_checks)
def spin_op_y(basis=basis, sites=1 ):
    site1=sites
    sigma = [['y', [[1.0, site1]]   ]   ] 
    return hamiltonian(sigma,[],
                       dtype=np.complex128,basis=basis,**no_checks)

def spindensity_qsl(psi,sites=[5,6,7]):
    sden= []
    for site in sites:
        S_x = spin_op_x(sites = site).expt_value(V = psi)#.toarray()
        S_y = spin_op_y(sites = site).expt_value(V = psi)#.toarray()
        S_z = spin_op_z(sites = site).expt_value(V = psi)#.toarray()
        sden.append([S_x, S_y, S_z])
    return sden

def Kitaev_correlations(eve):
    correlations=[]
    correlation1=spin_op(sites=(3,2) ).expt_value(eve,check=False)
    correlation2=spin_op(sites=(1,2) ).expt_value(eve,check=False)
    correlation3=spin_op(sites=(7,2) ).expt_value(eve,check=False)
    correlation=correlation1+correlation2+correlation3
    #print('step')
    return correlation



#### Function to evolve the system 

def evolve(H_static, H_dynamic, psi, dt, method='diag', time_dep=0):
    """a function to evolve the wave function in time
    if time independet, we return Ut and should be called outside
    the loop
    """
    size = H_static.shape[0]
    if method == 'gpu' or method == 'CN_gpu':
        import pycuda.gpuarray as gpuarray
        import pycuda.autoinit
#         from skcuda import linalg
#         from skcuda import misc
#         linalg.init()
    if not time_dep:
        if method=='expm': 
            Ut = la.expm(-1j*H_static.toarray()*dt/HBAR)
            return Ut
        elif method=='eig_vec':
            w,v = la.eigh(H_static.toarray())
            Ut = np.asarray([np.exp(-1j*w[i]*dt) for i in range(size)])
            return Ut
        elif method=='CN':
#             Ut = sla.inv((eye(size) + 1j*dt*H_static/(2*HBAR)).tocsc()) @ \
#                        (eye(size) - 1j*dt*H_static/(2*HBAR))
            Ut = la.inv(( np.eye(size) + 1j*dt*H_static.toarray()/(2*HBAR))) @ \
                       ( np.eye(size) - 1j*dt*H_static.toarray()/(2*HBAR))
            return Ut@psi
            
        elif method=='CN_spilu':
            B = sla.spilu(eye(size) + 1j*dt*H_static/(2*HBAR))
            return B
    
        elif method=='CN_gpu':
            mat      = (eye(size) + 1j*dt*H_static/(2*HBAR))
            a_gpu    = gpuarray.to_gpu(mat.toarray()) 
            ainv_gpu = linalg.inv(a_gpu, overwrite=True)
            mat_inv  = ainv_gpu.get()
            Ut       = mat_inv @ (eye(size) - 1j*dt*H_static/(2*HBAR))
            return Ut
    else: 
        if method=='expm': 
            Ht = H_static + H_dynamic
            Ut = la.expm(-1j*Ht.toarray()*dt/HBAR)
            psi_new = Ut @ psi
            return psi

        elif method=='CN':
            Ht = H_static + H_dynamic
            Ut = sla.inv(eye(size) + 1j*dt*Ht/(2*HBAR)) @ \
                        (eye(size) - 1j*dt*Ht/(2*HBAR))
            psi = Ut @ psi
            return psi
            
        if method=='eig_vec':
            Ht = H_static + H_dynamic
            w, v = la.eigh(Ht.toarray())
            Ut = np.asarray([np.exp(-1j*w[i]*dt) for i in range(2)])
            cn = [v[:,i].conj() @ psi for i in range(2)]
            psi = np.sum([cn[i] * (Ut[i] * v[:,i]) for i in range(2)], axis=0)
            return psi

        elif method=='CN_gpu':
            Ht = H_static + H_dynamic
            mat      = (eye(size) + 1j*dt*Ht/(2*HBAR))
            a_gpu    = gpuarray.to_gpu(mat.toarray()) 
            ainv_gpu = linalg.inv(a_gpu, overwrite=True)
            mat_inv  = ainv_gpu.get()
            Ut       = mat_inv @ (eye(size) - 1j*dt*Ht/(2*HBAR))
            psi = Ut @ psi
            return psi
