import numpy as np
import qutip as qt
from pulsee import simulation as sim
import matplotlib.pyplot as plot
from scipy.linalg import norm, svd
from scipy.sparse.linalg import svds

from tqdm import trange

# Global variable n, number of qubits
n = 2
# Helper state
helper_rho = None
proj_ops = None


def ts(q1, q2):
    # Helper function to perform the tensor product between
    # two Qobj
    return qt.tensor(q1, q2)


def gen_initial_state(n_qubits, temperature=1e-3):
    # Generates initial state based on how many qubits are given

    global n
    n = n_qubits

    if n == 1:
        raise Exception("Sorry, 1 < n < 5")

    if n == 2:
        # here the qubits are assumed to be nuclear spins
        # spin-1/2 particles are qubits
        QUANTUM_NUMBERS = [1 / 2, 1 / 2]
        # give a gyromagnetic ratio in MHz/T for each spin.
        GAMMA_2PIS = [4.1 / (2 * np.pi), 3.9 / (2 * np.pi), 3.7 / (2 * np.pi)]

    elif n == 3:
        QUANTUM_NUMBERS = [1 / 2, 1 / 2, 1 / 2]
        GAMMA_2PIS = [4.1 / (2 * np.pi), 3.9 / (2 * np.pi), 3.7 / (2 * np.pi)]

    elif n == 4:
        QUANTUM_NUMBERS = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
        GAMMA_2PIS = [4.1 / (2 * np.pi), 3.9 / (2 * np.pi), 3.7 / (2 * np.pi), 3.1 / (2 * np.pi)]

    else:
        raise Exception("Sorry, n > 4 not implemented yet.")

    # B0 is the field that the spins are placed in
    B0 = 2 * np.pi

    # Zeeman parameters of the magnetic field. Assume field along the z direction
    zeem_par = {'field magnitude': B0, 'theta_z': 0, 'phi_z': 0}

    # Per the pulsee formalizsm
    args = {}
    spin_par = []
    for qn, gam in zip(QUANTUM_NUMBERS, GAMMA_2PIS):
        spin_par.append({'quantum number': qn, 'gamma/2pi': gam})
    args['spin_par'] = spin_par
    args['zeem_par'] = zeem_par

    # Initial state (rho_0) is canonical
    spin, _, rho_0 = sim.nuclear_system_setup(temperature=temperature, **args)

    global helper_rho
    helper_rho = rho_0

    # Higher order "Pauli" operators
    Ix, Iy, Iz = (spin.I['x'], spin.I['y'], spin.I['z'])
    Id = qt.qeye(spin.dims[0])

    S = 1 / 2
    (Sx, Sy, Sz) = qt.spin_J_set(S)
    Id_S = qt.qeye(int(2 * S + 1))

    # 1 qubit Pauli operators
    ops_qubit = [Id_S, 2 * Sx, 2 * Sy, 2 * Sz]
    global proj_ops
    proj_ops = get_proj_ops(ops_qubit)

    return rho_0, [Ix, Iy, Iz], ops_qubit, proj_ops, Id_S, Id


def get_proj_ops(ops):
    projection_ops = []
    if n == 2:
        # P1 = ts(Id_S, Sx)
        # P2 = ts(Id_S, Sy)
        # P3 = ts(Id_S, Sz)
        # P4 = ts(Sx, Id_S)
        # P5 = ts(Sy, Id_S)
        # P6 = ts(Sz, Id_S)
        # P7 = ts(Sx, Sx)
        # P8 = ts(Sx, Sy)
        # P9 = ts(Sx, Sz)
        # P10 = ts(Sy, Sx)
        # P11 = ts(Sy, Sy)
        # P12 = ts(Sy, Sz)
        # P13 = ts(Sz, Sx)
        # P14 = ts(Sz, Sy)
        # P15 = ts(Sz, Sz)
        # P16 = ts(Id_S, Id_S)
        for i in ops:
            for j in ops:
                projection_ops.append(ts(i, j))
    elif n == 3:
        for i in ops:
            for j in ops:
                for k in ops:
                    projection_ops.append(ts(ts(i, j), k))
    elif n == 4:
        for i in ops:
            for j in ops:
                for k in ops:
                    for l in ops:
                        projection_ops.append(ts(ts(ts(i, j), k), l))
    return projection_ops


def perform_meas(X):
    # Function that effectuates the measurements that need to be 
    # physically performed 
    measurements = []
    for i in proj_ops:
        # Round any numbers to 10 decimal places
        measurements.append(np.round((i * X).tr(), 10))
    return measurements


def recover_state(measurements):
    # Recover the density matrix based on the performed 
    # measurements
    state = 0 * helper_rho
    for i in range(len(measurements)):
        state += ((1 / 2) ** n) * measurements[i] * proj_ops[i]
    return state


def fidelity(rho, sigma):
    # calculates the fidelity
    # adapted from https://doi.org/10.1063/1.1465412
    return np.round((rho * sigma).tr() / np.sqrt((rho ** 2).tr() * (sigma ** 2).tr()), 10)


def svst(x, L):
    U, S, V = svd(x, full_matrices=False)
    S[S <= L] = 0
    S[S > L] = S[S > L] - L
    return U.dot(np.diag(S)).dot(V)


def fista_with_nesterov(x_0, eps=1e-4, L=0.9, max_num_iterations=1000):
    # dims of the qubit space
    dims = x_0.dims
    x_0 = np.array(x_0)

    # store the current x for the iteration at step one
    x_prev = x_0
    x = x_0

    mask = x_0.copy()
    mask = np.ma.where(mask != 0)

    # at step 0 y_1 = x_0 and t1 =1
    y = x_0.copy()
    t = 1

    # cannot divide by 0, so making this as small as posible
    Z_old = float('inf')

    for _ in trange(max_num_iterations):

        # mask y_k same as x_0
        y[mask] = x_0[mask]
        # set the current x_{k} to p_L(y_k)
        x = svst(y, L)

        # apply nesterov acceleration on the gradient descent
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2

        # perform measurement on the current prediction
        meas = perform_meas(qt.Qobj(x, dims=dims))
        state = recover_state(meas)
        x = np.array(state.dag())

        y_next = x + ((t - 1) / t_next) * (x - x_prev)

        # assign values for next iteration
        x_prev = x
        t = t_next
        y = y_next

        # convergence check
        Z = np.sum((x[mask] - x_0[mask]) ** 2) + L * norm(x, ord='nuc')

        if abs(Z / Z_old-1) < eps:
            break
        Z_old = Z
    x = qt.Qobj(x, dims=dims)
    # make sure the final matrix is Hermitian
    Xenforced = (x + x.dag()) / 2
    return x, Xenforced.unit()


def admm(x_0, eps=1e-3, rho=0.05, L=0.9, max_num_iterations=1000):
    # dims of the qubit space
    dims = x_0.dims

    # store the current x for the iteration at step one
    x_0 = np.array(x_0)
    x = x_0

    mask = x_0.copy()
    mask = np.ma.where(mask != 0)

    # initialize z and u
    z = np.zeros_like(x_0)
    u = np.zeros_like(x_0)

    # cannot divide by 0, so making this as small as posible
    Z_old = float('inf')

    for _ in trange(max_num_iterations):

        # update values for next iteration
        x = (x_0 + rho * (z - u)) / rho

        # perform measurement on the current prediction
        meas = perform_meas(qt.Qobj(x, dims=dims))
        x = np.array(recover_state(meas))
        x[mask] = x_0[mask]

        z = svst(x + u, L / rho)

        # perform measurement on the current prediction
        meas = perform_meas(qt.Qobj(z, dims=dims))
        z = np.array(recover_state(meas))
        u = u + x - z

        # convergence check
        Z = np.sum((x[mask] - x_0[mask]) ** 2) + L / rho * norm(x, ord='nuc')
        if abs(Z / Z_old-1) < eps:
            break
        Z_old = Z
    x = qt.Qobj(x, dims=dims)
    # make sure the final matrix is Hermitian
    Xenforced = (x + x.dag()) / 2
    return x, Xenforced.unit()


def svd_k(M, tau, rk, l, rank):
    while True:
        # uses the Lanczos algorithm
        Uk, Sk, Vk = svds(M, k=rk)

        # Per the SVTA Algorithm, only keep the eigenvalues
        # that are greater then tau and get rid of the rest
        if np.min(Sk) <= tau or rk >= rank:
            if rk > rank:
                rk = rank
            for i in range(len(Sk)):
                if Sk[i] <= tau:
                    Sk[i] = 0
                    rk -= 1
                if Sk[i] > tau:
                    break
            # Because of the ways the svds algorithm is implemented,
            # flip all the matrices so that it is ordered highest to lowest
            # and make sure the flips of the 2D arrays are correct
            return np.fliplr(Uk), np.flip(Sk), np.flipud(Vk), rk
        # rk counts how many eigenstates are kept
        rk += l


def svt(M, eps=1e-4, delta=9e-1, k0=9.23e-1, l=1, rank=200, steps=1000):
    '''Implementation of the 
    Singular value thresholding (SVT) algorithm (SVTA)
    by Cai, Candes and Shen.
    
    See paper details at:
    https://doi.org/10.48550/arXiv.0810.3286
    '''
    # dims of the qubit space
    dims = M.dims
    rank = M.shape[0] - 2
    M = np.array(M)

    # Create a mask for faster computation
    M_ones = M.copy()
    M_ones[M_ones != 0] = 1

    # Initilize some parameters
    r = 0

    # Y is going to be SVD decomposed using the Lanczos algorithm
    Y = delta * k0 * M

    # The norm 2 of M is used for the hyperparameters
    M_norm2 = np.linalg.norm(M, ord=2)
    # we need a large tau to make sure that X minimizes M
    tau = delta * M_norm2 * k0
    loss = float('inf')

    for _ in trange(steps):
        # sk defines what the rank of the SVD decomposed matrices
        sk = r + 1

        if sk >= rank:
            # make sure that the SVD rank doesn't exceed some rank
            sk = rank

        # use the SVD algorithm from scipy.sparse.linalg.svds
        # uses the Lanczos algorithm to use the ``m'' most useful'' eigenvalues.
        # This algorithm should be much faster when using the huge matrix
        # Make sure to use the values from the matrix M that we already know
        mask = np.where(M_ones != 0)
        Y[mask] = M[mask]
        Uk, Sk, Vk, r = svd_k(Y, tau, sk, l, rank)

        # Here X is the matrix that supposedly minimizes M
        # np.maximum is used to discard any eigenvalue that is smaller than tau
        X = Uk @ np.diag(np.maximum(Sk - tau, 0)) @ Vk

        # perform measurement on the current X
        meas = perform_meas(qt.Qobj(X, dims=dims))
        X = recover_state(meas)

        M = qt.Qobj(M, dims=dims)

        # The loss is defined in the paper
        # M_ones is going to mask any values that are not in \Omega
        prev_loss = loss
        loss = ((M_ones * (X - M)) ** 2).sum() + delta * np.linalg.norm(X, ord='nuc')

        # define some basic condition to break the loop
        if abs(loss / prev_loss - 1) < eps:
            break

        # update Y
        Y += delta * M_ones * (M - X)

    # make sure the final matrix is Hermitian
    Xenforced = (X + X.dag()) / 2
    return X, Xenforced.unit()
