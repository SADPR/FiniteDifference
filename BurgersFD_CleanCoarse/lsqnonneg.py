import numpy as np
import time

def lsqnonneg(C, d, x0=None, tol=None, itmax_factor=100, max_support=None, rel_err_thresh=0.01):
    '''Linear least squares with nonnegativity constraints.

    (x, resnorm, residual) = lsqnonneg(C,d) returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real
    '''

    print("Running lsqnonneg NNLS algorithm")
    eps = 2.22e-16  # from Matlab precision tolerance
    def norm1(x):
        return np.abs(x).sum().max()

    def msize(x, dim):
        s = x.shape
        return 1 if dim >= len(s) else s[dim]

    if tol is None: 
        tol = 10 * eps * norm1(C) * (max(C.shape) + 1)

    C = np.asarray(C)
    (m, n) = C.shape
    P = np.zeros(n)
    Z = np.arange(1, n + 1)

    x = P if x0 is None or any(x0 < 0) else x0
    ZZ = Z

    resid = d - np.dot(C, x)
    w = np.dot(C.T, resid)

    outeriter = 0
    it = 0
    itmax = itmax_factor * n
    exitflag = 1

    # Start timing
    total_start_time = time.time()

    # Outer loop for positive coefficients set
    while np.any(Z) and np.any(w[ZZ - 1] > tol):
        outer_start_time = time.time()
        outeriter += 1

        t = w[ZZ - 1].argmax()
        t = ZZ[t]

        P[t - 1] = t
        Z[t - 1] = 0

        PP = np.where(P != 0)[0] + 1
        ZZ = np.where(Z != 0)[0] + 1

        CP = np.zeros(C.shape)
        CP[:, PP - 1] = C[:, PP - 1]
        CP[:, ZZ - 1] = np.zeros((m, msize(ZZ, 1)))

        # Solve for z in CP * z ≈ d
        z_start_time = time.time()
        z = np.dot(np.linalg.pinv(CP), d)
        print(f"  Time to compute z: {time.time() - z_start_time:.4f} seconds")

        z[ZZ - 1] = np.zeros(msize(ZZ, 1))

        # Inner loop to remove elements from positive set if they no longer belong
        while np.any(z[PP - 1] <= tol):
            it += 1
            if it > itmax:
                max_error = z[PP - 1].max()
                raise Exception(f"Iteration limit ({it}) exceeded. Increase tolerance tol (max_error={max_error}).")

            QQ = np.where((z <= tol) & (P != 0))[0]
            alpha = min(x[QQ] / (x[QQ] - z[QQ]))
            x += alpha * (z - x)

            ij = np.where((np.abs(x) < tol) & (P != 0))[0] + 1
            Z[ij - 1] = ij
            P[ij - 1] = 0
            PP = np.where(P != 0)[0] + 1
            ZZ = np.where(Z != 0)[0] + 1

            CP[:, PP - 1] = C[:, PP - 1]
            CP[:, ZZ - 1] = np.zeros((m, msize(ZZ, 1)))

            z = np.dot(np.linalg.pinv(CP), d)
            z[ZZ - 1] = np.zeros(msize(ZZ, 1))

        x = z
        resid = d - np.dot(C, x)
        w = np.dot(C.T, resid)

        # Monitor iteration size and error
        num_pos = (x > 0).sum()
        rel_err = np.linalg.norm(resid) / np.linalg.norm(d)
        print(f"Outer iteration {outeriter}, Active set size: {num_pos}, Relative error: {rel_err:.4f}, "
              f"Iteration time: {time.time() - outer_start_time:.4f} seconds")

        if rel_err < rel_err_thresh:
            print(f"Error threshold with relative error {rel_err} achieved. Exiting.")
            break
        if max_support is not None and num_pos >= max_support:
            print(f"Maximum solution vector support reached with {num_pos} nonzero elements. Exiting.")
            break

    # Print total time and results
    total_time = time.time() - total_start_time
    print(f"Total time for lsqnonneg: {total_time:.4f} seconds")
    return x, sum(resid * resid), resid


# Unittest
if __name__=='__main__':
    C = np.array([[0.0372, 0.2869],
                     [0.6861, 0.7071],
                     [0.6233, 0.6245],
                     [0.6344, 0.6170]])

    C1 = np.array([[0.0372, 0.2869, 0.4],
                      [0.6861, 0.7071, 0.3],
                      [0.6233, 0.6245, 0.1],
                      [0.6344, 0.6170, 0.5]])

    C2 = np.array([[0.0372, 0.2869, 0.4],
                      [0.6861, 0.7071,-0.3],
                      [0.6233, 0.6245,-0.1],
                      [0.6344, 0.6170, 0.5]])

    d = np.array([0.8587, 0.1781, 0.0747, 0.8405])

    [x, resnorm, residual] = lsqnonneg(C, d)
    dres = abs(resnorm - 0.8315)          # compare with matlab result
    print('ok, diff:', dres)
    if dres > 0.001:
        raise Exeption('Error')

    [x, resnorm, residual] = lsqnonneg(C1, d)
    dres = abs(resnorm - 0.1477)          # compare with matlab result
    print('ok, diff:', dres)
    if dres > 0.01:
        raise Exeption('Error')

    [x, resnorm, residual] = lsqnonneg(C2, d)
    dres = abs(resnorm - 0.1027)          # compare with matlab result
    print('ok, diff:', dres)
    if dres > 0.01:
        raise Exeption('Error')

    k = np.array([[0.1210, 0.2319, 0.4398, 0.9342, 0.1370],
                     [0.4508, 0.2393, 0.3400, 0.2644, 0.8188],
                     [0.7159, 0.0498, 0.3142, 0.1603, 0.4302],
                     [0.8928, 0.0784, 0.3651, 0.8729, 0.8903],
                     [0.2731, 0.6408, 0.3932, 0.2379, 0.7349],
                     [0.2548, 0.1909, 0.5915, 0.6458, 0.6873],
                     [0.8656, 0.8439, 0.1197, 0.9669, 0.3461],
                     [0.2324, 0.1739, 0.0381, 0.6649, 0.1660],
                     [0.8049, 0.1708, 0.4586, 0.8704, 0.1556],
                     [0.9084, 0.9943, 0.8699, 0.0099, 0.1911]])

    k1 = k-0.5

    l = np.array([0.4225, 0.8560, 0.4902, 0.8159, 0.4608, 0.4574, 0.4507, 0.4122, 0.9016, 0.0056])

    [x, resnorm, residual] = lsqnonneg(k, l)
    dres = abs(resnorm - 0.3695)          # compare with matlab result
    print('ok, diff:', dres)
    if dres > 0.01:
        raise Exeption('Error')

    [x, resnorm, residual] = lsqnonneg(k1, l)
    dres = abs(resnorm - 2.8639)          # compare with matlab result
    print('ok, diff:', dres)
    if dres > 0.01:
        raise Exeption('Error')

    C = np.array([[1.0, 1.0],
                     [2.0, 1.0],
                     [5.0, 1.0],
                     [6.0, 1.0],
                     [10.0, 1.0]])

    d = np.array([3, 5, 11, 13, 21])

    [x, resnorm, residual] = lsqnonneg(C, d)

    print([x, resnorm, residual])
