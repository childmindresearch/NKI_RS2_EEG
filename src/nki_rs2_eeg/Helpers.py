import numpy as np
from scipy.linalg import eigh          # generalized eigenvalue solver
from scipy.stats import f as f_dist    # F-distribution, for significance testing later



def rW_scalar_fast(Y):
    Yc = Y - Y.mean(axis=0, keepdims=True)    # center each subject (column)
    return np.sum(Yc**2)

def rB_scalar_fast(Y):
    Yc = Y - Y.mean(axis=0, keepdims=True)
    rT = np.sum(Yc.sum(axis=1)**2)            # sum_i ( sum_l centered )^2  = r_T
    rW = np.sum(Yc**2)
    return rT - rW                            # r_B = r_T - r_W   (Eq. 12)

def isc_scalar_fast(Y):
    N = Y.shape[1]
    return (1.0 / (N - 1)) * rB_scalar_fast(Y) / rW_scalar_fast(Y)
#%%
def RW_fast(X):
    '''Eq. (7) via one matrix multiply.Within-subject covariance, Eq. (7). Memory-safe: loops over subjects."""
'''
    T, D, N = X.shape
    #Xc = X - X.mean(axis=0, keepdims=True)          # center each subject over time
    #M  = Xc.transpose(1, 0, 2).reshape(D, T * N)    # stack all (i,l) columns -> (D, T*N)
    RW = np.zeros((D, D))                                  # float64 accumulator
    for l in range(N):
        print(f"calculating RW for {l}/{N} ")
        Xl = np.asarray(X[:, :, l], dtype=np.float64)      # one subject, (T, D)
        Xl -= Xl.mean(axis=0)                              # center over time
        RW += Xl.T @ Xl
    return RW                                # sum of outer products

def RT_fast(X):
    '''Total covariance via Eq. (14): sum over subject-averaged signal.'''
    T, D, N = X.shape
    xbar = np.zeros((T, D))                                # subject-average signal
    for l in range(N):
        xbar += np.asarray(X[:, :, l], dtype=np.float64)
    xbar /= N
    xbar -= xbar.mean(axis=0)                              # subtract grand mean
    return N**2 * (xbar.T @ xbar)

def RB_fast(X):
    '''Eq. (6) without any pairwise loop:  R_B = R_T - R_W   (Eq. 12).'''
    return RT_fast(X) - RW_fast(X)
#%%
def corrca(X, RW=None, RB=None):
    '''Solve CorrCA. Returns V (D x D, columns = components, sorted by ISC),
    and the ISC value rho_d of each component computed from Eq. (5).'''
    T, D, N = X.shape
    if RW is None: RW = RW_fast(X)
    if RB is None: RB = RB_fast(X)

    # generalized eigenproblem  R_B v = lambda R_W v   (Eq. 8)
    eigvals, V = eigh(RB, RW)                  # ascending order
    order = np.argsort(eigvals)[::-1]          # sort descending (largest ISC first)
    V = V[:, order]

    # ISC of each component, directly from Eq. (5):  rho = (1/(N-1)) v^T R_B v / v^T R_W v
    rho = np.array([(1.0/(N-1)) * (V[:, d] @ RB @ V[:, d]) / (V[:, d] @ RW @ V[:, d])
                    for d in range(D)])
    return V, rho

def project_component(X, v):
    '''Return Y of shape (T, N) with Y[i, l] = v . x^l_i  (Eq. 1).'''
    return np.einsum('idl,d->il', X, v)

def project_all(X, V):
    '''Return Y of shape (T, N, D): Y[i, l, d] = v_d . x^l_i for every component d.'''
    return np.einsum('idl,dc->ilc', X, V)

def forward_model(X, V, RW=None):
    '''Forward model A from Eq. (28): A = R_W V (V^T R_W V)^{-1}.'''
    if RW is None: RW = RW_fast(X)
    return RW @ V @ np.linalg.inv(V.T @ RW @ V)

def shrinkage(RW, gamma):
    '''Shrinkage-regularized within-subject covariance, Eq. (63).'''
    D = RW.shape[0]
    lam_bar = np.trace(RW) / D                       # mean eigenvalue
    return (1 - gamma) * RW + gamma * lam_bar * np.eye(D)

def tsvd_inverse(RW, K):
    '''Regularized inverse of R_W keeping K principal eigenvectors, Eqs. (60)-(61).'''
    vals, vecs = np.linalg.eigh(RW)                  # ascending
    idx = np.argsort(vals)[::-1][:K]                 # K largest
    U, L = vecs[:, idx], vals[idx]
    return U @ np.diag(1.0 / L) @ U.T                # tilde R_W^{-1}

def corrca_regularized(X, gamma=0.0, K=None):
    '''CorrCA with optional shrinkage (gamma) or TSVD (K). Returns V, rho.'''
    T, D, N = X.shape
    RW, RB = RW_fast(X), RB_fast(X)

    if K is not None:                                # TSVD path
        RW_inv = tsvd_inverse(RW, K)
        eigvals, V = np.linalg.eig(RW_inv @ RB)      # solve (R_W^{-1} R_B) v = lambda v
        V = np.real(V); eigvals = np.real(eigvals)
    else:                                            # shrinkage path
        RW_reg = shrinkage(RW, gamma)
        eigvals, V = eigh(RB, RW_reg)                # regularized R_W only to FIND directions

    order = np.argsort(eigvals)[::-1]
    V = V[:, order]
    # ISC is always measured against the TRUE (unregularized) R_W, Eq. (5),
    # so it respects the bound rho <= 1 (Appendix G.2). Regularization only
    # affects which directions V we find, not how we score them.
    rho = np.array([(1.0/(N-1)) * (V[:, d] @ RB @ V[:, d]) / (V[:, d] @ RW @ V[:, d])
                    for d in range(V.shape[1])])
    return V, rho

def f_statistic(rho, T, N):
    '''F-statistic from Eq. (26).'''
    return (T * (N - 1) * rho + T) / ((T - 1) * (1 - rho))

def isc_pvalues(rho, T, N):
    '''Parametric p-values for each component's ISC (Eq. 26 + F-distribution).'''
    F = f_statistic(rho, T, N)
    d1, d2 = T * (N - 1), T - 1
    return f_dist.sf(F, d1, d2)        # upper-tail probability


def per_subject_isc_direct(Y):
    '''Per-subject ISC for one component, Eqs. (84)-(85), summed exactly as written.
    Y has shape (T, N): the projected component, Y[i, l] = y^l_i.
    Returns rho of shape (N,).'''
    T, N = Y.shape
    ybar = Y.mean(axis=0)                       # bar y^l for each subject

    def r(k, l):                                # r_{kl}, Eq. (85)
        s = 0.0
        for i in range(T):
            s += (Y[i, k] - ybar[k]) * (Y[i, l] - ybar[l])
        return s

    rho = np.zeros(N)
    for k in range(N):
        num = 0.0
        den = 0.0
        for l in range(N):
            if l == k:
                continue
            num += r(k, l) + r(l, k)            # numerator of Eq. (84)
            den += r(l, l) + r(k, k)            # denominator of Eq. (84)
        rho[k] = num / den
    return rho

def per_subject_isc_fast(Y):
    '''Same as per_subject_isc_direct, vectorized. Y shape (T, N) -> rho shape (N,).'''
    T, N = Y.shape
    Yc = Y - Y.mean(axis=0, keepdims=True)      # center each subject over time
    R  = Yc.T @ Yc                              # R[k, l] = r_{kl}  (all pairs at once)
    diag = np.diag(R)                           # r_{kk}
    rho = np.zeros(N)
    for k in range(N):
        off = np.arange(N) != k                 # the l != k entries
        num = (R[k, off] + R[off, k]).sum()     # sum_{l!=k} (r_kl + r_lk)
        den = (diag[off] + diag[k]).sum()       # sum_{l!=k} (r_ll + r_kk)
        rho[k] = num / den
    return rho

def _within_cov(X, tmask):
    T, D, N = X.shape
    RW = np.zeros((D, D))
    for l in range(N):
        Xl = np.asarray(X[:, :, l][tmask], dtype=np.float64)   # contiguous (T_sel, D)
        Xl -= Xl.mean(axis=0)
        RW += Xl.T @ Xl
    return RW

def _total_cov(X, tmask):
    T, D, N = X.shape
    Tsel = int(tmask.sum())
    xbar = np.zeros((Tsel, D))
    for l in range(N):
        xbar += np.asarray(X[:, :, l][tmask], dtype=np.float64)
    xbar /= N
    xbar -= xbar.mean(axis=0)
    return N**2 * (xbar.T @ xbar)

def _test_sum_isc(X, tmask, V, n_components):
    T, D, N = X.shape
    Vk = V[:, :n_components]
    Tsel = int(tmask.sum())
    Y = np.empty((Tsel, N, n_components))
    for l in range(N):
        Y[:, l, :] = np.asarray(X[:, :, l][tmask], dtype=np.float64) @ Vk
    total = 0.0
    for c in range(n_components):
        Yc = Y[:, :, c] - Y[:, :, c].mean(axis=0, keepdims=True)
        rT = np.sum(Yc.sum(axis=1) ** 2)
        rW = np.sum(Yc ** 2)
        total += (1.0 / (N - 1)) * (rT - rW) / rW
    return total

def select_gamma(X, gammas=None, n_components=3, n_folds=5):
    """Pick shrinkage gamma by maximizing held-out test ISC (paper §3.5).
    Covariances computed once per fold and reused across gammas; never copies all of X."""
    T, D, N = X.shape
    if gammas is None:
        gammas = np.round(np.linspace(0.0, 0.9, 19), 3)
    bounds = np.linspace(0, T, n_folds + 1).astype(int)
    scores = np.full((len(gammas), n_folds), np.nan)
    for fi in range(n_folds):
        is_test = np.zeros(T, bool); is_test[bounds[fi]:bounds[fi+1]] = True
        RW_tr = _within_cov(X, ~is_test)                 # once per fold
        RB_tr = _total_cov(X, ~is_test) - RW_tr
        for gi, g in enumerate(gammas):
            try:
                eigvals, V = eigh(RB_tr, shrinkage(RW_tr, g))
                V = V[:, np.argsort(eigvals)[::-1]]
                scores[gi, fi] = _test_sum_isc(X, is_test, V, n_components)
            except np.linalg.LinAlgError:
                scores[gi, fi] = np.nan
        print(f"fold {fi+1}/{n_folds} done")
    mean_score = np.nanmean(scores, axis=1)
    valid = ~np.isnan(mean_score)
    best = gammas[valid][int(np.argmax(mean_score[valid]))]
    return best, gammas, mean_score