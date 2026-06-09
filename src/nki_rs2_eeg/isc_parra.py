#%%
import numpy as np
from scipy.linalg import eigh
from nki_rs2_eeg.config import (CONCAT_DATA_DIR, DERIVATIVES_DIR, GAMMA, N_COMPONENTS, TASK_ID, SESSION_ID, RUN_ID)
                    
#%%

def compute_ISC(r_B, r_W, N):
    """
    Compute Inter-Subject Correlation.

    Formula (equation 2 in the paper):
    ISC = (1 / N-1) * (r_B / r_W)

    Parameters
    ----------
    r_B : float
        Between-subject covariance
    r_W : float
        Within-subject covariance
    N   : int
        Number of subjects

    Returns
    -------
    float
        ISC value
    """
    return (1 / (N - 1)) * (r_B / r_W)


# per subject ISC
def compute_per_subject_ISC(R):
    """
    Compute ISC for each subject using the pairwise covariance matrix R.

    ISC_k = r_B_k / r_W_k
    where:
    r_B_k = sum_{l != k} R[k, l]  (between-subject covariance)
    r_W_k = R[k, k]                 (within-subject variance)

    Parameters
    ----------
    R : np.array, shape (N, N)
        Pairwise covariance matrix across subjects.

    Returns
    -------
    isc_values : np.array, shape (N,)
        ISC value for each subject.
    """
    N = R.shape[0]
    isc_values = np.zeros(N)

    for k in range(N):
        numerator = 0
        denominator = 0
        for l in range(N):
            if k == l:
                continue
            numerator += R[k, l] + R[l, k]
            denominator += R[k, k] + R[l, l]
        isc_values[k] = numerator / denominator

    return isc_values



#  with multiple components
def compute_ISC_all_components(X, W, n_components=N_COMPONENTS):
    """
    Compute per-subject ISC for each of the top components.

    Parameters
    ----------
    X            : np.array, shape (N, D, T)
    W            : np.array, shape (D, K)
    n_components : int

    Returns
    -------
    ISC_matrix : np.array, shape (N, n_components)
        ISC value per subject per component
    """
    N = X.shape[0]
    ISC_matrix = np.zeros((N, n_components))

    for comp in range(n_components):
        v = W[:, comp]  # spatial filter for this component
        Y = np.stack([v @ X[n] for n in range(N)], axis=0) # project all subjects
        Y_dm = Y - np.mean(Y, axis=1, keepdims=True)
        R = Y_dm @ Y_dm.T # pairwise covariance of projected data
        ISC_matrix[:, comp] = compute_per_subject_ISC(R)

    return ISC_matrix


#%%



# ============================================================
# FIT CORRCA TO GET W
# ============================================================
def within_subject_cov_old(X):
    """
    Within-subject covariance R_W, eq. (7).
        R_W = sum_i sum_l (x_i^l - xbar_*^l)(x_i^l - xbar_*^l)^T

    X : array (N, C, T)  -- N subjects, C channels, T timepoints
    returns : (C, C) scatter matrix
    """
    #X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=2, keepdims=True)      # center each subject by its temporal mean
    return np.einsum('lci,ldi->cd', Xc, Xc)

def within_subject_cov(X):
    """
    Within-subject scatter R_W, eq. (7):
        R_W = sum_l sum_i (x_i^l - xbar_*^l)(x_i^l - xbar_*^l)^T
    X : array (N, C, T)
    returns : (C, C) scatter matrix
    """
    N, C, T = X.shape
    R_W = np.zeros((C, C), dtype=np.float64)
    for l in range(N):                                  # sum over subjects l
        Xl = np.asarray(X[l], dtype=np.float64)         # (C, T), one subject
        Xc = Xl - Xl.mean(axis=1, keepdims=True)        # center by temporal mean xbar_*^l
        R_W += Xc @ Xc.T                                # sum over i + outer product over c,d
    return R_W

def compute_Rw(X):
    N, D, T = X.shape
    Rw = np.zeros((D, D))
    for n in range(N):
        Rw += np.cov(X[n])
    return Rw

def compute_Rt(X):
    N = X.shape[0]
    return N**2 * np.cov(np.mean(X, axis=0))

def compute_Rb(Rt, Rw, N):
    return (Rt - Rw) / (N - 1)

def regularize(Rw, gamma=0.1):
    D = Rw.shape[0]
    mean_var = np.mean(np.diag(Rw))
    return (1 - gamma) * Rw + gamma * mean_var * np.identity(D)

def fit_corrca(X, gamma=0.1):
    N, D, T      = X.shape
    #Rw           = compute_Rw(X)
    Rw = within_subject_cov(X)
    Rt = compute_Rt(X)
    Rb = compute_Rb(Rt, Rw, N)
    Rw_reg = regularize(Rw, gamma)
    evals, evecs = eigh(Rb, Rw_reg)
    evals = np.real(evals)
    evecs = np.real(evecs)
    sort_idx = np.argsort(evals)[::-1]
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]
    return evecs, evals

#%%


def compute_r_kl_matrix(Y_dm: np.ndarray) -> np.ndarray:
    """Compute the full (N x N) matrix of pairwise scalar covariances between all subjects.

    r_kl = sum_t (y_k_t - mean_k)(y_l_t - mean_l)

    Since Y_dm is already demeaned this simplifies to:
    r_kl = sum_t y_k_t * y_l_t

    Parameters
    ----------
    Y_dm : np.ndarray, shape (N, T)
        Demeaned projected signals

    Returns:
    -------
    R : np.ndarray, shape (N, N)
        R[k, l] = r_kl
    """
    # Matrix multiplication gives all pairwise dot products
    # at once — equivalent to looping over all pairs
    # (N, T) @ (T, N) → (N, N)
    R = Y_dm @ Y_dm.T
    return R


#%%
'''
X = np.load(f"{CONCAT_DATA_DIR}")  # shape (subjects, channels, samples)


print(f"\n=== OUTLIER DETECTION ===")
print(f"Mean ISC:          {ISC_average:.4f}")
print(f"Std ISC:           {std_isc:.4f}")
print(f"Threshold (mean - 2*std): {threshold:.4f}")
print(f"Flagged subjects:  {[s+1 for s in flagged]}")
#%%
# Weighted average  - weight by the group-level ISC per component.  
# Components with higher group ISC contribute more to the final score.

weights = np.maximum(ISC_group[:N_COMPONENTS], 0) / np.sum(ISC_group[:N_COMPONENTS])  # normalize to sum to 1
ISC_weighted_avg = ISC_matrix @ weights

print(f"\n=== MULTI-COMPONENT ISC ===")
print(f"ISC matrix shape: {ISC_matrix.shape}  (subjects x components)")
print(f"\nPer-subject ISC averaged across 5 components:")
for n in range(X.shape[0]):
    #flag = ' ← BAD SUBJECT' if n in flagged else ''
    print(f"  Subject {n+1:2d}: "
          f"simple avg = {ISC_average[n]:+.4f}  "
          f"weighted avg = {ISC_weighted_avg[n]:+.4f}"
          f"")

          '''
# %%


if __name__ == "__main__":
    try:
        if not CONCAT_DATA_DIR.exists():
            print(f"Error: Data file not found at {CONCAT_DATA_DIR}")
            print("Loading the cleaned data and generating the collated data file .")
            from nki_rs2_eeg.write_file import save_collated_condition_data
            save_collated_condition_data(
                session_id=SESSION_ID,
                task_id=TASK_ID,
                run_id=RUN_ID,
                onset_label="Onset Movie",
                offset_label="Offset Movie",
            )
            X = np.load(f"{CONCAT_DATA_DIR}")
        else:
            X = np.load(f"{CONCAT_DATA_DIR}")
        W, ISC_group = fit_corrca(X)
        print(f"\nW shape: {W.shape}")
        print(f"Group ISC Component 1: {ISC_group[0]:.4f}")

        v1 = W[:, 0]  # first spatial filter

        Y = np.stack([v1 @ X[n] for n in range(X.shape[0])], axis=0) # all subjects timeseries projected onto first component
        print(f"\nProjected data shape: {Y.shape}  (subjects x time)")
        Y_dm = Y - np.mean(Y, axis=1, keepdims=True)
        R = compute_r_kl_matrix(Y_dm)
        ISC_subjects = compute_per_subject_ISC(R)
        np.save(f"{DERIVATIVES_DIR}/sub-ALL_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_isc_per_subject.npy", ISC_subjects)
        np.save(f"{DERIVATIVES_DIR}/sub-ALL_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_isc_group.npy", ISC_group)
    except Exception as e:
        print(f"An error occurred: {e}")


#%%




