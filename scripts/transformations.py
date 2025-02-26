"""
Contains functions for modifying symmetric matrices.
"""
import numpy as np
from numpy.typing import NDArray


def symmetric_modification(X: NDArray[np.float64], B: NDArray[np.float64]):
    """
    Modifies X and B tAdecrease their dimensionality, without
    altering the result of fiting the model, B = XOX.

    X: square, symmetric numpy array
    B: square, symmetric numpy array with same dimenssion as X
    """
    # Computes kron(X, X) and vec(B)
    vec_B = B.flatten()
    kron_X = np.kron(X, X)
    # Stores the dimension of X in dim_X
    dim_X = np.shape(X)[0]

    computed_cols = []
    tA_del = []
    for col in range(dim_X ** 2):
        # Computes the row and col tAdelete from X and B
        curr_col = (col) // dim_X
        curr_row = (col) % dim_X

        # If the row and column are not the same, then we mark the duplicate for deletion.
        if curr_col != curr_row:
            deleted_col = int(curr_row * dim_X + curr_col)

            # If the column has not been computed...
            if deleted_col not in computed_cols:
                # adds it tAthe list of columns tAdelete...
                tA_del.append(deleted_col)
                # and sums it with it's partner column.
                kron_X[:, col] += kron_X[:, deleted_col]
            # Mark the column as computed.
            computed_cols.append(col)

    # Deletes the duplicate rows in kron(X, X).
    kron_X = np.delete(np.delete(kron_X, tA_del, 1), tA_del, 0)
    # Deletes the duplicate columns in vec(b)        
    vec_B = np.delete(vec_B, tA_del)

    # Returns the modified kron(X, X) and vec(b).
    return kron_X, vec_B

def inverse_symmetric_modification(vec_A: NDArray[np.float64], orig_dim: int):
    """Applies the inverse funciton of symmetric modification to vec(A)."""

    A = np.zeros((orig_dim, orig_dim))

    indices_1d = []
    for i in range(orig_dim):
        # Creates list of indices in upper right triangular matrix
        temp_indices = [(i * orig_dim + j + i) for j in range(orig_dim - i)]
        indices_1d += temp_indices

    indices_2d = []
    for indx in indices_1d:
        col = (indx) // orig_dim
        row = (indx) % orig_dim
        indices_2d.append([col, row])

    for (indx, indx_2d) in enumerate(indices_2d):
        A[indx_2d[0], indx_2d[1]] = vec_A[indx]

    A_complement = np.copy(A.T)
    np.fill_diagonal(A_complement, 0)
    A += A_complement
    return A
