import random
import numpy as np
from scipy.io import mmread
from scipy.sparse.csgraph import maximum_bipartite_matching
##################################################
# Constants
##################################################
SINKHORN_KNOPP_ITERATIONS = 1000
GRAPHS = ["Graphs/Cities/Cities.mtx",
          "Graphs/farm/farm.mtx",
          "Graphs/mycielskian11/mycielskian11.mtx",
          "Graphs/Sandi_sandi/Sandi_sandi.mtx",
          "Graphs/Trec6/Trec6.mtx",
          "Graphs/WorldCities/WorldCities.mtx",
          "Graphs/Trec13/Trec13.mtx",
          "Graphs/mycielskian12/mycielskian12.mtx"]

##################################################
# Sinkhorn-Knopp Scaling
##################################################
def normalize_rows(A):
    row_sum = np.sum(A, axis = 1)
    return A / np.atleast_2d(np.where(row_sum != 0, row_sum, 1)).T

def normalize_cols(A):
    col_sum = np.sum(A, axis = 0)
    return A / np.where(col_sum != 0, col_sum, 1)

def sinkhorn_knopp(A, num_iterations):
    for i in range(num_iterations):
        A = normalize_rows(A)
        A = normalize_cols(A)
    return A

##################################################
# Random Index Selection
##################################################
def get_row_sums(S):
    R = np.zeros(S.shape)
    R[0] = S[0]
    for i in range(1, S.shape[0]):
        R[i] = R[i - 1] + S[i]
    return R

def get_col_sums(S):
    C = np.zeros(S.shape)
    C[:,0] = S[:,0]
    for i in range(1, S.shape[1]):
        C[:,i] = C[:,i - 1] + S[:,i]
    return C

def get_row_index(R, c):
    num_rows = R.shape[0]
    x = random.random() * R[num_rows - 1][c]
    return np.searchsorted(R[:,c], x)

def get_col_index(C, r):
    num_cols = C.shape[1]
    x = random.random() * C[r][num_cols - 1]
    return np.searchsorted(C[r], x)

##################################################
# Karp Sipser (Only rules 0 and 1)
##################################################

def karp_sipser(E, row_deg, col_deg):
    #print(E)
    #print(row_deg)
    #print(col_deg)
    matched_edge = True
    dim = E.shape
    M = []
    while matched_edge:
        matched_edge = False
        #where returns tuple, [0] gives vector of indices
        deg_1_rows = np.where(row_deg == 1)[0]
        if len(deg_1_rows) > 0:
            matched_edge = True
            r = deg_1_rows[0]
            #where returns tuple, [0][0] gives first (only) index in vector of indices
            c = np.where(E[r] == 1)[0][0]
            M.append((r, c))
            #Update degrees
            c_rows = np.where(E[:,c] == 1)[0]
            for row in c_rows:
                row_deg[row] -= 1
            E[:,c] = np.zeros(dim[0])
            col_deg[c] = 0
        else:
            #where returns tuple, [0] gives vector of indices
            deg_1_cols = np.where(col_deg == 1)[0]
            if len(deg_1_cols) > 0:
                matched_edge = True
                c = deg_1_cols[0]
                #where returns tuple, [0][0] gives first (only) index in vector of indices
                r = np.where(E[:,c] == 1)[0][0]
                M.append((r, c))
                #Update degrees
                r_cols = np.where(E[r] == 1)[0]
                for col in r_cols:
                    col_deg[col] -= 1
                E[r] = np.zeros(dim[1])
                row_deg[r] = 0
            else:
                nonzero_rows = np.nonzero(row_deg)[0]
                if len(nonzero_rows) > 0:
                    matched_edge = True
                    r = nonzero_rows[0]
                    c = np.where(E[r] == 1)[0][0]
                    M.append((r, c))
                    #Update degrees
                    r_cols = np.where(E[r] == 1)[0]
                    for col in r_cols:
                        col_deg[col] -= 1
                    E[r] = np.zeros(dim[1])
                    row_deg[r] = 0
                    c_rows = np.where(E[:,c] == 1)[0]
                    for row in c_rows:
                        row_deg[row] -= 1
                    E[:,c] = np.zeros(dim[0])
                    col_deg[c] = 0
    return M

##################################################
# Two Sided Algorithm
##################################################

def two_sided(A):
    S = sinkhorn_knopp(A, SINKHORN_KNOPP_ITERATIONS)
    R = get_row_sums(S)
    C = get_col_sums(S)
    dim = A.shape
    E = np.zeros(dim)
    row_deg = np.zeros(dim[0])
    col_deg = np.zeros(dim[1])
    excluded_rows = []
    excluded_cols = []
    for r in range(dim[0]):
        c = get_col_index(C, r)
        if A[r][c] == 0:
            #print("Selected invalid column vertex (or row vertex has degree zero)")
            excluded_rows.append(r)
        else:
            E[r][c] = 1
            row_deg[r] += 1
            col_deg[c] += 1
    for c in range(dim[1]):
        r = get_row_index(R, c)
        if A[r][c] == 0:
            #print("Selected invalid row vertex (or column vertex has degree zero)")
            excluded_cols.append(c)
        else:
            if E[r][c] == 0:
                E[r][c] = 1
                row_deg[r] += 1
                col_deg[c] += 1
    print("Excluded", len(excluded_rows), "rows")
    print("Excluded", len(excluded_cols), "columns")
    return karp_sipser(E, row_deg, col_deg)

##################################################
# Read graph data, and start algorithm
##################################################

#Graph Index
graph_index = 7

#Graph Data
G = mmread(GRAPHS[graph_index])
A = np.where(G.toarray() != 0, 1, 0)

print(GRAPHS[graph_index])
print("Dim:", G.shape, "Edges:", G.nnz)
print(A)
M = two_sided(A)
print("Matching Cardinality:", len(M))#, "Matching:", M)
max_cardinality = len(np.where(maximum_bipartite_matching(G) != -1)[0])
print("Maximum Cardinality:", max_cardinality)
print("Ratio:", len(M) / max_cardinality)
