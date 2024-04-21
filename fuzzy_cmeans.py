import numpy as np
# A function to calculate d^2(X_{n m} - C{k m})
def distance(x, c):
    return (np.sum((x - c)) ** 2)
# A function to calculate F which is rate of 
def cal_F(U, C, W):
    F = 0
    for n in range(N):
        for k in range(K):
            for m in range(M):
                F += U[n, k] ** alpha * W[k, m] ** q * distance(X[n, m], C[k, m])
    return F
# A function to calculate U, where u_{n k} represents the degree of membership of the nth sample to the kth cluster
def cal_U():
    for n in range(N):
        for k in range(K):
            denominator_pro_max = 0
            for l in range(K):
                numerator = 0
                denominator = 0
                for m in range(M):
                    numerator += W[k, m] ** q * distance(X[n, m], C[k, m])
                    denominator += W[l, m] ** q * distance(X[n, m], C[l, m])
                denominator_pro_max += (numerator / denominator) ** (1 / (alpha - 1)) # :)))
            U[n, k] = 1 / denominator_pro_max
    return U
# A function to calculate C, where c_{k m} represents the mth feature value of the centroid of the kth cluster
def cal_C():
    for k in range(K):
        for m in range(M):
            numerator = np.sum(np.fromiter(((U[n, k] ** alpha * distance(X[n, m], C[k, m]) * X[n,m]) for n in range(N)), dtype='float32'))
            denominator = np.sum(np.fromiter(((U[n, k] ** alpha * distance(X[n, m], C[k, m])) for n in range(N)), dtype='float32'))
            C[k, m] = numerator / denominator
    return C
# A function to calculate Dw_{k m}, which is going to be used to calculate W
def cal_DW(k, m):
    dw = np.sum(np.fromiter(((U[n, k] ** alpha * distance(X[n, m], C[k, m])) for n in range(N)), dtype='float32'))
    return dw
# A function to calculate W, where w_km represents the weight assigned to the mth feature in the kth cluster
def cal_W():
    for k in range(K):
        for m in range(M):
            dw_km = cal_DW(k, m)
            if dw_km == 0:  # if Dw_{k m} == 0, then W_{k m} is 1/h_m, where h_m is the count of 0 in the kth row of the Dw matrix
                h_m = len([s for s in range(M) if cal_DW(k, s) == 0])
                W[k, m] = 1 / h_m
            elif any(cal_DW(k, s) == 0 for s in range(M)):  # if dw_km != 0 and there is at least one 0 in the kth row of the Dw matrix, W_{k m} is 0
                W[k, m] = 0
            else:   # if dw_km != 0 and there is no 0 in the kth row of the Dw matrix, then W_{k m} is as follows
                W[k, m] = 1 / (np.sum(np.fromiter(((dw_km / cal_DW(k, s)) ** (1 / (q - 1)) for s in range(M)), dtype='float32')))
    return W
# MAIN
X = np.genfromtxt("./iris.csv", delimiter=',') # Save our data set in an array
N = X.shape[0]  # Number of samples in data set
M = X.shape[1]  # Number of features
K = 3   # Number of clusters
t_max = 200 # Maximum repetitions
t = 0   # current turn
epsilon = 10 ** (-1)
alpha = 2   # Fuzzy degree
q = 2   # Exponent of attribute weight

# Initialize the arrays 
W = np.full((K, M), 1/M)
C = np.random.choice(X.flatten(), size=(K, M))
U = np.full((N, K), 1/K)
F=[0, 0]
# The main loop
while (t < t_max):
    t += 1
    U = cal_U() # Update cluster assignment matrix
    C = cal_C() # Update cluster center matrix
    W = cal_W() # Update feature weight matrix
    F[0] = F[1]
    F[1] = cal_F(U, C, W)
    if (abs(F[1] - F[0]) <= epsilon):
        break

print("\n\n******\nRepetitions:\t  ", t, "\nPrecision:   ", abs(F[1]-F[0]))
#limit the print operator to 2 decimal digits (to avoid showing values like "4.31426571e-01")
np.set_printoptions(precision=2, suppress=True)

print("\n\nU:\n\n", U)
print("\n\nC:\n\n", C)
print("\n\nW:\n\n", W, "\n******")

# Save U and C in csv files
np.savetxt("U.csv", U, delimiter=",", fmt="%.2f")
np.savetxt("C.csv", C, delimiter=",", fmt="%.2f")
