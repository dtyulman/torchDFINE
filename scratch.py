import numpy as np
import scipy


x = 1
u = 1

A = np.diag([1.004])
B = np.diag([-0.55])
Q = 0*np.eye(x)
R = 1e5*np.eye(u)

K = lambda P: np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

#iteratively solve DARE
P1 = np.random.rand(x,x)
P2 = np.random.rand(x,x)
for i in range(int(1e5)):
    P2 = A.T @ P1 @  A - A.T @ P1 @ B @ K(P1) + Q
    if np.allclose(P1,P2) and i >= 1e5 or (P1==P2).all():
        print(i, np.mean(np.abs(P1-P2)))
        break
    if i % 10000 == 0:
        print(i, np.mean(np.abs(P1-P2)))
    P1 = P2


#compare to scipy's solution
print('P1=', P1, 'K1=', K(P1))
P = scipy.linalg.solve_discrete_are(A,B,Q,R)
if np.allclose(P1,P):
    print('close to scipy:')
print('P =', P, 'K =', K(P))

#%%
