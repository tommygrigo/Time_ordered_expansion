import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import pandas as pd
from tqdm import tqdm
from forbiddenfruit import curse

curse(np.ndarray,'H',property(fget=lambda A: A.conj().T))


#### Parameters

N1 = 120
N2 = 80

N = N1+N2

DeltaE = 1
deltae = .5
lam = 3e-3

Nt = 300
dt = 1

#### Model definition

S0 = DeltaE*np.array([[1,0],[0,-1]])
S1 = np.eye(2)
S2 = np.array([[0,1],[1,0]])
S3 = np.array([[0,-1j],[1j,0]])

E0 = np.eye(N)
E1 = np.zeros_like(E0)
E1[:N1,:N1] = np.diag((np.arange(N1)+1)*deltae/N1)
E1[N1:,N1:] = np.diag(DeltaE+(N1+np.arange(N2)+1)*deltae/N2)

B = np.zeros_like(E0)
B[N1:,:N1] = np.random.randn(N2*N1).reshape([N2,N1])
#B[:N1,N1:] = np.random.randn(N2*N1).reshape([N1,N2])
B = B*N1*N2/(np.sum(B**2))
B = lam*B

E2 = (B + B.H)/2
E3 = 1j*(B - B.H)/2

Es = [E0,E1,E2,E3]


H_tot = np.kron(S0,E0) + np.kron(S1,E1) + np.kron(S2,E2) #+ np.kron(S3,E3)

U = sp.linalg.expm(-1j*H_tot*dt)

c = np.kron(np.array([[0,0],[0,1]]), np.eye(N))

#### Simulation

ts = np.arange(Nt+1)*dt

s0 = np.array([[0,0],[0,1]])
e0 = np.zeros([N,N])
e0[N1:,N1:] = np.eye(N2)/N2
#e0[:N1,:N1] = np.eye(N1)/N1
rho_0 = np.kron(s0,e0)

rho_t = rho_0

true_traj = [np.real(np.trace(c@rho_t))]
for t in tqdm(ts[:-1]):
    rho_t = U @ rho_t @ U.H
    true_traj.append(np.real(np.trace(c@rho_t)))

true_traj = np.array(true_traj)

#### Data saving

gamma_1 = 2*np.pi*lam**2*N1/deltae
gamma_2 = 2*np.pi*lam**2*N2/deltae

gamma = gamma_1+gamma_2

header = ['t', 'y']
df = pd.DataFrame(np.array([ts,true_traj]).T, columns=header)
df.to_csv('highly_non_markovian_exact.csv',index=False)

plt.figure()
plt.plot(ts,true_traj,'k',label='exact')
plt.plot(ts,gamma_1/gamma + gamma_2/gamma * np.exp(-gamma*ts),label='HAM')
plt.grid()
plt.ylabel(r'$\rho_{11}(t)$')
plt.xlabel('$t$')

plt.legend()
plt.show()