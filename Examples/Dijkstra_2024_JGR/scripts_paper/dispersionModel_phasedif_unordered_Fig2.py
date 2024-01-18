import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import step as st
import nifty as ny
colours = st.configure()
Ls_contour = 1

"""
Linear dispersion model
Ast = Qs_x +AKh s_xx = 0
with s(0,t)=ssea

Solved assuming Q=Q^0+Q^1(t)
All other parameters constant.
Q^1(t) is assumed periodic -> use FT.
"""

def fs(t, c):
    return np.sum(np.real(c*np.exp(np.arange(len(c))*1j*omega*t)))

def fs_vec(t, c):
    c = c.reshape((len(c), 1))
    t = t.reshape((1,len(t)))
    n = np.arange(len(c)).reshape((len(c),1))
    return np.sum(np.real(c*np.exp(n*1j*omega*t)), axis=0)

########################################################################################################################
# Parameters
########################################################################################################################
Kh0 = 100
Kh1 = 60
A = 1e4
ssea = 30
Q0 = 100
Q1 = 0
k = 3.4
N = 15
L = k*A*Kh0/Q0
jmax = 100
tmax = 500 # for fft in post-processing
T = 20*A**2*Kh0/Q0**2
print(L)
########################################################################################################################
# Prep
#######################################################################################################################
Tadj0 = ((Q0/(A*Kh0))**2*Kh0)**(-1)
t = np.linspace(0,T,tmax)
iso = ssea*np.exp(-k)

########################################################################################################################
# Computation
########################################################################################################################
omega = 2*np.pi/T
Qmat = np.diag(Q0*np.ones(2*N+1)) + .5*np.diag(np.conj(Q1)*np.ones(2*N),1) + .5*np.diag(Q1*np.ones(2*N),-1)
Dmat = np.diag(np.arange(-N,N+1)*1j*omega)
AKHmat = A*(np.diag(Kh0*np.ones(2*N+1)) + .5*np.diag(np.conj(Kh1)*np.ones(2*N),1) + .5*np.diag(Kh1*np.ones(2*N),-1))

B = np.zeros((2*(2*N+1), 2*(2*N+1)), dtype=complex)
C = np.zeros((2*(2*N+1), 2*(2*N+1)), dtype=complex)
B[:2*N+1,:2*N+1] = Qmat
B[:2*N+1:,2*N+1:] = AKHmat
B[2*N+1:,:2*N+1] = np.eye(2*N+1)

C[:2*N+1,:2*N+1] = A*Dmat
C[2*N+1:,2*N+1:] = np.eye(2*N+1)

M = np.matmul(np.linalg.inv(B),C)
l, P = np.linalg.eig(M)
indices = [i for i in range(0, len(l)) if np.real(l[i])<-1e-19]
# rows = [N] + list(range(2*N+1,3*N+1)) + list(range(3*N+2,4*N+2))    # pseudo-neumann
rows = [N] + list(range(0,N)) + list(range(N+1,2*N+1))    # dirichlet
b = np.zeros(2*N+1)
b[0] = ssea

l_neg = l[indices]
if len(l_neg)!=2*N+1:
    print(l_neg)
P_neg = P[:, indices]
P_reduced = P_neg[rows,:]
c = np.dot(np.linalg.inv(P_reduced), b)

s00 = np.dot(np.matmul(P_neg,np.diag(np.exp(l_neg*0))), c)
s0max = np.max(ny.invfft2(s00,0,1500))
sigma = np.dot(np.matmul(P_neg,np.diag(np.exp(l_neg*L))), c)
s = ny.eliminateNegativeFourier(sigma[:2*N+1],0)/s0max*30
s0 = s[0]
s1 = s[1]

########################################################################################################################
# time dependent adjustment time
########################################################################################################################
s_time = ny.invfft2(s,0,tmax)
Ls = (-1/L*np.log(s_time/30))**(-1)
Q_time = Q0 + Q1*np.cos(2*np.pi*np.linspace(0,1,tmax))
Kh_time = Kh0 + Kh1*np.cos(2*np.pi*np.linspace(0,1,tmax))

Tdelay = (-(np.angle(s1)-np.angle(Q1))/(2*np.pi))*T
phi = np.angle(s1)*180/np.pi
print(Tdelay/(3600*24))
########################################################################################################################
# Plot
########################################################################################################################
### Fig 1 ###
plt.figure(1, figsize=(1,1))
plt.plot(t/(3600*24), Kh_time)
plt.ylabel('$K_h$ ($m^2/s$)')
plt.ylim(0,200)
plt.xlabel('t (d)')

plt.twinx()
plt.plot(t/(24*3600), s_time, color=colours[1])
ax1 = plt.gca()
ax1.set_ylabel('s at $x/L_0=3.4$ (psu)', color=colours[1])
ax1.tick_params(axis='y', labelcolor=colours[1])
plt.ylim(0, 3)



st.show()

