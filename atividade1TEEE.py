'''
O código a seguir é uma adaptação, que faz pequenas mudanças,  a fim de
aplicar o TMM para modos TM (atividade sugerida em aula).
O código original pode ser encontrado em: https://github.com/adophobr/PhotonicIntegratedCircuits/blob/main/jupyter/ass_dlt_guide.ipynb

'''


import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import scimath as SM
import warnings

warnings.filterwarnings('ignore') 

nmbOfPoints = 1000
lmbd = 1e-6
k0 = 2*np.pi/lmbd

'''
#exemplo 3
n = np.array([1.45,3.5,1.45])
d = np.array([0,0.25e-6,0])


#exemplo 4

n = np.array([1.45, 1.56, 1.45, 1.56, 1.45])
d = np.array([0, 0.75e-6, 0.5e-6, 0.75e-6, 0])
'''
#exemplo 4

n = np.array([1.45, 1.56, 1.45, 1.56, 1.45])
d = np.array([0, 0.75e-6, 0.5e-6, 0.75e-6, 0])

nmbOfLayers = np.size(n)
neff = np.linspace(0, 4, nmbOfPoints)
beta = neff*k0

alpha = np.zeros([nmbOfPoints, nmbOfLayers], dtype=np.complex64)
sigma = np.zeros([nmbOfPoints, nmbOfLayers], dtype=np.complex64)

alpha[:,0] = SM.sqrt(beta**2 - (k0*n[0])**2)
sigma[:,0] = alpha[:,0]*d[0]


Twg = np.zeros([2, 2, nmbOfPoints], dtype=np.complex64)
Twg[0,0,:] = 1
Twg[1,1,:] = 1
Tj = np.zeros([2, 2, nmbOfPoints], dtype=np.complex64)
for i in range(nmbOfLayers-1):
  alpha[:,i+1] = SM.sqrt(beta**2 - (k0*n[i+1])**2)
  sigma[:,i+1] = alpha[:,i+1]*d[i+1]
  Tj[0,0,:] = 0.5 * (1+alpha[:,i]/alpha[:,i+1])*np.exp( sigma[:,i])
  Tj[0,1,:] = 0.5 * (1-alpha[:,i]/alpha[:,i+1])*np.exp(-sigma[:,i])
  Tj[1,0,:] = 0.5 * (1-alpha[:,i]/alpha[:,i+1])*np.exp( sigma[:,i])
  Tj[1,1,:] = 0.5 * (1+alpha[:,i]/alpha[:,i+1])*np.exp(-sigma[:,i])
  Twg = np.einsum('mnr,ndr->mdr', Tj, Twg)

TwgTm = np.zeros([2, 2, nmbOfPoints], dtype=np.complex64)
TwgTm[0,0,:] = 1
TwgTm[1,1,:] = 1
TjTm = np.zeros([2, 2, nmbOfPoints], dtype=np.complex64)
for i in range(nmbOfLayers-1):
  alpha[:,i+1] = SM.sqrt(beta**2 - (k0*n[i+1])**2)
  sigma[:,i+1] = alpha[:,i+1]*d[i+1]
  TjTm[0,0,:] = 0.5 * (1+alpha[:,i]/alpha[:,i+1])*np.exp( 1j*sigma[:,i])
  TjTm[0,1,:] = 0.5 * (1-alpha[:,i]/alpha[:,i+1])*np.exp(-1j*sigma[:,i])
  TjTm[1,0,:] = 0.5 * (1-alpha[:,i]/alpha[:,i+1])*np.exp( 1j*sigma[:,i])
  TjTm[1,1,:] = 0.5 * (1+alpha[:,i]/alpha[:,i+1])*np.exp(-1j*sigma[:,i])
  TwgTm = np.einsum('mnr,ndr->mdr', TjTm, TwgTm)

fig1 = plt.figure(figsize=(15,5))
plt.plot(neff, Twg[0,0,:], label = 'TMM TE', linewidth = 2)
plt.plot(neff, TwgTm[0,0,:], label = 'TMM TM', linewidth = 2)
plt.locator_params(nbins=16)
plt.xlim([n.min()*0.999, n.max()*1.001])
plt.ylim([-5,5])
plt.grid(True)
plt.xlabel(r'$n_{eff}$')
plt.legend(loc = 'upper left')

neffTE1 = 1.508509
neffTE0 = 1.5215379
#1.496048287587425 - TE0
#1.484221244965693 - TE1
neffTM0 = 1.516064

beta = neffTE1*k0

alpha = SM.sqrt(beta**2 - (k0*n)**2)
sigma = alpha*d

A = np.zeros(nmbOfLayers, dtype=np.complex128)
B = np.zeros(nmbOfLayers, dtype=np.complex128)
# Equation 16
A[0] = 1
B[0] = 0

Tj = np.zeros([2, 2], dtype=np.complex128)
for i in range(nmbOfLayers-1):
  Tj[0,0] = (1+alpha[i]/alpha[i+1])*np.exp( sigma[i])/2
  Tj[0,1] = (1-alpha[i]/alpha[i+1])*np.exp(-sigma[i])/2
  Tj[1,0] = (1-alpha[i]/alpha[i+1])*np.exp( sigma[i])/2
  Tj[1,1] = (1+alpha[i]/alpha[i+1])*np.exp(-sigma[i])/2
  #Equation 13
  A[i+1] = Tj[0,0] * A[i] + Tj[0,1] * B[i]
  B[i+1] = Tj[1,0] * A[i] + Tj[1,1] * B[i]

t  = np.cumsum(d)
lS = np.insert(t, 0, 0)
if d[0] == 0.0:
    csL = 2e-6
    t = np.insert(t, 0, -csL)
    t[-1] = t[-1] + csL
else:
    t = np.insert(t, 0, 0)
    t[-1] = t[-1] + d[-1]

dx = np.sum(d)/100
Ey_j = []
lent = []
for i in range(nmbOfLayers):
    len = np.arange(t[i], t[i+1], dx)
    # Equation 4
    E_temp = A[i] * np.exp(+alpha[i] * (len - lS[i])) + B[i] * np.exp(-alpha[i] * (len - lS[i]))
    Ey_j = np.hstack((Ey_j, E_temp))
    lent = np.hstack((lent, len))

fig2 = plt.figure(figsize=(8,6))
plt.plot(lent/1e-6, np.real(Ey_j/Ey_j.max()), linewidth = 2)
plt.grid()
plt.xlim([lent.min()/1e-6,lent.max()/1e-6])
plt.ylim([-1,1])
plt.xlabel('x')
plt.ylabel('Field amplitude')

#-=-=-=-=-=-=-=-=-=-= adaptação para plotar Hy -=-=-=-=-=-=-=-=-=-=-= #

neffTM0 = 1.516064
#neffTM1 = 1.452971

beta = neffTM0*k0

alpha = SM.sqrt(beta**2 - (k0*n)**2)
sigma = alpha*d

C = np.zeros(nmbOfLayers, dtype=np.complex128)
D = np.zeros(nmbOfLayers, dtype=np.complex128)
# Equation 16
C[0] = 1
D[0] = 0

Tj = np.zeros([2, 2], dtype=np.complex128)
for i in range(nmbOfLayers-1):
  Tj[0,0] = (1+alpha[i]/alpha[i+1])*np.exp( 1j*sigma[i])/2
  Tj[0,1] = (1-alpha[i]/alpha[i+1])*np.exp(-1j*sigma[i])/2
  Tj[1,0] = (1-alpha[i]/alpha[i+1])*np.exp( 1j*sigma[i])/2
  Tj[1,1] = (1+alpha[i]/alpha[i+1])*np.exp(-1j*sigma[i])/2
  #Equation 13
  C[i+1] = Tj[0,0] * C[i] + Tj[0,1] * D[i]
  D[i+1] = Tj[1,0] * C[i] + Tj[1,1] * D[i]

t  = np.cumsum(d)
lS = np.insert(t, 0, 0)
if d[0] == 0.0:
    csL = 2e-6
    t = np.insert(t, 0, -csL)
    t[-1] = t[-1] + csL
else:
    t = np.insert(t, 0, 0)
    t[-1] = t[-1] + d[-1]

dx = np.sum(d)/100
Hy_j = []
lent = []
for i in range(nmbOfLayers):
    len = np.arange(t[i], t[i+1], dx)
    # Equation 4
    H_temp = C[i] * np.exp(+1j*alpha[i] * (len - lS[i])) + D[i] * np.exp(-1j*alpha[i] * (len - lS[i]))
    Hy_j = np.hstack((Hy_j, H_temp))
    lent = np.hstack((lent, len))

fig3 = plt.figure(figsize=(8,6))
plt.plot(lent/1e-6, np.real(Hy_j/Hy_j.max()), linewidth = 2)
plt.grid()
plt.xlim([lent.min()/1e-6,lent.max()/1e-6])
plt.ylim([-1,1])
plt.xlabel('x')
plt.ylabel('Field amplitude')



plt.show()