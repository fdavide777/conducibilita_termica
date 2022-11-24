import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#d_al = np.array([7.,9.5,12.,14.5,17,19.5,22.,24.5,27.,29.5,32.,34.5,37.,39.5,42.])
#d_ot = np.array([65.,8.7,10.8,13,15.1,17.3,19.4,21.6,23.7,25.9,28.0,30.1,32.3,34.4,36.6,38.7,40.9,43.1,45.2,47.4])
d_al = np.array([7.,12.,17.,22.,27.,32.,37.,42.])
d_ot = np.array([6.5,10.8,15.1,19.4,23.7,28.0,32.3,36.6,40.9,45.2,47.4])

sigma_d_al = np.full(d_al.shape, 0.1)
sigma_d_ot = np.full(d_ot.shape, 0.1)
print(sigma_d_al)
print(sigma_d_ot)

def line(x, m, q):
    '''Modello di  fit lineare'''
    return m * x + q

t_al = np.array([41.3,37.0,33.7,31.0,28.0,26.6,23.0,20.4])
t_ot = np.array([31.7,31.5,28.9,29.,27.3,25.5,24.7,23.2,21.7,20.,19.8])

sigma_t_al= np.array([.2,.2,.2,.1,.1,.1,.1,.1])
sigma_t_ot= np.array([.3,.3,.2,.2,.2,.2,.2,.2,.2,.2,.2])

#alluminio
plt.figure('Grafico Posizione-Temperatura - [Alluminio]')
plt.errorbar(d_al,t_al, sigma_t_al, sigma_d_al, fmt='|', color ='tomato', label = 'Barre di errore')
plt.scatter(d_al,t_al, marker='o', color = 'crimson', label = 'Dati raccolti')

popt_al, pcov_al = curve_fit(line, d_al, t_al, sigma=sigma_t_al, p0=[0.5,40])
m_al_hat, q_al_hat = popt_al

sigma_m_al, sigma_q_al = np.sqrt(pcov_al.diagonal())
print('VALORE PENDENZA RETTA ALLUMINIO = ',m_al_hat,u"\u00B1", sigma_m_al)
print('VALORE INTERCETTA RETTA ALLUMINIO = ', q_al_hat, u"\u00B1", sigma_q_al)

x = np.linspace(0., 40., 100)
plt.plot(x, line(x, m_al_hat, q_al_hat), label = 'Fit')
plt.xlabel('Posizione [cm]')
plt.ylabel('Temperatura [C°]')
plt.grid(which='both', ls='dashed', color='gray')
plt.savefig('posizione_temperatura.pdf')
x = np.linspace(0., 50, 100)
plt.legend()
plt.plot(x, line(x, m_al_hat, q_al_hat))

#rame
plt.figure('Grafico Posizione-Temperatura - Rame')
plt.errorbar(d_ot,t_ot, sigma_t_ot, sigma_d_ot, fmt='|', color = 'orchid', label = 'Barre di errore')
plt.scatter(d_ot,t_ot, marker='o', color = 'plum', label = 'Dati raccolti')

popt_ot, pcov_ot = curve_fit(line, d_ot, t_ot, sigma=sigma_t_ot, p0=[0.5,40])
m_ot_hat, q_ot_hat = popt_ot

sigma_m_ot, sigma_q_ot = np.sqrt(pcov_ot.diagonal())
print('VALORE PENDENZA RETTA OTTONE = ',m_ot_hat, u"\u00B1", sigma_m_ot)
print('VALORE INTERCETTA RETTA OTTONE = ',q_ot_hat, u"\u00B1", sigma_q_ot)

x = np.linspace(0., 40., 100)
plt.plot(x, line(x, m_ot_hat, q_ot_hat), label = 'Fit')
plt.xlabel('Posizione [cm]')
plt.ylabel('Temperatura [C°]')
plt.grid(which='both', ls='dashed', color='gray')
plt.savefig('posizione_temperatura.pdf')
x = np.linspace(0., 50, 100)
plt.legend()
plt.plot(x, line(x, m_ot_hat, q_ot_hat))

#residui alluminio
plt.figure('residui_al')

r_al = t_al - line(d_al, m_al_hat, q_al_hat)
r_al=(r_al/sigma_t_al)
print(r_al)

plt.errorbar(d_al, r_al, 1, fmt='.r', label='Scarti Errore Misura')
plt.xlabel("Distanza [cm]")
plt.ylabel("Residui")
plt.legend()
plt.grid(which='both', ls='dashed', color='gray')
plt.axhline(0, color='black')
plt.show()


#residui rame
plt.figure('residui_rm')

r_ot = t_ot - line(d_ot, m_ot_hat, q_ot_hat)
r_ot=(r_ot/sigma_t_ot)
print(r_ot)

plt.errorbar(d_ot, r_ot, 1, fmt='.r', label='Scarti Errore Misura')
plt.xlabel("Distanza [cm]")
plt.ylabel("Residui")
plt.legend()
plt.grid(which='both', ls='dashed', color='gray')
plt.axhline(0, color='black')
plt.show()


