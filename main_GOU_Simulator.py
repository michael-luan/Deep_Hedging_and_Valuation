# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:18:01 2022

@author: sebja
"""

from GOU_Simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np

model = Simulator()
S = model.Sim(10_000)


plt.plot(model.t, S[:500,:].T, alpha=0.1, linewidth=1)

lvls = [0.1,0.5,0.9]
qtl = np.quantile(S, lvls, axis=0)
for i, lvl in enumerate(lvls):
    plt.plot(model.t, qtl[i,:], linewidth=1, label=r'$q_{' + str(lvls[i]) + '}$')
    
plt.plot(model.t, S[0,:], color='k', linewidth=1)
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$S_t$', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.ylim(5,15)
plt.show()