# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:06:18 2022

@author: THY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


xr_df = pd.DataFrame({'Dataset': ['CIFAR10-0.01', 'CIFAR10-0.1', 'CIFAR10-0.5', 'CIFAR10-1', 'CIFAR100-5',
                                 'CIFAR100-10', 'CIFAR100-30', 'FEMNIST', 'Shakespeare'],
                      'sr': [0.8806, 0.8154, 0.7417, 0.7092, 0.6794, 0.6618, 0.6052, 0.8256, 0.4993],
                      'pr': [0.8234, 0.7898, 0.7225, 0.7043, 0.6578, 0.6448, 0.5846, 0.4793, 0.501]
                      })

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]


plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(1,1, figsize=(22, 10), sharey=True, dpi=100)

axes.set_xticks([])
axes.spines['top'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.spines['left'].set_visible(False)
axes.spines['right'].set_visible(False)

axes.bar(x='Dataset', height="sr", data = xr_df, color = 'gray', width=0.6, alpha=0.3, label='mask shared parameters')
ax_twin = axes.twinx()
ax_twin.bar(x='Dataset', height="pr", data = xr_df, color = 'orange', width=0.3, label='mask personalized parameters')
axes.set_ylim(0, np.max(xr_df['sr'] + 0.2))
ax_twin.set_ylim(0, np.max(xr_df['sr'] + 0.2))
axes.set_ylabel('Accuracy (Acc)', color ='black')
axes.set_xlabel('Dataset', color ='black')
ax_twin.set_yticks([])
axes.legend(loc='upper left')
ax_twin.legend(loc='upper right')

plt.show()



