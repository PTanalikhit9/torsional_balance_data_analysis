#by thanabodi 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['savefig.dpi'] = 1200

#Read excel file
file_name = r"file directory"
sheet_name = ['air', 'water', 'sugar', 'salt'] 

fig, ax_disp = plt.subplots()
#fig.text(0.5, 0.975, 'With mathtext', horizontalalignment='center', verticalalignment='top')
label = ['Air', 'Water', '30% Sugar-water', '30% Salt-water']
#color=["#B7E6A5", "#46AEA0", "#00718B", "#003147",]# "#D55E00", "#CC79A7", "#F0E442"]
color=['#F28522', '#009ADE', '#AF58BA', '#FFC61E']
linestyle=["-", "--", "-.", ":", "-", "--", "-."]
marker=['h', 'o', '^', 's']

for i in range(len(sheet_name)):
    raw_data = pd.read_excel(file_name, sheet_name[i])
    times, times_err = np.array(raw_data["times"]), np.array(raw_data["times_err"])
    position, position_err = np.array(raw_data["x_avg"]), np.array(raw_data["x_err"])
    disp, disp_err = np.array(raw_data["disp_avg"]), np.array(raw_data["disp_err"])

    ax_disp.scatter(times, disp, marker=marker[i], label=label[i], edgecolors='k', zorder=1, color=color[i])
    ax_disp.errorbar(times, disp, xerr=times_err, yerr=disp_err, linestyle='', capsize=3, zorder=0, color=color[i])

ax_disp.set_title('Displacement of beam spot in various medium')
ax_disp.set_xlabel('Times (s)')
ax_disp.set_ylabel('Displacement (m)')
ax_disp.legend()
ax_disp.ticklabel_format(axis='y', style='sci', scilimits=(-2,2), useMathText=True)

plt.show()
