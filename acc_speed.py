import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('speed-eao2018.pdf');
plt.rc('font',family='Times New Roman')

fig, ax = plt.subplots()  # type:figure.Figure, axes.Axes
ax.set_title('The Performance $vs.$ Speed on Sonar-Dataset', fontsize=15)
ax.set_xlabel('Tracking Speed (FPS)', fontsize=15)
ax.set_ylabel('EAO', fontsize=15)


trackers = ['SiamRPN', 'SiamRPNpp', 'SiamCAR', 'SiamMask', 'Ours']
speed = np.array([23, 50, 77, 63, 57])
speed_norm = np.array([23, 50, 77, 63, 57]) / 48
performance = np.array([0.526, 0.582, 0.547, 0.723, 0.752])

circle_color = ['cornflowerblue', 'deepskyblue',  'turquoise', 'gold', 'r']
# Marker size in units of points^2
volume = (300 * speed_norm/5 * performance/0.6)  ** 2

ax.scatter(speed, performance, c=circle_color, s=volume, alpha=0.4)
ax.scatter(speed, performance, c=circle_color, s=20, marker='o')
# text
ax.text(speed[0] - 2.37, performance[0] - 0.031, trackers[0], fontsize=10, color='k')
ax.text(speed[1] - 1.00, performance[1] - 0.005, trackers[1], fontsize=10, color='k')
ax.text(speed[2] - 3.5, performance[2] - 0.05, trackers[2], fontsize=10, color='k')
ax.text(speed[3] - 2.4, performance[3] - 0.032, trackers[3], fontsize=10, color='k')
ax.text(speed[4] - 2.9, performance[4] - 0.040, trackers[4], fontsize=10, color='k')


ax.grid(which='major', axis='both', linestyle='-.') # color='r', linestyle='-', linewidth=2
ax.set_xlim(10, 90)
ax.set_ylim(0.4, 0.85)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)



fig.savefig('speed-eao2018.png')


pdf.savefig()
pdf.close()
plt.show()
