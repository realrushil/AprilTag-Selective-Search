import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6.4,3.8) 
import statistics
import numpy as np

x_values = [-1, -0.5, 0, 0.5, 1]
n = 255

plt.xticks(np.arange(-1, 1.01, 0.5))
plt.xlim([-1.5, 2])
plt.ylim([0.2, 1])
plt.yticks(np.arange(0.2, 1.01, 0.1))
plt.xlabel('Lateral Distance (m)')
plt.ylabel('Accuracy')


for i in range(1, 6):
    y_values = []
    y_trial = []
    y_error = []
    count = 0
    k = i
    rname = str(k) + '_ACC-V.txt'
    lbl = 'n=' + str(k)
    file = open(rname, 'r')
    if i==1:
        plot_color = (193/n,39/n,45/n)
        point_type = '^'
    if i==2:
        plot_color = (0/n,0/n,167/n)
        point_type = 's'
    if i==3:
        plot_color = (238/n,204/n,22/n)
        point_type = 'o'
    if i==4:
        plot_color = (0/n,129/n,118/n)
        point_type = 'D'
    if i==5:
        plot_color = (179/n,179/n,179/n)
        point_type = '*'
    for line in file:
        count += 1
        y_trial.append(float(line))
        if (count % 3 == 0):
            trial_avg = statistics.mean(y_trial)
            trial_stdv = statistics.stdev(y_trial)
            y_values.append(trial_avg)
            y_error.append(trial_stdv)
            y_trial = []
    plt.errorbar(x_values, y_values, color=plot_color, yerr=y_error, fmt='o', markersize=8, capsize=5, label=lbl, marker=point_type)
    file.close()
plt.legend(loc="upper right", prop={'size':10})
plt.title('n Plot - Vertical Paths')
plt.show()










