import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6.4,3.8) 
import numpy as np
import statistics

x_values = [1, 1.5, 2, 2.5, 3, 3.5]
indx = len(x_values)
plt.ylim([0, 80])
plt.yticks(np.arange(0, 80, 10))
plt.xlabel('Frontal Distance (m)')
plt.ylabel('Frames Per Second')
bar_width = 0.35

bar1 = np.arange(len(x_values))
bar2 = [i+bar_width for i in bar1]

plt.xticks(bar1+bar_width/2, x_values)

n=255

rname = 'Base-H.txt'

file = open(rname, 'r')
y_values = []
y_trial = []
y_error = []
count = 0
for line in file:
    count += 1
    y_trial.append(float(line))
    if (count % 3 == 0):
        trial_avg = statistics.mean(y_trial)
        trial_stdv = statistics.stdev(y_trial)
        y_values.append(trial_avg)
        y_error.append(trial_stdv)
        y_trial = []
print(y_values)
plt.bar(bar1, y_values, bar_width, label='Full Search Method', color = (0/n,0/n,167/n))
plt.errorbar(bar1, y_values, yerr=y_error, fmt="o", color=(0*0.5/n, 0*0.5/n, 167*0.5/n), markersize=5, capsize=5)
file.close()

rname = 'Speed-H.txt'
file = open(rname, 'r')
y_values = []
y_values = []
y_trial = []
y_error = []
count = 0
for line in file:
    count += 1
    y_trial.append(float(line))
    if (count % 3 == 0):
        trial_avg = statistics.mean(y_trial)
        trial_stdv = statistics.stdev(y_trial)
        y_values.append(trial_avg)
        y_error.append(trial_stdv)
        y_trial = []
print(y_values)

plt.bar(bar2, y_values, bar_width, label='Proposed Method', color=(193/n,39/n,45/n))
plt.errorbar(bar2, y_values, yerr=y_error, fmt='o', color=(193*0.5/n, 39*0.5/n, 45*0.5/n), markersize=5, capsize=5)
plt.legend(loc='upper right')
file.close()
plt.title('Speed Comparison - Horizontal Paths')
plt.show()

