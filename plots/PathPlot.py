from audioop import avg
import matplotlib.pyplot as plt
import statistics

ax = plt.axes(projection='3d')
tag_level = 0.37
y_const = 0
ax.set_xlim(-1.5, 1.1)
ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1])
ax.set_ylim(-4, 0)
ax.set_yticks([-1, -1.5, -2, -2.5, -3, -3.5])
ax.set_zlim(-0.1, 4)
ax.set_zticks([])
mov_num = 0

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
x = 0.08
worldX = [-x, x, x, -x, -x]
worldZ = [tag_level+x, tag_level+x, tag_level-x, tag_level-x, tag_level+x]


worldY = [0, 0, 0, 0, 0]

ax.plot(worldX, worldY, worldZ, color='black', linewidth=0.1)


for i in range(0, 6):
    front_num = 1+i/2
    if ((front_num - round(front_num)) == 0):
        front_num = round(front_num)
    front_num = str(front_num)
    if i==0:
        plot_color = 'red'
    if i==1:
        plot_color = 'red'
    if i==2:
        plot_color = 'red'
    if i==3:
        plot_color = 'red'
    if i==4:
        plot_color = 'red'
    if i==5:
        plot_color = 'red'
    for j in range(1, 4):
        if front_num == '3.5':
            z_off = 0.5
        else:
            z_off = 0
        base_name = front_num + '-720-' + str(j)
        wname = 'XYZ-' + base_name + '.txt'
        print(wname)

        file = open(wname, 'r')

        count = 0
        x_values = []
        y_values = []
        z_values = []

        for line in file:
            count +=1
            rem = count % 3
            if (rem == 1):
                x_values.append(float(line))
            if (rem == 2):
                y_values.append(y_const)
            if (rem == 0):
                z_values.append(float(line)-z_off)
        file.close()
        ax.plot(x_values, z_values, y_values, color=plot_color, linewidth=0.1, alpha = 0.2)
        avgX = []
        avgY = []
        avgZ = []
        print(x_values)
        plot_count = 0
        for n in x_values:
            if plot_count > mov_num:
                totX = []
                totY = []
                totZ = []
                for i in range(plot_count-9, plot_count+1):
                    totX.append(x_values[i])
                    totY.append(y_values[i])
                    totZ.append(z_values[i])
                avgX.append(statistics.median(totX))
                avgY.append(statistics.median(totY))
                avgZ.append(statistics.median(totZ))
            plot_count+=1
        #ax.plot(avgX, avgZ, avgY, color='brown', linewidth=1)

for i in range(0, 5):
    front_num = i/2
    if ((front_num - round(front_num)) == 0):
        front_num = round(front_num)
    front_num = str(front_num)
    if i==0:
        plot_color = 'blue'
    if i==1:
        plot_color = 'blue'
    if i==2:
        plot_color = 'blue'
    if i==3:
        plot_color = 'blue'
    if i==4:
        plot_color = 'blue'
    up_bound = float(front_num) - 1.5 + 0.5
    low_bound = float(front_num) - 1.5 - 0.5
    for j in range(1, 4):
        base_name = front_num + '-720-' + str(j) + '-v'
        wname = 'XYZ-' + base_name + '.txt'
        print(wname)

        file = open(wname, 'r')

        count = 0
        x_values = []
        y_values = []
        z_values = []

        x_mov = []
        z_mov = []

        x_stat = []
        for line in file:
            count +=1
            rem = count % 3
            if (rem == 1):
                x_values.append(float(line))
            if (rem == 2):
                y_values.append(y_const)
            if (rem == 0):
                z_values.append(float(line))
        print(len(x_values))
        print(len(y_values))
        print(len(z_values))
        file.close()
        ax.plot(x_values, z_values, y_values, color=plot_color, linewidth=0.1, alpha = 0.2)
        avgX = []
        avgY = []
        avgZ = []
        print(x_values)
        plot_count = 0
        for n in x_values:
            if plot_count > mov_num:
                totX = []
                totY = []
                totZ = []
                for i in range(plot_count-9, plot_count+1):
                    totX.append(x_values[i])
                    totY.append(y_values[i])
                    totZ.append(z_values[i])
                avgX.append(statistics.median(totX))
                avgY.append(statistics.median(totY))
                avgZ.append(statistics.median(totZ))
            plot_count+=1
        #ax.plot(avgX, avgZ, avgY, color='blue', linewidth=1)
        
for i in range(1, 5):
    front_num = i
    front_num = str(front_num)
    if i==0:
        plot_color = 'blue'
    if i==1:
        plot_color = 'blue'
    if i==2:
        plot_color = 'blue'
    if i==3:
        plot_color = 'blue'
    if i==4:
        plot_color = 'blue'
    up_bound = float(front_num) - 1.5 + 0.5
    low_bound = float(front_num) - 1.5 - 0.5
    for j in range(1, 4):
        base_name = 'r' + front_num + '-720-' + str(j)
        wname = 'XYZ-' + base_name + '-txt'
        file = open(wname, 'r')

        count = 0
        x_values = []
        y_values = []
        z_values = []

        x_mov = []
        z_mov = []

        x_stat = []
        for line in file:
            count +=1
            rem = count % 3
            if (rem == 1):
                x_values.append(float(line))
            if (rem == 2):
                y_values.append(y_const)
            if (rem == 0):
                z_values.append(float(line))
        print(len(x_values))
        print(len(y_values))
        print(len(z_values))
        file.close()
        ax.plot(x_values, z_values, y_values, color='black', linewidth=0.1, alpha = 0.8)
        
plt.show()


