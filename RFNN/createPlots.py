import RFNN.parameters as para
import matplotlib.pyplot as plt


def createPlot(folders, targetName, labels = map(lambda x: 'label %d' % x, range(10)), title = ''):
    colors = ['red', 'green', 'blue', 'black', 'yellow', 'purple']
    plt.clf()
    for i in range(len(folders)):
        lpara = para.parameters('/home/wouter/Documents/DockerMap%s/para' % folders[i]); plt.plot(lpara.acc_test, color = colors[i], label = labels[i])
    plt.xlabel('epochs')
    plt.ylabel('error-rate')
    plt.title(title)
    plt.legend()
    plt.figure(num =1, figsize = (20,20), dpi = 800)
    plt.savefig('/home/wouter/Documents/DockerMap/%s.jpg' % targetName)


# lpara = para.parameters('/home/wouter/Documents/DockerMap/results/results-300/para'); plt.plot(lpara.acc_test, color = (0,0,0.5), label = '600 trainingcases - fftAbs - ADAM - exponential lr')
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results-300-1000-abs-300epochs/results-300-1/para'); plt.plot(lpara.acc_test, color = (0,0.5,0), label = '300 trainingcases - adadelta')
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results/results-300-1/para'); plt.plot(lpara.acc_test, color = (0.5,0,0), label = '300 trainingcases - ADAM -lr 10^-4')
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results-300-1000-abs-adam/results-300-1/para'); plt.plot(lpara.acc_test, color = (0.5,0.5,0.5), label = '300 trainingcases - ADAM - lr 10^0')
 
 
 