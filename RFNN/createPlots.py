import RFNN.parameters as para
import matplotlib.pyplot as plt


def createPlot(folders, targetName, labels = map(lambda x: 'label %d' % x, range(10)), title = '', xlabel = '', ylabel = ''):
    colors = ['red', 'green', 'blue', 'black', 'yellow', 'purple', 'orange', 'grey', 'pink']
    plt.clf()
    for i in range(len(folders)):
        lpara = para.parameters('/home/wouter/Dropbox/thesis/results/%s/para' % folders[i]); plt.plot(lpara.acc_train, lpara.acc_test, color = colors[i], label = labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(prop={'size':10})
    plt.xscale('log')
    plt.yscale('log')
    plt.figure(num =1, figsize = (20,20), dpi = 800)
    plt.savefig('/home/wouter/Dropbox/thesis/%s.jpg' % targetName)


# lpara = para.parameters('/home/wouter/Documents/DockerMap/results/results-300/para'); plt.plot(lpara.acc_test, color = (0,0,0.5), label = '600 trainingcases - fftAbs - ADAM - exponential lr')
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results-300-1000-abs-300epochs/results-300-1/para'); plt.plot(lpara.acc_test, color = (0,0.5,0), label = '300 trainingcases - adadelta')
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results/results-300-1/para'); plt.plot(lpara.acc_test, color = (0.5,0,0), label = '300 trainingcases - ADAM -lr 10^-4')
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results-300-1000-abs-adam/results-300-1/para'); plt.plot(lpara.acc_test, color = (0.5,0.5,0.5), label = '300 trainingcases - ADAM - lr 10^0')
 
 
 