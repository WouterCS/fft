import RFNN.parameters as para
import matplotlib.pyplot as plt

def createPlot(folders, targetName, labels = map(lambda x: 'label %d' % x, range(10)), title = '', xlabel = '', ylabel = ''):
    colors = ['red', 'green', 'blue', 'black', 'yellow', 'purple', 'orange', 'grey', 'pink']
    plt.clf()
    for i in range(len(folders)):
        lpara = para.parameters('/home/wouter/Dropbox/thesis/results/%s/para' % folders[i]); plt.plot(lpara.acc_test, color = colors[i], label = labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(prop={'size':10})
    #plt.xscale('log')
    #plt.yscale('log')
    plt.figure(num =1, figsize = (20,20), dpi = 800)
    plt.savefig('/home/wouter/Dropbox/thesis/%s.jpg' % targetName)

    
    
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results/results-300/para'); plt.plot(lpara.acc_test, color = (0,0,0.5), label = '600 trainingcases - fftAbs - ADAM - exponential lr')
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results-300-1000-abs-300epochs/results-300-1/para'); plt.plot(lpara.acc_test, color = (0,0.5,0), label = '300 trainingcases - adadelta')
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results/results-300-1/para'); plt.plot(lpara.acc_test, color = (0.5,0,0), label = '300 trainingcases - ADAM -lr 10^-4')
# lpara = para.parameters('/home/wouter/Documents/DockerMap/results-300-1000-abs-adam/results-300-1/para'); plt.plot(lpara.acc_test, color = (0.5,0.5,0.5), label = '300 trainingcases - ADAM - lr 10^0')
 
 #from RFNN.createPlots import createPlot;reload(RFNN.createPlots);from RFNN.createPlots import createPlot
 #
#createPlot(map(lambda x: 'results-%d00-1' % x, [50,100,200,600]), 'test',map(lambda x: 'Number of examples: %d00' % x, [3,10,20,50,100,200,600]), 'Adadelta 3 to 3e-2', xlabel = 'Epochs', ylabel = 'test error-rate')

i = 5/0


plt.clf()
lpara = para.parameters('/home/wouter/Dropbox/thesis/sortedResults/absFFT/model-40-to-5/results-2000-1/para')
plt.plot(lpara.acc_test, color = 'blue', label = 'with dropout')
lpara = para.parameters('/home/wouter/Dropbox/thesis/results/results-2000-1/para')
lpara.acc_test  = lpara.acc_test + ([lpara.acc_test[-1]] * ( 600 - len(lpara.acc_test) ))
plt.plot(lpara.acc_test, color = 'red', label = 'without dropout')
#plt.xticks(range(7), ['60,000', '20,000', '10,000', '5,000', '2,000', '1,000', '300'])
plt.legend(prop={'size':15}, loc = 'upper right')
plt.figure(num =1, figsize = (20,20), dpi = 800)
plt.xlabel('epochs')
plt.ylabel('test error-rate (%)')
plt.ylim(0,30)
plt.title('The difference in using or not using dropout')
plt.savefig('/home/wouter/Dropbox/thesis/%s.jpg' % 'dropoutComparison')
