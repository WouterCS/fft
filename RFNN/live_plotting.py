# Import colors for plotting
import RFNN.visualization.stylesheet as stylesheet

# Example usage
# from live_plotting import update_live_plots
# from functools import partial
# show_training_function = partial(update_live_plots, fig=fig1, ax1=ax1, ax2=ax2, ax3=ax3, ax4=ax4)

# Event handler
def update_live_plots(parameters, fig, ax1, ax2, ax3, ax4):
    #print('plot start')
    # Loss plot
    # if ax1.lines:
        # print('plot1')
        # line1 = ax1.lines[0]
        # line1.set_xdata(parameters.epochs[1:])
        # line1.set_ydata(parameters.loss)

        # ax1.relim()
        # ax1.autoscale_view()
    # else:
        # ax1.plot(parameters.epochs[1:], parameters.loss, color=stylesheet.color1, linewidth=2.5, label='MNIST')
        # ax1.legend(loc='center', bbox_to_anchor=(0.5, 1), ncol=2, fancybox=True, shadow=False)

    # if ax2.lines:
        # print('plot2')
        # line1 = ax2.lines[0]
        # line1.set_xdata(parameters.epochs)
        # line1.set_ydata(parameters.sigmas[0])
        # ax2.relim()
        # ax2.autoscale_view()
    # else:
        # ax2.plot(parameters.epochs, parameters.sigmas[0], color=stylesheet.color1, linewidth=2.5, label='MNIST')
        # ax2.legend(loc='center', bbox_to_anchor=(0.5, 1), ncol=2, fancybox=True, shadow=False)

    # Test error plot
    # if ax1.lines:
        # print('plot3')
        # line1 = ax1.lines[0]
        # line1.set_xdata(parameters.acc_epochs[1:])
        # line1.set_ydata(parameters.acc_test[1:])

        # ax1.relim()
        # ax1.autoscale_view()
    # else:
        # ax1.plot(parameters.acc_epochs, parameters.acc_test, color=stylesheet.color1, linewidth=2.5, label='MNIST')
        # ax1.legend(loc='center', bbox_to_anchor=(0.5, 1), ncol=2, fancybox=True, shadow=False)

    # # Train error plot
    # if ax4.lines:
        # print('plot4')
        # line1 = ax4.lines[0]
        # line1.set_xdata(parameters.acc_epochs[1:])
        # line1.set_ydata(parameters.acc_train[1:])

        # ax4.relim()
        # ax4.autoscale_view()
    # else:
        # ax4.plot(parameters.acc_epochs, parameters.acc_train, color=stylesheet.color1, linewidth=2.5, label='MNIST')
        # ax4.legend(loc='center', bbox_to_anchor=(0.5, 1), ncol=2, fancybox=True, shadow=False)

    # Redraw the canvas
    #fig.canvas.draw()

    return