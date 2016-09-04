from matplotlib.pyplot import plot, show

from sin_ergy.data.utils import load_training

plot_X, plot_Y = load_training(sequence=True,window_size=5)
for x in range(0,plot_X.shape[0]):
    plot(range(0,plot_X.shape[1]),plot_Y[x,:,0])
show()