import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
def make_saddle() -> np.ndarray:
    x,y = np.meshgrid(np.linspace(-3,3),np.linspace(-8,8))
    z = (1+x**2)*(100-y**2)
    return x,y,z

def make_plot(coor):
    fig,ax = plt.subplots(subplot_kw={'projection':"3d"})
    x,y,z = coor

    surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm,antialiased=False)
    fig.colorbar(surf)
    ax.set_zlim(0,1000)

    plt.savefig("saddle_plot.png")

def main():
    z = make_saddle()
    make_plot(z)
    return 0

if __name__ == '__main__':
    main()

