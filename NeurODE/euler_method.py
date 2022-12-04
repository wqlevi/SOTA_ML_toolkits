'''
The way of Euler method on calculating function solution is as follows:
    - knowing the derivative of the function(e.g. 1st derivative here)
    - divide x range to N portions
    - apply iterative increment based on previous point and slope 
'''
import numpy as np
from matplotlib.pyplot import plot, show, legend, subplots
from sys import version

def Euler_func(N,x):
    a,b = 0,5
    y_euler = np.ones(N)
    y_euler[0]=0
    h = (b-a)/N
    for i in range(N-1):
        y_euler[i+1] = y_euler[i]+h*((-1/5)*y_euler[i] + np.exp(-x[i]/5)*np.cos(x[i])) # y_t+1 = y_t + h*f(t,y), where h is step size(default = 1)
    return y_euler
def version_check():
    '''
    This function does NOT functioning in python2
    '''
    if version[0] == '2':
        raise Exception("Python 2 is not functioning here!")
if __name__ == '__main__':
    version_check()
    N = 1000
    x = np.linspace(0,5,N)[:,None]
    y = np.exp(-(x/5))*np.sin(x)

    y_Euler = Euler_func(N,x)

    fig,ax = subplots()
    ax.plot(x,y,'orange',label='Analytical solution')
    ax.plot(x,y_Euler,'g--',label='Euler method')

    legend()
    show()




