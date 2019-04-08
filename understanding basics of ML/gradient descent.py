# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:30:55 2018

@author: Pankaj Mishra
"""

import numpy as np
import matplotlib.pyplot as plt


""" Code for Gradient Decent"""

X = [0.5,2.5]
Y = [0.2,0.9]

def f(w,b,x): #Sigmoid function
    return 1.0/(1.0+np.exp(-(w*x+b)))

def error (w,b):
    err=0.0
    for x ,y in zip(X,Y):
        fX = f(w,b,x)
        err += 0.5*(fX - y)**2
    return err

def grad_b(w,b,x,y):
    fX = f(w,b,x)
    return (fX-y)*fX*(1-fX)

def grad_w (w,b,x,y):
    #print(w)
    #print(b)
    #print(x)
    #print(y)
    fx = f(w,b,x)
    #print(fx)
    return (fx - y)*fx * (1-fx)*x

def do_gradient_descent():
    w, b, eta, max_epochs = -2,-2,1.0,100
    err = []
    for i in range(max_epochs):
        dw, db = 0,0
        for x, y in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
        
        w = w - eta*dw
        b = b - eta*db
        err.append(error(w,b))
        
    return err

err_out = do_gradient_descent()
ep = range(100)
plt.plot(ep, err_out)
plt.show()