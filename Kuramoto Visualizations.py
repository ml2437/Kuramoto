#Copyright 2021 Max Lipton
#Email: ml2437@cornell.edu
#Twitter: @Maxematician

import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objects as go
import math
from ipywidgets import interactive, HBox, VBox



def randomverts(n):
    verts = np.random.uniform(-10,10,(n,3))

    for i in range(n):
        verti = verts[i]
        verts[i] = verti / np.linalg.norm(verti)
        
    return verts

def centroid(verts, n):
    Z = np.array([0,0,0])
    for i in range(n):
        Z = Z + verts[i]
    
    return Z / n

def iterate(verts,n,h):
    Z = centroid(verts,n)
    for i in range(n):
        verti = verts[i]
        xdot = Z - np.dot(Z, verti)*verti
        verts[i] = verti + h * xdot
        verts[i] = verts[i] / np.linalg.norm(verts[i])
    return verts

def iterateRK(verts, n, h):
    Z = centroid(verts,n)
    f = lambda v: Z - np.dot(Z,v)*v
    k1 = np.apply_along_axis(f,1,verts)
    Z = centroid(k1,n)
    k2 = np.apply_along_axis(f,1,verts + (h / 2) * k1)
    Z = centroid(k2,n)
    k3 = np.apply_along_axis(f,1,verts + (h / 2) * k2)
    Z = centroid(k3,n)
    k4 = np.apply_along_axis(f,1,verts + h * k3)
    return verts + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def iteratebackwards(verts, n, h):
    Z = centroid(verts,n)
    for i in range(n):
        verti = verts[i]
        xdot = Z - np.dot(Z, verti)*verti
        verts[i] = verti - h * xdot
        verts[i] = verts[i] / np.linalg.norm(verts[i])
    return verts

def iteratebackwardsRK(verts, n, h):
    Z = centroid(verts,n)
    f = lambda v: Z - np.dot(Z,v)*v
    k1 = np.apply_along_axis(f,1,verts)
    Z = centroid(k1,n)
    k2 = np.apply_along_axis(f,1,verts - (h / 2) * k1)
    Z = centroid(k2,n)
    k3 = np.apply_along_axis(f,1,verts - (h / 2) * k2)
    Z = centroid(k3,n)
    k4 = np.apply_along_axis(f,1,verts - h * k3)
    return verts - (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    

#Create a first moment Kuramoto simulation with n bodies, with time-step h, for T steps, using Euler's Method
def randominits(n,h,T):
    verts = np.zeros((n,3,T))
    verts[:,:,0] = randomverts(n)
    
    for i in range(T-1):
        newverts = iterate(verts[:,:,i],n, h)
        verts[:,:,i+1] = newverts
        
    return verts

#Create a first moment Kuramoto simulation with n bodies, with time-step h, for T steps, using Runge-Kutta
def randominitsRK(n,h,T):
    verts = np.zeros((n,3,T))
    verts[:,:,0] = randomverts(n)
    
    for i in range(T-1):
        newverts = iterateRK(verts[:,:,i],n, h)
        verts[:,:,i+1] = newverts
        
    return verts

def randominitsbackwards(n,h,T):
    verts = np.zeros((n,3,T))
    verts[:,:,0] = randomverts(n)
    
    for i in range(T-1):
        newverts = iteratebackwards(verts[:,:,i],n, h)
        verts[:,:,i+1] = newverts
        
    return verts

def majoritycluster(n, h, T):
    clustersize = math.floor(0.8 * n)
    minoritysize = n - clustersize
    verts = np.zeros((n,3,T))
    cluster = np.random.uniform(0,10, (1,3))
    cluster = cluster / np.linalg.norm(cluster)
    
    for i in range(clustersize):
        verts[i,:,0] = cluster
        
    for j in range(minoritysize):
        vert = np.random.uniform(-10,10,(1,3))
        vert = vert / np.linalg.norm(vert)
        verts[clustersize + j,:,0] = vert
        
    for k in range(T-1):
        newverts = iterate(verts[:,:,k],n, h)
        verts[:,:,k+1] = newverts
    
    return verts

def majorityclusterbackwards(n, h, T):
    clustersize = math.floor(0.8 * n)
    minoritysize = n - clustersize
    verts = np.zeros((n,3,T))
    cluster = np.random.uniform(0,10, (1,3))
    cluster = cluster / np.linalg.norm(cluster)
    
    for i in range(clustersize):
        verts[i,:,0] = cluster
        
    for j in range(minoritysize):
        vert = np.random.uniform(-10,10,(1,3))
        vert = vert / np.linalg.norm(vert)
        verts[clustersize + j,:,0] = vert
        
    for k in range(T-1):
        newverts = iteratebackwards(verts[:,:,k],n, h)
        verts[:,:,k+1] = newverts
    
    return verts

def majorityclusterbackwardsRK(n, h, T):
    clustersize = math.floor(0.8 * n)
    minoritysize = n - clustersize
    verts = np.zeros((n,3,T))
    cluster = np.random.uniform(0,10, (1,3))
    cluster = cluster / np.linalg.norm(cluster)
    
    for i in range(clustersize):
        verts[i,:,0] = cluster
        
    for j in range(minoritysize):
        vert = np.random.uniform(-10,10,(1,3))
        vert = vert / np.linalg.norm(vert)
        verts[clustersize + j,:,0] = vert
        
    for k in range(T-1):
        newverts = iteratebackwardsRK(verts[:,:,k],n, h)
        verts[:,:,k+1] = newverts
    
    return verts
    