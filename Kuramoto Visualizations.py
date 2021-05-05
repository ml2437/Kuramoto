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
    
    
def visualize(verts):
    n, _, T = np.shape(verts)

    py.init_notebook_mode()

    xs = verts[:,0,0]
    ys = verts[:,1,0]
    zs = verts[:,2,0]


    f = go.FigureWidget(
        data=[
            go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(
            size=4,
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        ))]
    )


    f.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-100,100],),
                         yaxis = dict(nticks=4, range=[-50,100],),
                         zaxis = dict(nticks=4, range=[-100,100],),),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))

    def update_z(frequency):
        f.data[0].x = verts[:,0,frequency]
        f.data[0].y = verts[:,1,frequency]
        f.data[0].z = verts[:,2,frequency]
        f.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-2,2],),
                         yaxis = dict(nticks=4, range=[-2,2],),
                         zaxis = dict(nticks=4, range=[-2,2],),),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))
        f.update_layout(scene_aspectmode='cube')


    freq_slider = interactive(update_z, frequency=(0, T-1, 1))
    vb = VBox((f, freq_slider))
    vb.layout.align_items = 'center'
    vb
    return vb
