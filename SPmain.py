# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:35:32 2015

2D Sierpinski Carpet
@author: Dan Goldsmith & Dan Locke
"""
import numpy as np
np.set_printoptions(threshold=np.nan)
from numpy import linalg
from scipy import sparse
import SPfn as SPfn

# c and r must be 3^m
z_all=[]
for no_iterations in range(0,4):
    m=3
    a=1
    mass_electron = 9.10938*10**31
    hbar = 1.0545718*10**(-34)
    scale_fact = hbar**2/(2*mass_electron*a**2)
    #no_iterations = 1                  # from 0->4
    UseSparseMatrices = False
    qcf = SPfn.QC(m,a,no_iterations)      # create object
    qcf.CreateMap()
    H = qcf.genMatrix(sparseflag=UseSparseMatrices)
    
    
    if UseSparseMatrices == True:
        no_points = len(qcf.xmap)
        k = 5#int(3*no_points/8)
        w, v = sparse.linalg.eigsh(H, k, which = 'SM')
    
    if UseSparseMatrices == False:
        w, v = linalg.eigh(H)
    
        n=len(w)
        nt = np.linspace(1,n,n)
        E = sorted(w/2)
        #qcf.plotEnergies(nt, E)
        polyN = np.poly1d(np.polyfit(E, nt, 5))
        #qcf.plotN(E, polyN,nt)
        
        
        # Limit plot to linear + interesting region
        sym_type='oo'
        n = len(w)
        top = int(3*n/8)
        bottom = int(n/4)
        print('Total number of points in stadium: {}'.format(n))
        print('Points taken in the range {} : {}'.format(bottom, top))
        nt = np.linspace(bottom,top,top-bottom+1)
        redE = E[bottom:(top+1)]
        redPolyN = np.poly1d(np.polyfit(redE, nt, 6))
        #qcf.plotN(redE, redPolyN,nt)
        #qcf.plotPolyN(nt, redPolyN, redE)
        #qcf.plotDelta(nt, redPolyN, redE, sym_type)
        
        # # Delta-3 Statistic
        # delta3 = qcf.delta3(redE, nt, redPolyN)
        # qcf.plotDelta3(delta3)
        # comp_time = (time.clock()-t0)/60.0
        # print("TIME TO COMPUTE: %f mins" %comp_time)
        
    X, Y = np.meshgrid(qcf.xt, qcf.yt)
    no_timesteps=60
    plotevery=5
    wvfn = qcf.makeGaussian(fwhm=4, center=[(3**m+1)//2,0], k=100)
    dt=1     #actually dt/hbar
    hbar = 1#1.0545718*10**(-34)
    zprop=[]
    for s in range(0,no_timesteps):
        if s > 0: wvfn = qcf.Propagate(wvfn, H, dt/hbar)
        if s%plotevery==0:
            print(s)
            z = qcf.arrayFromZ(wvfn, qcf.xt, qcf.yt)
            z[z==0.0] = np.nan
            #print(z)
            #qcf.plotContour(X,Y,z,s*dt)
            zprop.append(z.real)
    z_all.append(zprop)

qcf.ContourAnim(X, Y, z_all, maxtime=no_timesteps//plotevery)