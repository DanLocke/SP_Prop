# -*- coding: utf-8 -*-
"""
2D Sierpinski Carpet Propagation of Gaussian Wave Packet
@author: Dan Goldsmith & Dan Locke
"""
import numpy as np
np.set_printoptions(threshold=np.nan)
import SPfn as SPfn
m=3
a=1
z_all=[]

for no_iterations in range(0,4):
    print("Propagating Gaussian for %i iterations..." %no_iterations)
    UseSparseMatrices = False
    qcf = SPfn.QC(m,a,no_iterations)      # create object
    qcf.CreateMap()
    H = qcf.genMatrix(sparseflag=UseSparseMatrices)
    X, Y = np.meshgrid(qcf.xt, qcf.yt)
    
    #--------------------------------------#
    # PARAMETERS FOR GAUSSIAN AND PLOTTING #
    #======================================#
    no_timesteps=300
    plotevery=10
    p=10                    # ~0-20
    dt=0.002                 # ~0.001-0.05
    fwhm=4
    center=[(3**m+1)//2,0]
    #--------------------------------------#    
    
    wvfn = qcf.makeGaussian(p, fwhm, center)
    zprop=[]
    for s in range(0,no_timesteps):
        if s > 0: wvfn = qcf.Propagate(wvfn, H, dt)
        if s%plotevery==0:
            print(s)
            z = qcf.arrayFromZ(wvfn, qcf.xt, qcf.yt, s)
            zprop.append(z.real)
    z_all.append(zprop)

qcf.ContourAnim(X, Y, z_all, maxtime=no_timesteps//plotevery, timestep=dt*plotevery)