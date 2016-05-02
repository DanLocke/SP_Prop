# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import scipy.integrate
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import leastsq
import scipy as sci
from scipy import sparse
from scipy.optimize import curve_fit
import cmath
import matplotlib.animation as animation

class QC:
    def __init__(self,m,a, no_iterations):
        self.plottextsize = 24
        plt.rcParams.update({'font.size': self.plottextsize})
        self.r = 3**m
        self.c = 3**m
        self.sidelength = (self.c+1)//2
        self.x = (self.c-1)*a#float(input("Enter width (x axis) of well: "))
        self.y = (self.r-1)*a#float(input("Enter depth (y axis) of well: "))
        self.xmap=[]
        self.ymap=[]
        self.Umap=[]
        self.Q=[]
        
        self.M_centre=[]
        self.M=[]            # contains all elements of vert&hor, no repeats
        self.M_vert=[]
        self.M_hor=[]
        self.Mdiag=[]        # contains all elements of 2 diags, no repeats
        self.MdiagA=[]
        self.MdiagB=[]
        
        self.xt = np.linspace(0,self.x,self.c+2)
        self.yt = np.linspace(0,self.y,self.r+2)
        self.M_sym=[]
        self.sym_ind=[]
        self.no_points=0
        self.m = m
        self.iterations=no_iterations-m
        self.iterations_label = no_iterations
        
    def FillM(self):
        no_points=len(self.xmap)
        for I in range(0,no_points):
                if self.xmap[I]==self.sidelength: self.M_vert.append(I)
                if self.ymap[I]==self.sidelength: self.M_hor.append(I)
                if self.xmap[I]==self.sidelength or self.ymap[I]==self.sidelength: self.M.append( I )
                if self.xmap[I]==self.sidelength and self.ymap[I]==self.sidelength: self.M_centre.append( I )
                if self.xmap[I]==self.ymap[I]: self.MdiagA.append(I)
                if self.xmap[I]==self.c+1-self.ymap[I]: self.MdiagB.append(I)
                if self.xmap[I]==self.ymap[I] or self.xmap[I]==self.c+1-self.ymap[I]: self.Mdiag.append(I)

    
    def delta(self,var,a):
        if var in a: return 1
        else:        return 0
    
    def deltaQRTR(self,i,I,j,J):
        # check it works right for diags
        DQ1 = (1+self.delta(self.Q[i][I],self.M_centre))*(1+self.delta(self.Q[j][J],self.M_centre))*(1+self.delta(self.Q[i][I],self.M))*(1+self.delta(self.Q[j][J],self.M))
        DQ2 = (1+self.delta(self.Q[i][I],self.M_centre))*(1+self.delta(self.Q[j][J],self.M_centre))*(1+self.delta(self.Q[i][I],self.Mdiag))*(1+self.delta(self.Q[j][J],self.Mdiag))
        return np.power(DQ1*DQ2,1/2)
        
        
    def inside(self, i, j):
        x = j-1
        y = i-1
        iter_t = 1
        while(x>0 or y>0): 
            if(x%3==1 and y%3==1):
                if iter_t > -self.iterations: 
                    return 0
            x = x // 3 
            y = y // 3
            iter_t += 1
        return 1 

    def CreateMap(self):
        k=0
        for i in range(1,self.r+1):
            Umap_t=[]
            for j in range(1,self.r+1):
                if self.inside(i,j): 
                    self.xmap.append(j)
                    self.ymap.append(i)
                    Umap_t.append(k)
                    k+=1
                    
                else: 
                    Umap_t.append(np.nan)
            self.Umap.append(Umap_t)
        self.no_points=(len(self.xmap))

    def Symmetry(self, sym_type):
        
        sym_list = [0,1,2,3,4,5,6,7]
        
        if sym_type[0]==1: 
            for m in self.M_vert:
                self.M_sym.append(m)
            for i in [2,3,4,5]:
                if i in sym_list: sym_list.remove(i)
            
        if sym_type[1]==1: 
            for m in self.M_hor: 
                self.M_sym.append(m)
            for i in [4,5,6,7]:
                if i in sym_list: sym_list.remove(i)
            
        if sym_type[2]==1: 
            for m in self.MdiagA: 
                self.M_sym.append(m)
            for i in [1,2,3,4]:
                if i in sym_list: sym_list.remove(i)
            
        if sym_type[3]==1: 
            for m in self.MdiagB: 
                self.M_sym.append(m)
            for i in [3,4,5,6]:
                if i in sym_list: sym_list.remove(i)
                
        self.sym_ind = sym_list

    
    def CreateQs(self):
        i = self.xmap
        sidelength = (self.c+1)//2
        no_points = len(i)
        self.Q=[[] for x in range(8)]

        for i in range(0,sidelength):
            for j in range(0,sidelength):
                if j<=i:
                    if np.isnan(self.Umap[i][j]) == False:
                        self.Q[0].append(self.Umap[i][j])
                        self.Q[3].append(self.Umap[i][-(j+1)])
                        self.Q[4].append(self.Umap[-(i+1)][-(j+1)])
                        self.Q[7].append(self.Umap[-(i+1)][j])
                        
                        self.Q[1].append(self.Umap[j][i])
                        self.Q[2].append(self.Umap[j][-(i+1)])
                        self.Q[5].append(self.Umap[-(j+1)][-(i+1)])
                        self.Q[6].append(self.Umap[-(j+1)][i]) 
        

    def genMatrix(self, sparseflag):
        i = self.xmap
        sidelength = (self.c+1)//2
        no_points = len(i)
        M = []        
        for I in range(0,no_points):
            R=np.zeros(no_points)
            for J in range(0,no_points):
                R[I] = 4.0
                if (J!=I and self.adjacentpoint(I,J)): R[J] = -1
            M.append(R)
        if sparseflag == 1: M2 = sparse.csr_matrix(M)
        if sparseflag == 0: M2 = np.array(M)
        return M2
        
        
    def adjacentpoint(self, I,J):
        adjbool = 0
        if (abs(self.xmap[J]-self.xmap[I])==1) and (self.ymap[J]== self.ymap[I]): adjbool = 1-adjbool
        if (abs(self.ymap[J]-self.ymap[I])==1) and (self.xmap[J]== self.xmap[I]): adjbool = 1-adjbool
        return adjbool

    def arrayFromZ(self, v, xt, yt):
        z= np.zeros( (self.r+2, self.c+2), dtype=complex )
        for i in range(0,self.r):
            for j in range(0,self.r):
                if np.isnan(self.Umap[i][j])==False:
                    z[i+1][j+1] = v[self.Umap[i][j]]
        z = z*z.conjugate()
        #z=z.real
        #print(z)
        Ix=[]
        for i in range(0,self.r+2):
            Ix.append(scipy.integrate.simps(z[i,:], x=xt))    
        I = scipy.integrate.simps(Ix, x=yt)
        return z/I
    
    def delta3(self, redE, nt, redPolyN):
        a,b = np.polyfit(redE, nt, 1)
        varDelta3 = []
        l = len(redE)
        for c in range(4*l//10,6*l//10):#(l//10, 9*l//10):
            temp=[]
            for L in range(1,4*l//10):#l//10+1):
                r = redE[c-L:c+L]
                t=((nt[c-L:c+L] - redPolyN(redE[c-L:c+L]))**2)
                temp.append(scipy.integrate.simps(t, x=r)/L)
            varDelta3.append(temp)
        Delta3=np.sum(varDelta3, axis=0)
        for j in range(0,len(Delta3)):
            Delta3[j] = Delta3[j] / (len(varDelta3))
        return Delta3 
        
    def makeGaussian(self, fwhm=5, center=None, k=1):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        size = self.r
        #x = np.arange(0, size, 1, float)
        #y = x[:,np.newaxis]
        
        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
        
        err = fwhm/(2*np.sqrt(2*np.log(2)))
        gaussian=[]
        for y in range(1,self.r+1):
            for x in range(1,self.c+1):
                #if np.isnan(self.Umap[x][y]) == True:
                if np.isnan(self.Umap[x-1][y-1]) == False:
                    gaussian.append( np.exp(-( (x-x0)**2 + (y-y0)**2 ) / (2*err**2) ) * cmath.exp(1j*k*(y-y0)) )
        return np.array(gaussian)#+1j*np.zeros(len(gaussian))
        
    def Propagate(self, wvfn, H, dt):

        Id = np.identity(self.no_points)
        wvfn2 = np.zeros(len(wvfn))+1j*np.zeros(len(wvfn))
        for I in range(0,len(wvfn)):
            for J in range(0,len(wvfn)):
                if (Id[I][J]+1j*H[I][J]*dt*0.5)!=0:
                    wvfn2[I] += wvfn[J]*(Id[I][J]-1j*H[I][J]*dt*0.5)/(Id[I][J]+1j*H[I][J]*dt*0.5) #check indices
        #print(H)
        return wvfn2
    ################################# PLOTTING #################################
       
    def plotMap(self):
        fig = plt.figure(figsize=(15,15))
        plt.plot(self.xmap, self.ymap, 'ro')
        plt.title('Points', fontsize=self.plottextsize+2)
        plt.xlim(0,max(self.xmap)+1)
        plt.ylim(0,max(self.ymap)+1)
        plt.xlabel('$x$',fontsize=self.plottextsize+2)
        plt.ylabel('$y$',fontsize=self.plottextsize+2)
        plt.show()
        fig.savefig('Map.png')

    def plot2Dpsi(self, X,Y,z,m): 
        fig = plt.figure(figsize=(14,12))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X,Y,z, cmap=cm.coolwarm, cstride=1, rstride=1)
        fig.colorbar(surf)
        plt.xlabel('$x_i$',fontsize=self.plottextsize+2)
        plt.ylabel('$y_i$',fontsize=self.plottextsize+2)
        ax.set_zlabel('$|\psi|^2$',fontsize=self.plottextsize+2)
        plt.legend()
        plt.show()
        fig.savefig('Billiard_%i.png' %m)
        
    def plotEnergies(self,nt, E):
        fig = plt.figure(figsize=(15,10))
        plt.plot(nt, E, label='Numeric')
        plt.title('Energy vs n', fontsize=self.plottextsize+2)
        plt.xlabel('$n$',fontsize=self.plottextsize+2)
        plt.ylabel('$Energy$',fontsize=self.plottextsize+2)
        plt.legend()
        plt.show()
        fig.savefig('2D_E_vs_n.png')  
        
    def plotN(self, E,polyN, nt):
        fig = plt.figure(figsize=(15,10))
        plt.plot(polyN(E), E, label='Polynomial fit')
        plt.plot(nt, E, label='Numeric')
        plt.title('n vs Energy', fontsize=self.plottextsize+2)
        plt.xlabel('n',fontsize=self.plottextsize+2)
        plt.ylabel('Energy',fontsize=self.plottextsize+2)
        plt.legend()
        plt.show()
        fig.savefig('Billiard_n_vs_E.png')
        
    def plotPolyN(self, nt, polyN, E):
        fig = plt.figure(figsize=(15,10))
        plt.plot(nt, polyN(E), label='Polynomial fit')
        plt.plot(nt,nt, label='Perfect fit')
        plt.title('n vs Polynomial fit in n', fontsize=self.plottextsize+2)
        plt.xlabel('n',fontsize=self.plottextsize+2)
        plt.ylabel('Polynomial fit in n',fontsize=self.plottextsize+2)
        plt.legend()
        plt.show()
        fig.savefig('Billiard_n_vs_polyn.png')
        
    def plotDelta(self, nt, PolyN, E, sym_type):
        delta=[]
        ntt = []
        for i in range(0,len(nt)-1):
            delta.append(abs(PolyN(E[i+1])-PolyN(E[i])))
            ntt.append(nt[i]-nt[0])
        fig = plt.figure(figsize=(15,10))
        entries, bin_edges, patches = plt.hist(delta, bins=20,  histtype='bar', normed=True, label='Delta = n($E_{i+1}$) - n($E_{i}$)')
        x = 0.5*(bin_edges[1:] + bin_edges[:-1])
        y = entries
        # hacky way of ignoring empty bins 
        # Note: ROOT does this for least squares fits to histograms
        xy=[]
        xy.append(x)
        xy.append(y)        
        x = xy[0][xy[1]!=0]
        y = xy[1][xy[1]!=0]
        def fitfunc(x, *p):
            return p[0]*(x**p[1])*np.exp(-(x**(p[1]+1))/p[2])
        init = [1.2,0,1.2]
        sigma_y = np.sqrt(y)  # Adding Poisson errors in y
        popt, pcov  = curve_fit(fitfunc, x, y, p0=init, sigma=sigma_y, absolute_sigma=False)
        c = popt
        c_err = np.sqrt(np.diag(pcov))
        print('Fit Coefficients: {}, {}, {}'.format(c[0],c[1],c[2]))
        x_hd = np.linspace(0,bin_edges[-1],100)
        x=x_hd        
        plt.plot(x, fitfunc(x, *c), label='Fit: b = %1.3f \u00B1 %1.3f' %(c[1], c_err[1]) )
        dn=[]
        for i in range(0,len(nt)-1):
            dn.append(PolyN(E[i+1])-PolyN(E[i]))
        D = np.mean(dn)
        print('Average nearest neighbour spacing, D: %f' %D)
        
        Ps = (1/D)*np.exp(-x/D)
        plt.plot(x, Ps, label='Poisson')
        Pw = ((np.pi*x)/(2*D**2))*np.exp((-np.pi*x**2)/(4*D**2))
        plt.plot(x, Pw, label='Wigner')
        plt.title('Delta histogram, %s iterations (3^%s)' %(self.iterations_label,self.m), fontsize=self.plottextsize+2)
        plt.ylabel('Frequency',fontsize=self.plottextsize+2)
        plt.xlabel('Delta',fontsize=self.plottextsize+2)
        plt.legend()
        plt.show()
        fig.savefig('Delta_plots/%s/nbar_hist_%s_%s.png' %(sym_type, self.r, self.iterations_label))
        filename = 'Delta_plots/%s/parameters.txt' %sym_type
        par_file = open(filename, 'a')
        par_file.write('%i\t%i\t%f\t%f\t%f\n' %(self.r,self.c,c[0],c[1],c[2]))
        par_file.close()
        return c[1], c_err[1]        
        

    def masking(self, z):
        # Masking usually works with quads, this is a hacky way of preserving boundary zeros
        # although it doesn't work correctly anyway, needs to mask until half way between points. SHIT.
        for i in range(1,self.c+1):
            for j in range(1,self.c+1):
                if z[i][j]==0:
                    if z[i+1][j]==0 and z[i-1][j]==0 and z[i][j-1]==0 and z[i][j+1]==0:
                        z[i][j] = np.nan
            
        return z   
        
    def plotContour(self,X,Y,z,time):
        fig = plt.figure(figsize=(15,15))
        
        # overlay the well - bit fucked, especially for smaller numbers of points
        # for i in range(0,self.r):
        #     for j in range(0,self.r):
        #         if np.isnan(self.Umap[i][j]): z[i+1][j+1] = np.nan
        #z[ z==0 ] = np.nan              # convert zeros to Nan for masking TOO BIG
        #z = self.masking(z)             # mask internal zeros only TOO SMALL
        
        
        no_bins = 100
        lev = np.linspace(0,np.nanmax(z),no_bins)
        #plt.gca().patch.set_color('1')
        my_cmap = plt.cm.get_cmap('gist_rainbow')
        plt.contourf(X, Y, z, levels = lev , cmap=my_cmap)
        #plt.contourf(X, Y, z, cmap=my_cmap)
        #z_min, z_max = 0, np.abs(z).max()
        #plt.pcolormesh(X, Y, z, cmap=my_cmap, vmin=z_min, vmax=z_max)
        plt.title('Contour plot at time = %s' %time, fontsize=self.plottextsize+10)
        plt.xlabel('$x$',fontsize=self.plottextsize+12)
        plt.ylabel('$y$',fontsize=self.plottextsize+12)
        plt.colorbar()
        plt.show()
        fig.savefig('Contour_%i_%s_%s.png' %(time, self.r, self.iterations_label))
        
    def ContourAnim(self, X, Y, z, maxtime):
        #z[z==0.0] = np.nan
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        im=[]
        interp_meth=None

#        for vert in range(0,2):
#            for hor in range(0,2):
#                axes[vert,hor].set_title('%i iterations' % its)
#                im_t = axes[vert,hor].imshow(z[its][0], interpolation=interp_meth, extent=(np.amin(X), np.amax(X), np.amin(Y), np.amax(Y)), cmap=cm.gist_rainbow)
#                its+=1
            
        fig.set_figheight(30)
        fig.set_figwidth(35)   
        #lev = np.linspace(0,np.nanmax(z[3][0]),no_bins=500)
        #my_cmap = plt.cm.get_cmap('gist_rainbow')

        def init():  
            its=0
            for vert in range(0,2):
                for hor in range(0,2):
                    axes[vert,hor].set_title('%i iterations' % its)
                    im_t = axes[vert,hor].imshow(z[its][0], interpolation=interp_meth, extent=(np.amin(X), np.amax(X), np.amin(Y), np.amax(Y)), cmap=cm.gist_rainbow)
                    im.append(im_t)                    
                    its+=1  
            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            plt.colorbar(im[3], cax=cax)
            return 0
            
        def animate(i): 
            plt.suptitle(r't = %i ms' % i, fontsize=30)
            
            for its in range(0,4):
                im[its].set_data(z[its][i])
                
            cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            plt.colorbar(im[3], cax=cax)
            return im
            
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=maxtime)       
        
        plt.rcParams['animation.ffmpeg_path'] ='C:\\Users\\Daniel\\Downloads\\WebMConverter\\FFmpeg\\ffmpeg.exe'
        FFwriter = animation.FFMpegWriter()
        anim.save('basic_animation.mp4', writer = FFwriter, fps=10)
        plt.show()
        anim.save('animation.mp4')
            
    def plotDelta3(self, delta3):
        fig = plt.figure(figsize=(15,10))
        L = np.linspace(1, len(delta3), len(delta3))
        plt.plot(L, delta3, label='Delta-3')
        plt.title('Delta-3 vs L', fontsize=self.plottextsize+2)
        plt.xlabel('$interval, L$',fontsize=self.plottextsize+2)
        plt.ylabel('$Delta-3$',fontsize=self.plottextsize+2)
        plt.xlim(0, max(L))
        plt.legend()
        plt.show()
        fig.savefig('Delta-3.png')