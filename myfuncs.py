"""
useful functions

requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

def myeig(M:np.ndarray):
    """
    Output a hermitian matrix M's eigvals and corresponding eigvecs in the order of eigvals' values.
    
    Return a tuple of np.ndarray: `(eigvals, eigvecs)`.
    """
    eigs=np.linalg.eig(M)
    eigs2=[[np.real(eigs[0][i]),eigs[1][i]] for i in range(len(M))]
    eigs2.sort(key=lambda x:x[0])
    return np.array([eigs2[i][0] for i in range(len(M))]),np.array([eigs2[i][1] for i in range(len(M))])

def delta(a,b):
    """Kronecker delta"""
    return 0 if a!=b else 1

def theta(x):
    """Heaviside theta"""
    return 0 if x<0 else 1

def plotband(H,kdots:np.ndarray,labels:list[str],ndots=10,title="",legend=False,legendnames=[],hlines=False,hpos=0,savefig=False,show=True,savename=''):
    """
    A function plotting bands along the lines connecting `kdots`.

    - H is a function returning the **list** of energy eigvals of wavevector k.
    - `kdots` are coordinates of connecting points.
    - `labels` are names of `kdots`.
    - `ndots` are number of dots on one connecting line.
    """
    nk=len(kdots)
    krange=[]
    kplotrange=[]
    klabel=[0]
    kcount=0
    for i in range(nk-1):
        kstart=kdots[i]
        kend=kdots[i+1]
        krangei=[]
        for n in range(len(kstart)):
            kn=np.linspace(kstart[n],kstart[n]*1/ndots+kend[n]*(1-1/ndots),ndots)
            krangei.append(kn)
        krangei=np.transpose(np.array(krangei))
        krange.extend(krangei)

        kdistancei=np.linalg.norm(kend-kstart)
        kplotrangei=kcount+np.linspace(0,kdistancei*(1-1/ndots),ndots)
        kplotrange.extend(kplotrangei)

        klabel.append(kcount+kdistancei)
        kcount+=kdistancei     

    krange=np.array(krange)
    kplotrange=np.array(kplotrange)
    klabel[-1]=klabel[-1]-1/ndots*np.linalg.norm(kdots[-1]-kdots[-2])

    elist=[]
    for k in krange:
        elist.append(H(k))
    elist=np.transpose(np.array(elist))
    for i in range(len(elist)):
        line=elist[i]
        if legend:
            plt.plot(kplotrange,line,label=legendnames[i])
        else:
            plt.plot(kplotrange,line)

    emax,emin=max(elist.flatten()),min(elist.flatten())
    plt.xlim(kplotrange[0],kplotrange[-1])
    plt.ylim(emin-0.2*abs(emin),emax+0.2*abs(emax))
    plt.xlabel("k")
    plt.ylabel("E")
    plt.xticks(klabel,labels)
    plt.vlines(klabel,emin-0.2*abs(emin),emax+0.2*abs(emax),colors="black",linewidth=1)
    plt.title(title)
    if hlines:
        plt.hlines(hpos,kplotrange[0],kplotrange[-1],colors="black",linewidth=1)
    if legend:
        plt.legend()
    if savefig:
        plt.savefig(savename,dpi=600)
    if show:
        plt.show()