# Generated with SMOP  0.41
from libsmop import *
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m

    #-----------------------------------------------------------------------
#dyneye
#Copyright (C) Floris van Breugel, 2013.
#  
#florisvb@gmail.com
    
    #This function was originally written by Nathan Powell
    
    #Released under the GNU GPL license, Version 3
    
    #This file is part of dyneye.
    
    #dyneye is free software: you can redistribute it and/or modify it
#under the terms of the GNU General Public License as published
#by the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#    
#dyneye is distributed in the hope that it will be useful, but WITHOUT
#ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#License for more details.
    
    #You should have received a copy of the GNU General Public
#License along with dyneye.  If not, see <http://www.gnu.org/licenses/>.
    
    #------------------------------------------------------------------------
    
    
@function
def ukf_sqrt(y=None,x0=None,f=None,h=None,Q=None,R=None,u=None,*args,**kwargs):
    varargin = ukf_sqrt.varargin
    nargin = ukf_sqrt.nargin

    N=length(y)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:30
    
    if size(Q,3) == 1:
        Q=repmat(Q,concat([1,1,N]))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:33
    
    if size(R,3) == 1:
        R=repmat(R,concat([1,1,N]))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:37
    
    nx=length(x0)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:40
    ny=size(y,1)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:41
    nq=size(Q,1)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:42
    nr=size(R,1)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:43
    a=0.01
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:45
    
    b=2
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:46
    
    L=nx + nq + nr
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:47
    
    l=dot(a ** 2,L) - L
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:48
    
    g=sqrt(L + l)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:49
    
    Wm=concat([l / (L + l),dot(1 / (dot(2,(L + l))),ones(1,dot(2,L)))])
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:51
    
    Wc=concat([(l / (L + l) + (1 - a ** 2 + b)),dot(1 / (dot(2,(L + l))),ones(1,dot(2,L)))])
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:52
    
    if Wc(1) > 0:
        sgnW0='+'
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:55
    else:
        sgnW0='-'
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:57
    
    ix=arange(1,nx)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:60
    iy=arange(1,ny)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:61
    iq=arange(nx + 1,(nx + nq))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:62
    ir=arange((nx + nq + 1),(nx + nq + nr))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:63
    # Construct initial augmented covariance estimate
    Sa=zeros(L,L)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:66
    Sa[iq,iq]=chol(Q(arange(),arange(),1))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:67
    Sa[ir,ir]=chol(R(arange(),arange(),1))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:68
    # Pre-allocate
    Y=zeros(ny,dot(2,L) + 1)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:71
    
    x=zeros(nx,N)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:72
    
    P=zeros(nx,nx,N)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:73
    
    ex=zeros(nx,dot(2,L) + 1)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:74
    ey=zeros(ny,dot(2,L) + 1)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:75
    x[arange(),1]=x0
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:77
    P[arange(),arange(),1]=eye(nx)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:78
    S=chol(P(arange(),arange(),1))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:79
    for i in arange(2,N).reshape(-1):
        # Generate sigma points
        Sa[ix,ix]=S
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:83
        Sa[iq,iq]=chol(Q(arange(),arange(),i))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:86
        Sa[ir,ir]=chol(R(arange(),arange(),i))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:87
        xa=concat([[x(arange(),i - 1)],[zeros(nq,1)],[zeros(nr,1)]])
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:89
        X=concat([xa(concat([dot(g,Sa.T),dot(- g,Sa.T)]) + dot(xa,ones(1,dot(2,L))))])
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:90
        for j in arange(1,(dot(2,L) + 1)).reshape(-1):
            X[ix,j]=f(X(ix,j),u(arange(),i - 1),X(iq,j))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:94
            Y[arange(),j]=h(X(ix,j),u(arange(),i - 1),X(ir,j))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:95
        # Average propagated sigma points
        x[arange(),i]=dot(X(ix,arange()),Wm.T)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:99
        yf=dot(Y,Wm.T)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:100
        Pxy=zeros(nx,ny)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:103
        for j in arange(1,(dot(2,L) + 1)).reshape(-1):
            ex[arange(),j]=dot(sqrt(abs(Wc(j))),(X(ix,j) - x(arange(),i)))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:105
            ey[arange(),j]=dot(sqrt(abs(Wc(j))),(Y(arange(),j) - yf))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:106
            Pxy=Pxy + dot(dot(Wc(j),(X(ix,j) - x(arange(),i))),(Y(arange(),j) - yf).T)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:107
        __,QR=qr(ex(arange(),arange(2,end())).T,nargout=2)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:110
        fprintf('----------------')
        ex(arange(),1)
        fprintf('----------------')
        S=cholupdate(QR(ix,ix),ex(arange(),1),sgnW0)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:114
        if any(isnan(y(i))):
            continue
        __,QR=qr(ey(arange(),arange(2,end())).T,nargout=2)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:121
        Syy=cholupdate(QR(iy,iy),ey(arange(),1),sgnW0)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:122
        K=Pxy / (dot(Syy.T,Syy))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:125
        x[arange(),i]=x(arange(),i) + dot(K,(y(i) - h(x(arange(),i),u(arange(),i),zeros(nr,1))))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:126
        U=dot(K,Syy.T)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:127
        for j in arange(1,ny).reshape(-1):
            S=cholupdate(S,U(arange(),j),'-')
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:129
        P[arange(),arange(),i]=dot(S.T,S)
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:132
    
    s=zeros(nx,length(y))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:135
    for i in arange(1,nx).reshape(-1):
        s[i,arange()]=sqrt(squeeze(P(i,i,arange())))
# /home/caveman/Sync/LAB_Private/RESEARCH/code/PyNumDiff/pynumdiff/model_based/ukf_sqrt.m:137
    
    return x,P,s
    return x,P,s
    
if __name__ == '__main__':
    pass
    
