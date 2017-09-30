#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Kriging 模型，用于数据拟合"""


import numpy as np
import matplotlib.pyplot as plt
import GlobalOptimizeMethod.DOE as doe
from matplotlib import cm

class Kriging(object):
    def __init__(self,dataPoints,value,min=None,max=None):
        """points是插值点，n行d列，d是点的维度，n是点的数目
        min和max是拟合空间的范围，用以对数据进行归一化。如果min和max都等于0，说明点数据已经是归一化之后的数据了"""
        if(min==None and max==None):
            self.points=dataPoints
        else:
            self.points=self.uniformization(dataPoints,min,max)
        points=self.points
        num=points.shape[0]
        d=points.shape[1]

        #设置计算相关系数中所使用的theta和p值
        self.theta=np.zeros(d)
        self.p=np.zeros(d)
        for i in range(0,d):
            self.theta[i]=10
            self.p[i]=2
        self.theta[0]=0.11
        self.theta[1]=0.5

        self.R=np.zeros((num,num))
        for i in range(0,num):
            for j in range(0,num):
                self.R[i,j]=self.correlation(points[i,:],points[j,:])
        self.R_1=np.linalg.inv(self.R)
        print(np.dot(self.R_1,self.R))
        F=np.zeros((num,1))+1
        R_1=self.R_1
        self.ys = value.reshape((num, 1))
        ys=self.ys

        R_1=np.eye(R_1.shape[0])
        beta0=np.dot(F.T,R_1)
        denominator=np.dot(beta0,F)
        numerator=np.dot(beta0,ys)
        beta0=numerator/denominator
        self.beta0=beta0

        factor=ys-beta0*F
        sigma2=np.dot(factor.T,R_1)
        sigma2=np.dot(sigma2,factor)/num
        self.sigma2=sigma2

    def getY(self,x):
        num=self.points.shape[0]
        r=np.zeros((num,1))
        for i in range(0,num):
            r[i]=self.correlation(self.points[i,:],x)
        F=np.zeros((num,1))+1
        R_1=self.R_1
        factor=self.ys-self.beta0*F
        y=np.dot(r.T,R_1)
        y=self.beta0+np.dot(y,factor)

        f1=np.dot(F.T,R_1)
        f1=(1-np.dot(f1,r))**2
        f2=np.dot(F.T,R_1)
        f2=np.dot(f2,F)
        f1=f1/f2
        f2=np.dot(r.T,R_1)
        f2=np.dot(f2,r)
        self.varience=self.sigma2*(1-f2+f1)
        return y

    def uniformization(self,points,min,max):
        """将各点缩放在【0,1】*n的空间中"""
        d=points.shape[1]
        num=points.shape[0]
        # max=np.zeros(d)
        # min=np.zeros(d)
        # for i in range(0,d):
        #     max[i]=np.max(points[:,i])
        #     min[i]=np.min(points[:,i])
        p=np.zeros(points.shape)
        for i in range(0,num):
            for j in range(0,d):
                p[i,j]=(points[i,j]-min[j])/(max[j]-min[j])
        return p

    def correlation(self,point1,point2):
        """获取相关系数，point1和point2是一维行向量"""
        d=point1.shape[0]
        R=np.zeros(d)
        theta=self.theta
        p=self.p
        for i in range(0,d):
            R[i]=-theta[i]*np.abs(point1[i]-point2[i])**p[i]
        return np.exp(np.sum(R))




if __name__=="__main__":
    def func(x1,x2):
        return 100*(x1**2-x2)**2+(1-x1)**2
    sampleNum=20
    min=np.array([-2.048,-2.048])
    max=np.array([2.048,2.048])
    lh=doe.LatinHypercube(2,sampleNum,min,max)
    value=np.zeros(sampleNum)
    for i in range(0,sampleNum):
        value[i]=func(lh.realSamples[i,0],lh.realSamples[i,1])
    kriging=Kriging(lh.samples,value)

    numy=10
    numx=10
    x=np.arange(min[0],max[0],0.2)
    y=np.arange(min[1],max[1],0.2)
    numx=x.shape[0]
    numy=y.shape[0]
    X,Y=np.meshgrid(x,y)
    value1=np.zeros((numy,numx))
    value2=np.zeros((numy,numx))
    for i in range(0,numx):
        for j in range(0,numy):
            value1[j,i]=kriging.getY(np.array([X[j,i],Y[j,i]]))
            value2[j,i]=func(X[j,i],Y[j,i])
    fig=plt.figure()
    ax1=fig.add_subplot(121)
    map1=ax1.imshow(value1,cmap=cm.get_cmap('Blues'))
    plt.colorbar(mappable=map1, cax=None, ax=None,shrink=0.5)
    ax1.set_title('estimate')

    ax2=fig.add_subplot(122)
    map2=ax2.imshow(value2,cmap=cm.get_cmap('Blues'))
    plt.colorbar(mappable=map2, cax=None, ax=None,shrink=0.5)
    ax2.set_title('real')

    # CS = plt.contour(X, Y, value1)
    # plt.clabel(CS, inline=0, fontsize=10)
    # plt.title('Simplest default with labels')
    plt.show()
