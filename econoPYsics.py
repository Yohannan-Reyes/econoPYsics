# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 00:27:49 2017

@author: Yohan Reyes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import kurtosis
import math
import os
import csv
import scipy as sp
from scipy.stats import norm
import entroPY
import quant
from tqdm import tqdm



def cmax(Asset,n_days):
    
    Asset = pd.DataFrame(Asset);
    m = Asset.shape[1]
    cols=list(Asset.columns)
    c_max = pd.DataFrame(0, index=np.arange(Asset.shape[0]), columns=cols)
    #     c_max = pd.DataFrame(0, index=np.arange(1), columns=list(Asset))
    #    c_max = pd.DataFrame(0, index=np.arange(len(Asset)-n_days), columns=list(Asset))
    #     c_max = pd.DataFrame(0, index=np.arange(Asset.shape[0]-n_days+1), columns=list(Asset));
#     Asset = pd.DataFrame.as_matrix(Asset)
    i2 = 0;
#     temp = [];
    
    for i in range(n_days,len(Asset)+1):
        c_max.iloc[i-1,:] = Asset.iloc[i2:i,:].max(axis=0)
        i2 = i2+1
        
        '''
    for i in range(n_days,len(Asset)+1):
        if i == n_days:
            c_max.iloc[0,:] = Asset[i2:i,:].max(axis=0)
            i2 = i2+1
        else:
            temp = Asset[i2:i,:].max(axis=0)    
            temp = temp.reshape((1,m))
            temp = pd.DataFrame(temp)
            temp.columns = cols
            c_max = c_max.append(temp) 
            i2 = i2+1
            '''
    return c_max


    
def cmin(Asset,n_days):
    
    Asset = pd.DataFrame(Asset);
    m = Asset.shape[1]
    cols=list(Asset.columns)
    c_min = pd.DataFrame(0, index=np.arange(Asset.shape[0]), columns=cols)
    #     c_min = pd.DataFrame(0, index=np.arange(1), columns=list(Asset))
    #     c_min = pd.DataFrame(0, index=np.arange(len(Asset)-n_days), columns=list(Asset))
    #     c_min = pd.DataFrame(0, index=np.arange(Asset.shape[0]-n_days), columns=list(Asset));
#     Asset = pd.DataFrame.as_matrix(Asset);
    i2 = 0;
#     temp = [];

    for i in range(n_days,len(Asset)+1):
        c_min.iloc[i-1,:] = Asset.iloc[i2:i,:].min(axis=0)
        i2 = i2+1
    '''     
    for i in range(n_days,len(Asset)+1):    
        if i == n_days:
            c_min.iloc[0,:] = Asset[i2:i,:].min(axis=0)
            i2 = i2+1
        else:
            temp = Asset[i2:i,:].min(axis=0)    
            temp = temp.reshape((1,m))
            temp = pd.DataFrame(temp)
            temp.columns = cols
            c_min = c_min.append(temp) 
            i2 = i2+1
    '''
    return c_min


def dec_i(Asset,n_days):

    Asset = pd.DataFrame(Asset);
    Asset = (Asset.abs())
    

    N_x = []
    N_s = []
    N_s_k = []
    N_s_k1 = []    
    K_x = []
    K_s = []
    K_s1 = []
    K_i = []
    x_min = []
    x_min_k = [];
    x_min_k1 = [];
    x_i1 = []
    x_i = []
    x_i_k = []
    i3=0
    alphas1 = []
    sigmas1 = []
    alphas0 = []
    sigmas0 = []
    for i in Asset.columns:

        d_i = []
        d_i = Asset.loc[Asset[i] == 0][i]
        d_max = pd.DataFrame(np.zeros((d_i.shape[0], 1)))

        temp = []
        temp = Asset[i].copy()

        for i1 in range(len(d_i.index)-1):
            m = d_i.index[i1]-n_days+1
            n = d_i.index[i1+1]-n_days+1
            d_max.iloc[i1] = max(temp[m:n])
        

        d_max.columns = [i]
        d_max = d_max.abs()
        d_max = d_max.sort_values(by=i,axis=0,ascending=False)
        d_max = d_max.reset_index(drop=True)
        d_max = d_max.loc[d_max[i] != 0]

        N_x.append(len(d_max))
        
        K_x.append(kurtosis(d_max, fisher=True))
        
        K_s.append(quant.accumulated_kurt(d_max))
        
        plt.figure()
        values, base = np.histogram(d_max, bins=40)
        # values, base = np.histogram(np.log(d_max), bins=40)
        cumulative = np.cumsum(values)
        plt.plot(base[:-1], cumulative, c='blue')
        
        d_max = d_max.sort_values(by=[i],axis=0,ascending=True)
        d_max = d_max.reset_index(drop=True)
        
        plt.figure()
        K_i.append(quant.accumulated_kurt(d_max))
        plt.plot(K_i[i3])
        
        temp0 = quant.accumulated_kurt(d_max)
        i2 = temp0.columns[0]
        
        i4 = len(temp0)-2
        temp2 = temp0.iloc[len(temp0)-1,:][0]
        temp3 = min(temp0.iloc[i4:len(temp0),0])
        while (temp3<temp2) & (temp3==abs(temp3)):
            temp2 = temp3.copy()
            i4 = i4-1
            temp3 = min(temp0.iloc[i4:len(temp0),0])
        i4 = i4+1
        x_min_k.append(d_max.iloc[i4,:])
        N_s_k.append(len(d_max.iloc[i4:,:]))
        x_i_k.append(d_max.iloc[i4:,:].copy())
        
        
        '''
#        for i4 in range(len(temp0)-2,-1,-1):
            temp3 =  min(temp0.iloc[i4:len(temp0),0])
            if temp3>=temp2:
                x_min_k.append(d_max.iloc[i4,0])
            temp2 = temp3.copy()
        '''
        if (len(temp0.loc[temp0[i2] > 0])!=0):
            temp1 = temp0.loc[temp0[i2] > 0].index[0].copy()
            
            x_min.append(d_max.iloc[temp1,0])
            x_min_temp = d_max.iloc[temp1,0]
            K_s1.append(kurtosis(d_max.iloc[temp1:,:], fisher=True))
            x_i.append(d_max)
            x_i1_temp = d_max.iloc[temp1:,:].copy()
            x_i1.append(d_max.iloc[temp1:,:].copy())
            N_s_temp = len(d_max.iloc[temp1:,:])
            N_s.append(len(d_max.iloc[temp1:,:]))
            
            '''
            plt.hist(temp, bins=25, alpha=0.6, color='g')
            values, base = np.histogram(temp2, bins=40)
            plt.hist()
            n, bins, patches = plt.hist(pd.DataFrame.as_matrix(temp2), bins=25, alpha=1, color='g')
    #        cumulative = np.cumsum(values)
    #        plt.plot(base[:-1], cumulative, c='blue')
            plt.loglog(values,base)
            plt.loglog(temp2)        
            '''
            '''
            temp5 = temp0.iloc[temp1:i4,0].copy()
            temp5 = pd.DataFrame(temp5)
            temp1 = temp5.loc[temp5[i2] > 0].index[0].copy()
            x_min_k1.append(d_max.iloc[temp1,0])
            N_s_k1.append(len(x_i1[i3]))
            '''
            
            alpha_temp = alpha_coeff(N_s_temp,x_min_temp,x_i1_temp)
            alphas0.append(alpha_temp) 
            sigmas0.append(sigma_coef(alpha_temp,N_s_temp))
            
            alphas1.append(alpha_coeff(N_s_k[i3],x_min_k[i3],x_i_k[i3]))        
            sigmas1.append(sigma_coef(alphas1[i3],N_s_k[i3]))

        else:
            
            alphas0.append(float('inf'))        
            sigmas0.append(float('inf'))
            
            alphas1.append(float('inf'))        
            sigmas1.append(float('inf'))
        
        i3 = i3+1


#     return x_min_k,N_s_k,x_min_k1,N_s_k1,x_min_k1,x_i,x_i_k,x_min,N_s,x_i1
    return alphas0, sigmas0, alphas1, sigmas1, x_min
# d_max is also known as x_i

def get_drops(Asset,n_days):

    Asset = pd.DataFrame(Asset);
    Asset = (Asset.abs())
    
    drops_max = []

    for i in Asset.columns:
        d_i = []
        d_i = Asset.loc[Asset[i] == 0][i]
        d_max = pd.DataFrame(np.zeros((d_i.shape[0], 1)))

        temp = []
        temp = Asset[i].copy()

        for i1 in tqdm(range(len(d_i.index)-1)):
            m = d_i.index[i1]-(n_days-1)
            n = d_i.index[i1+1]-(n_days-1)
            d_max.iloc[i1] = max(temp[m:n])
        

        d_max.columns = [i]
        d_max = d_max.abs()
#        d_max = d_max.sort_values(by=i,axis=0,ascending=False)
#        d_max = d_max.reset_index(drop=True)
        d_max = d_max.loc[d_max[i] != 0]
        drops_max.append(d_max)
        
    return drops_max

def alpha_coeff(N_s,x_min,x_i1):
    
    temp = (x_i1)/x_min
    temp = np.log(temp)
    temp = sum(temp.iloc[:,0])
#    temp = sum(temp)
    alpha = 1+(1/temp)*N_s
    return alpha

def sigma_coef(alpha,N_s):
    temp = pow(float(N_s),1/2.0)
    sigma = (alpha-1.0)/temp
    return sigma
    
def max_drops(Asset,n_days):
    c_max = cmax(Asset,n_days)
    c_min = cmin(Asset,n_days)
    
    d_downs = Asset.iloc[n_days-1:,:].subtract(c_max, fill_value=0) 
    d_ups = Asset.iloc[n_days-1:,:].subtract(c_min, fill_value=0)

    cols = [];    
    i1 = 0;
    for i2 in d_downs:
        cols.append(list(d_downs)[i1]+' drop downs ' + str(n_days))
        i1 = i1+1;
    d_downs.columns = cols;

    cols = [];    
    i1 = 0;
    for i2 in d_ups:
        cols.append(list(d_ups)[i1]+' drop ups ' + str(n_days))
        i1 = i1+1;
    d_ups.columns = cols;


    return d_downs, d_ups

def multi_max_drops(Asset, vector_days):

    Asset = pd.DataFrame(Asset);
    cols = [];
    
    i1 = 0;
    for i2 in vector_days:
        i1 = 0
        for i in list(Asset):
            cols.append(list(Asset)[i1]+' ' + str(i2))
            i1 = i1+1;
      
    i1 = 0;
    i2 = Asset.shape[1]-1
    
    temp1 = [];
    temp1 = pd.DataFrame(temp1);

    temp2 = [];
    temp2 = pd.DataFrame(temp2);

    d_downs = []
    d_ups = []
    
    d_downs = pd.DataFrame(d_downs);
    d_ups = pd.DataFrame(d_ups);
    
    for i in tqdm(vector_days):
        if i == vector_days[0]:
            d_downs, d_ups = max_drops(Asset,i)
        else: 
            temp1, temp2 = max_drops(Asset,i)
            d_downs = pd.concat([d_downs, temp1], axis=1)
            d_ups = pd.concat([d_ups, temp2], axis=1)
    
    d_downs.columns = cols;
    d_ups.columns = cols;
        
    return d_downs, d_ups


def power_laws_price(Asset,n_days):
    
    Asset = pd.DataFrame(Asset);
    c_i = Asset.copy()
    #vc_i = pd.DataFrame.as_matrix(c_i)
    #     Asset = Asset.iloc[:l-2,:]
    
    l = len(Asset)
    r_i = quant.logret_multi(Asset.iloc[:l-2,:],[1])

# TODO: make this another function
    plt.figure()
    for i in r_i:
        temp = r_i[i].iloc[2:]
        values, base = np.histogram(temp, bins=40)
        # values, base = np.histogram(np.log(d_max), bins=40)
        cumulative = np.cumsum(values)
        plt.plot(base[:-1], cumulative)
# TODO: Make this another functin        
    for i in r_i:
        plt.figure()
        temp = r_i[i].iloc[2:]
        mu, std = norm.fit(temp) # as the mean (mu), and the std are necessary 
        # to fit a dist. maybe using the definitions of Mark E. Newman for
        # lepto data, can get to results
        plt.hist(temp, bins=25, alpha=0.6, color='g')

    S_r = r_i.std()
    K_r = r_i.kurt()
    N_r = len(r_i)
    
#    c_max = cmax(Asset,n_days)
#    c_min = cmin(Asset,n_days)   
#    d_downs = c_i.iloc[n_days-1:,:].subtract(c_max, fill_value=0) 
#    d_ups = c_i.iloc[n_days-1:,:].subtract(c_min, fill_value=0)

    d_downs, d_ups = max_drops(Asset,n_days)

    drops = pd.concat([Asset, d_downs, d_ups], axis = 1)
    drops_corr = drops.iloc[2:,:].corr()

    alphas0_d, sigmas0_d, alphas1_d, sigmas1_d, d_max_d = dec_i(d_downs.iloc[n_days-1:,:],n_days)
    alphas0_u, sigmas0_u, alphas1_u, sigmas1_u, d_max_u = dec_i(d_ups.iloc[n_days-1:,:],n_days)
    
    return alphas0_d, sigmas0_d, alphas1_d, sigmas1_d, alphas0_u, sigmas0_u, alphas1_u, sigmas1_u


def EntWindow(Asset,window,function,m,delay):

    Asset = pd.DataFrame(Asset);
    Ent_window = []
    Ent_window = pd.DataFrame(0, index=np.arange(Asset.shape[0]), columns=['Data']);
#    Ent_window = pd.DataFrame(0, index=np.arange(Asset.shape[0]), columns=list(Asset));
    Ent = [];
    Ent = pd.DataFrame(Ent,columns=list(Asset));
    window = int(window);

    
    for i1 in Asset:
        i2 = 0;
        temp = Asset[i1].copy()
        Ent_window = []
        Ent_window = pd.DataFrame(0, index=np.arange(Asset.shape[0]), columns=['Data']);
        if function == 'PE':
            for i in tqdm(range(window,len(temp)+1)):
                Ent_window.iloc[i-1] = entroPY.permutation_entropy(temp.iloc[i2:i],m,delay)
                i2 = i2+1                
        elif function == 'Shannon':
            for i in tqdm(range(window,len(temp)+1)):
                Ent_window.iloc[i-1] = entroPY.shannon_entropy(temp.iloc[i2:i])
                i2 = i2+1
        
        Ent[i1] = Ent_window
    
    return Ent


# vector_window = [80,100,180];

def multi_EntWindow(Asset,vector_window,function,m,delay):

    Asset = pd.DataFrame(Asset);
    cols = [];
    
    i1 = 0;
    for i2 in vector_window:
        i1 = 0
        for i in list(Asset):
            cols.append(list(Asset)[i1]+' Entropy window ' + str(i2))
            i1 = i1+1;
    
    i1 = 0;
    i2 = Asset.shape[1]-1
    
    temp = [];
    temp = pd.DataFrame(temp);
    Ent_Window = [];
    Ent_Window = pd.DataFrame(Ent_Window);
    
    for i in vector_window:
        if i == vector_window[0]:
            Ent_Window_multi = EntWindow(Asset,i,function,m,delay);
        else: 
            temp = EntWindow(Asset,i,function,m,delay);
            Ent_Window_multi = pd.concat([Ent_Window_multi, temp], axis=1)
    
    Ent_Window_multi.columns = cols;
    
    return Ent_Window_multi

###############################################################################
# Time



def len_dec(Asset,n_days):

    Asset = pd.DataFrame(Asset);
    Asset = (Asset.abs())

    dec = []
    for i in Asset.columns:

# cut from here
        d_i = []
        d_i = Asset.loc[Asset[i] == 0][i]
        d_max = pd.DataFrame(np.zeros((d_i.shape[0], 1)))

        temp = []
        temp = Asset[i].copy()

        for i1 in range(len(d_i.index)-1):
            m = d_i.index[i1]-(n_days-1)
            n = d_i.index[i1+1]-(n_days-1)
            d_max.iloc[i1] = len(temp[m+1:n])
        

        d_max.columns = [i]
        d_max = d_max.abs()
#        d_max = d_max.sort_values(by=i,axis=0,ascending=False)
#        d_max = d_max.reset_index(drop=True)
#        d_max = d_max.loc[d_max[i] != 0]
        dec.append(d_max)

    return dec



def time_dec(Asset,n_days):

    Asset = pd.DataFrame(Asset);
    Asset = (Asset.abs())

    dec = []
    counter = 0
    d_max = pd.DataFrame(np.zeros((Asset.shape[0], 4)))
    cols = [] 

    for i in Asset.columns:
        d_i = []
        d_i = Asset.loc[Asset[i] == 0][i]


        temp = []
        temp = Asset[i].copy()
        
        cols.append('zeros '+i)

        '''
        for i1 in range(len(d_i.index)-1):
            m = d_i.index[i1]-(n_days-1)
            n = d_i.index[i1+1]-(n_days-1)
            d_max.iloc[i1] = len(temp[m+1:n])
        '''        

        i3 = 0
        for i1 in tqdm(range(len(d_i)-1),ascii=True, desc='decreases in time of '+i):
            for i2 in range((d_i.index[i1+1]-d_i.index[i1])):
                d_max.iloc[i3,counter] = i2
                i3 = i3+1
#        d_max = d_max.abs()
#        d_max = d_max.sort_values(by=i,axis=0,ascending=False)
#        d_max = d_max.reset_index(drop=True)
#        d_max = d_max.loc[d_max[i] != 0]
        counter = counter+1
    
    d_max.columns = cols
    return d_max


