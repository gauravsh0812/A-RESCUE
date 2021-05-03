# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:13:32 2020

@author: gaura
"""
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')
applications = {'astar':2, 'bwaves':10, 'bzip2':18, 'gamess':3, 'gromacs':7, 'patricia':1, 'soplex':7, 'tonto':2, 'zeusmp':1}


#fig1 = plt.figure(figsize=(15, 15))

for index, app in  enumerate(list(applications.keys())):
    
    file = pd.read_excel(r"C:\Users\gaura\OneDrive\Desktop\Simpoint_data_gem5\user_experience_project\SPEC\d_cache\XLSX files\{}_SPEC_data.xlsx".format(app))
    phase = applications[app]
    
    file1 = file.loc[:,['phase no.','latency', 'energy']]
    file1 = file1.loc[file1['phase no.'] == phase]   
    
    x= file1.iloc[:,-2]
    y= file1.iloc[:,-1]
    matplotlib.rcParams.update({'font.size': 7})
    
    #ax = fig.add_subplot(3, 3, index+1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x,y, '-o')                                   #nor_energy v/s nor_Lat 
    #ax.set_xlabel('Latency')
    #ax.set_ylabel('Energy')
    #ax.set_title(app, fontname="Times New Roman Bold")
    plt.xlabel('Latency')
    plt.ylabel('Energy')
    plt.title(app)
    plt.show()
    plt.tight_layout()
    plt.savefig(r'C:\Users\gaura\OneDrive\Desktop\Simpoint_data_gem5\user_experience_project\SPEC\d_cache\Plots\Seperated_phase_plot\{}_Para-optimal curve'.format(app))

'''  
    ax1 = fig1.add_subplot(3, 3, index+1)
    #color = 'tab:red'
    ax1.set_ylabel('Latency')
    ax1.plot(x, '-o')
    #ax1.tick_params(axis='x', labelcolor=color)
    ax1.set_title(app, fontname="Times New Roman Bold")
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Energy', color=color)  # we already handled the x-label with ax1
    ax2.plot(y, '-o')
    #ax2.tick_params(axis='y', labelcolor=color)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
'''
   # fig1.tight_layout()

#plt.title('Para-optimal curve')
#plt.savefig('Para-optimal curve')
