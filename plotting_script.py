import os 
import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

file = 'm_cjpeg'
df = pd.read_excel(r'C:\Users\gaura\OneDrive\Desktop\Simpoint_data_gem5\user_experience_project\SPEC\MI bench\{}.xlsx'.format(file))
for i in range(1,9):
    df_1 = df.loc[df['phase no.'] == i]
    df_1 = df_1.iloc[:,[1,2, -3, -2]]
    x = df_1.iloc[:,2]
    y = df_1.iloc[:,3]
    t = df_1.iloc[:,1]

    
    matplotlib.rcParams.update({'font.size': 5})
    plt.figure(1)
    plt.subplot(3,3,i)
    plt.plot(x,y,'-o')                                   #nor_energy v/s nor_Lat 
    #plt.title('phase-{}_bzip_SPEC_data'.format(i))
    plt.show()
    plt.savefig('{}_retTime_energy_lat_plots'.format(file))

    plt.legend()
