import pandas as pd
from openpyxl import Workbook

arr = [['10us', '26us'], ['1ms', '75us'], ['50us', '75us'], ['1ms', '75us', '50us'], ['10us', '26us', '50us'], ['26us', '50us', '75us'], ['10us', '26us', '50us', '75us']]
#val = [['10us', '26us', '50us', '75us'], ['26us', '50us', '75us'], ['50us', '75us'], ['1ms', '75us', '50us'], ['1ms'], ['10us'], ['75us'], ['1ms', '75us'], ['10us', '26us']]
k=0
book = Workbook()
sheet = book.active

#applications = {'astar' : 4, 'bwaves' : 13, 'bzip2' : 21, 'gamess' : 8, 'gobmk': 12, 'gromacs' : 9, 'h264ref': 9 , 'hmmer': 20, 'lbm':17, 'leslie3d' : 13, 'libquantum':6, 'mcf':9, 'milc':16, 'namd':6, 'omnetpp':8, 'povray':5, 'sjeng':6, 'soplex':12, 'tonto':9 , 'xalancbmk':14, 'zeusmp':20, 'gemsfdtd':12,'gsm':2,'lame':16, 'patricia':10, 'm_cjpeg':9, 'm_djpeg':2 }

df_1 = pd.read_excel(r'C:\Users\gaura\OneDrive\Desktop\Simpoint_data_gem5\user_experience_project\SPEC\d_cache\old\DATA_FOR_Classifier.xlsx','new_data')
df_1 = df_1.iloc[:,[0,1,2]]
df_1 = df_1.iloc[1:262,:]

for j in range(1,262):
    
    df_row = df_1.iloc[j,:].values
    df_row = pd.DataFrame(df_row)
    df_row = df_row.T
    app_array = df_row[0].values
    app = app_array[0]
    phs_array = df_row[1].values
    phs = int(phs_array[0])
    v_array = df_row[2].values
    v = v_array[0]
    v = v.split(",")
    
    df = pd.read_excel(r'C:\Users\gaura\OneDrive\Desktop\Simpoint_data_gem5\user_experience_project\SPEC\d_cache\XLSX files\{}_SPEC_data.xlsx'.format(app))
    #phs = applications[app]
            
    #for phs in range(1,phs):
        
    
    
    k += 1 
    for i in range(len(arr)):
        temp=[]
        L1 = list(set(v)-set(arr[i]))
        if L1 != []:
            for l in L1:
                df_row_1 = df[(df['phase no.'] == phs) & (df['ret_time_sec'] == l) ]
                edp_l = df_row_1['edp']
               # edp_l = pd.Series(edp_l).astype('float64')
                edp_l = pd.DataFrame(edp_l)
                e = edp_l['edp'].values
                print(-e[0],phs,l, 'L1')
                temp.append(-(e[0]))
        
        L2 = list(set(arr[i]) - set(v))
        if L2 != []:
            for l in L2:
                df_row_1 = df[(df['phase no.'] == phs) & (df['ret_time'] == l) ]
                edp_l = df_row_1['edp']
                #edp_l = pd.Series(edp_l).astype('float64')
                edp_l = pd.DataFrame(edp_l)
                e = edp_l['edp'].values
                temp.append(e[0])
                print(e[0],phs,l, 'L2')
        
        sheet.cell(row=k, column=i+1).value = sum(temp)

book.save('edp_new_data.xlsx')
