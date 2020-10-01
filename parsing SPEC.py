import os 
from openpyxl import Workbook
import pandas as pd
import xlrd 

#df = pd.read_excel(r'C:\Users\gaura\OneDrive\Desktop\Simpoint_data_gem5\user_experience_project\SPEC\d_cache\data.xlsx')
#os.chdir('C:\Users\gaura\OneDrive\Desktop\Simpoint_data_gem5\user_experience_project\SPEC\d_cache')
loc = (r"C:\Users\gaura\OneDrive\Desktop\Simpoint_data_gem5\user_experience_project\SPEC\d_cache\data.xlsx") 
  
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 
#sheet.cell_value(0, 0) 
col =[1,9,17,25,33, 41, 49]
#col = [9]
file='bzip'
book = Workbook()
sheet_1 = book.active
r1=0
for i in col:
    r =0 
    while r < 227:
        if sheet.cell_value(r, i-1)  == file:
            r1+=1
            for k in range(7):
                sheet_1.cell(row=r1, column=k+1).value = sheet.cell_value(r, i -1 + k)
        r+=1
        
book.save('{}_SPEC_data.xlsx'.format(file))

