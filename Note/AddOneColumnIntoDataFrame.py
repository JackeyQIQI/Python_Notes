#向DataFrame中插入一列D，值全为ColumnD

import pandas as pd
filepath = 'H://TestPYTHON//datapub.csv'
dataheader = ['A','B','C']
data_original = pd.read_csv(filepath, sep=',', header=0, names=dataheader, skip_blank_lines=True)
print data_original
data_original['D']='ColumnD'

print data_original

# In [4]: %run "H:/TestPYTHON/public.py"    
# A   B  C
# 0  a1  b1  
# 1  a2  b2  
# 2  a3  b3  
#
# A   B  C        D
# 0  a1  b1  ColumnD
# 1  a2  b2  ColumnD
# 2  a3  b3  ColumnD