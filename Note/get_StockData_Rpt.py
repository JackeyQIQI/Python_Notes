'''
Get stock report data
- continue from the dropped point
    - sometimes the get data function will be time out and the whole program will stop
    - then just change the lists and re-run this program and it will continue from the dropped point
Created on Sep 27, 2016
@author: JiaqiLu
'''

import tushare as ts
import pandas
import os

#定义路径，起始年份，结束年份季度
filepath='D:\WorkFile\StkProj\data_20160927\stk_rpt'
startYear=2010
endYear=2016
endSeason=2

#如果要断点续传，修改这里
#'201001', '201002', '201003', '201004', '201101', '201102', '201103', '201104', '201201', '201202', '201203', '201204', '201301', '201302', '201303', '201304', '201401', '201402', '201403', '201404', '201501', '201502', '201503', '201504', '201601', '201602'
list_rpt=[]
list_profit=[]
list_operation=[]
list_growth=[]
list_debtpay=[]
list_cashflow=[]

#全部季度列表
list_all=[]
for y in range(startYear,endYear+1):
    if y == endYear:
        for s in range(1,endSeason+1):
            season=str(y)+'0'+str(s)
            list_all.append(season)
    else:
        for s in range(1,5):
            season=str(y)+'0'+str(s)
            list_all.append(season)
            
print(list_all)

#业绩报告（主表）
filename = filepath+'_'+str(startYear)+'01_'+str(endYear)+'0'+str(endSeason)+'.csv'
for q in list_all:
    if q not in list_rpt:
        y=int(q[:4])
        s=int(q[-1])
        print("\ngetting, main report: " + q)
        df=ts.get_report_data(y,s)
        df['quarter']=q
        list_rpt.append(q)
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=None)
        else:
            df.to_csv(filename)

#盈利能力
filename = filepath + '_profit'+'_'+str(startYear)+'01_'+str(endYear)+'0'+str(endSeason)+'.csv'
for q in list_all:
    if q not in list_profit:
        y=int(q[:4])
        s=int(q[-1])
        print("\ngetting, profit report: " + q)
        df=ts.get_profit_data(y,s)
        df['quarter']=q
        list_profit.append(q)
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=None)
        else:
            df.to_csv(filename)

#营运能力
filename = filepath + '_operation'+'_'+str(startYear)+'01_'+str(endYear)+'0'+str(endSeason)+'.csv'
for q in list_all:
    if q not in list_operation:
        y=int(q[:4])
        s=int(q[-1])
        print("\ngetting, operation report: " + q)
        df=ts.get_operation_data(y,s)
        df['quarter']=q
        list_operation.append(q)
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=None)
        else:
            df.to_csv(filename)

#成长能力
filename = filepath + '_growth'+'_'+str(startYear)+'01_'+str(endYear)+'0'+str(endSeason)+'.csv'
for q in list_all:
    if q not in list_growth:
        y=int(q[:4])
        s=int(q[-1])
        print("\ngetting, growth report: "+ q)
        df=ts.get_growth_data(y,s)
        df['quarter']=q
        list_growth.append(q)
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=None)
        else:
            df.to_csv(filename)

#偿债能力
filename = filepath + '_debtpay'+'_'+str(startYear)+'01_'+str(endYear)+'0'+str(endSeason)+'.csv'
for q in list_all:
    if q not in list_debtpay:
        y=int(q[:4])
        s=int(q[-1])
        print("\ngetting, debtpay report: " + q)
        df=ts.get_debtpaying_data(y,s)
        df['quarter']=q
        list_debtpay.append(q)
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=None)
        else:
            df.to_csv(filename)
            
#现金流量
filename = filepath + '_cashflow'+'_'+str(startYear)+'01_'+str(endYear)+'0'+str(endSeason)+'.csv'
for q in list_all:
    if q not in list_cashflow:
        y=int(q[:4])
        s=int(q[-1])
        print("\ngetting_cashflow report: " + q)
        df=ts.get_cashflow_data(y,s)
        df['quarter']=q
        list_cashflow.append(q)
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=None)
        else:
            df.to_csv(filename)
