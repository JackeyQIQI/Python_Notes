'''
通过tushare包下载所有股票的历史前复权数据
第一次运行时使用第一段代码
因下载时间较长，容易time out中断,re-run时用第二段代码断点续传

@by JackeyLu
'''

#历史复权数据（全历史数据）（all in one file）
import tushare as ts
import pandas
import os
filename="D:\WorkFile\StkProj\data_20160919\stk_hist_daily.csv"

df = ts.get_stock_basics()
df.to_csv('D:\WorkFile\StkProj\data_20160919\stock_basics.csv')

for code in df.index:
    date = df.ix[code]['timeToMarket']
    if date == 0 or code == '601500':   #排除个别会报错终止的新股
        continue
    else:
        sdate = str(date)
        stdate = sdate[:4]+'-'+sdate[4:6]+'-'+sdate[6:8]
        print('\ngetting:'+code)
        hdf = ts.get_h_data(code,start=stdate,retry_count=10)
        hdf['code']=code
        if os.path.exists(filename):
            hdf.to_csv(filename, mode='a', header=None)
        else:
            hdf.to_csv(filename)



#历史复权数据(全历史数据, all in one file, 断点续传)
import tushare as ts
import pandas
import os

filename="D:\WorkFile\StkProj\data_20160919\stk_hist_daily.csv"
odf = pandas.read_csv(filename, sep=',', header=0,  skip_blank_lines=True, index_col=7)

df = ts.get_stock_basics()
df.to_csv('D:\WorkFile\StkProj\data_20160919\stock_basics.csv')

for code in df.index:
    date = df.ix[code]['timeToMarket']
    if date == 0 or code == '601500':  #排除个别会报错终止的新股
        continue
    if int(code) in odf.index:
        continue
    else:
        sdate = str(date)
        stdate = sdate[:4]+'-'+sdate[4:6]+'-'+sdate[6:8]
        print('\ngetting:'+code)
        hdf = ts.get_h_data(code,start=stdate,retry_count=20)
        hdf['code']=code
        if os.path.exists(filename):
            hdf.to_csv(filename, mode='a', header=None)
        else:
            hdf.to_csv(filename)
