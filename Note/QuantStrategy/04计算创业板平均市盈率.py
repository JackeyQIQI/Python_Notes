# -*- coding: utf-8 -*-
import os
import pandas as pd

# ========== 遍历数据文件夹中所有股票文件的文件名，得到股票代码列表stock_code_list
stock_code_list = []
for root, dirs, files in os.walk('trading-data@full/stock data'):# 注意：这里请填写数据文件在您电脑中的路径
    if files:
        for f in files:
            if '.csv' in f:
                stock_code_list.append(f.split('.csv')[0])


# ========== 根据上一步得到的代码列表，遍历所有创业板股票，将这些股票合并到一张表格all_stock中
all_stock = pd.DataFrame()

# 遍历每个创业板的股票
for code in stock_code_list:
    # 创业板股票代码都是以3开头，不是3开的股票跳过
    if code[2] != '3':
        continue

    # 从csv文件中读取该股票数据
    stock_data = pd.read_csv('trading-data@full/stock data/' + code + '.csv',
                             parse_dates=[1])# 注意：这里请填写数据文件在您电脑中的路径

    # 删除PE_TTM值为空的数据行
    stock_data = stock_data[stock_data['PE_TTM'].notnull()]

    # PE_TTM = 总市值 / 净利润_TTM，这里通过这个公式计算净利润_TTM
    stock_data['净利润'] = stock_data['market_value'] / stock_data['PE_TTM']

    # 选取需要的字段，去除其他不需要的字段
    stock_data = stock_data[['code', 'date', 'market_value', '净利润']]

    # 将该股票的合并到output中
    all_stock = all_stock.append(stock_data, ignore_index=True)


# ========== 基于all_stock表格，通过groupby语句，计算创业板股票每天的平均市盈率
# 通过groupby语句计算每天所有股票的市值之和、净利润之和，以及当天交易的股票的数量
output = all_stock.groupby('date')[['market_value', '净利润']].sum()
output['股票数量'] = all_stock.groupby('date').size()

# 平均市盈率 = 所有股票的市值之和 / 所有股票的净利润之和
output['创业板平均市盈率'] = output['market_value'] / output['净利润']

# 算好的数据输出
output.to_csv('创业板平均市盈率.csv', encoding='gbk') # 注意：这里请填写数据文件在您电脑中的路径
print output