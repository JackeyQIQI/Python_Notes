# -*- coding: utf-8 -*-
"""
@author: yucezhe
@contact: QQ:2089973054 email:xjc@yucezhe.com
"""
import pandas as pd

# ========== ��ԭʼcsv�ļ��е����Ʊ���ݣ����ַ�����sh600000Ϊ��

# �������� - ע�⣺��������д�����ļ����������е�·��
stock_data = pd.read_csv('stock data/sh600000.csv', parse_dates=[1])

# �����ݰ��ս������ڴ�Զ��������
stock_data.sort('date', inplace=True)


# ========== �����ƶ�ƽ����

# �ֱ����5�ա�20�ա�60�յ��ƶ�ƽ����
ma_list = [5, 20, 60]

# ����������ƶ�ƽ����MA - ע�⣺stock_data['close']Ϊ��Ʊÿ������̼�
for ma in ma_list:
    stock_data['MA_' + str(ma)] = pd.rolling_mean(stock_data['close'], ma)

# ����ָ��ƽ���ƶ�ƽ����EMA
for ma in ma_list:
    stock_data['EMA_' + str(ma)] = pd.ewma(stock_data['close'], span=ma)

# �����ݰ��ս������ڴӽ���Զ����
stock_data.sort('date', ascending=False, inplace=True)

# ========== ����õ����������csv�ļ� - ע�⣺��������д����ļ����������е�·��
stock_data.to_csv('sh600000_ma_ema.csv', index=False)