# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:38:38 2018
@author: JackeyLu@MF
### Tool Package for data analysis, score card modeling, and tests

### how to use:
    ### method 1:
    # exec(open("D:/02_Code/00_tools/00_package/score_card_modeling.py", encoding='utf-8').read())
    # function

    ### method 2:
    # import sys
    # sys.path.append('D:/02_Code/00_tools/00_package/')
    # import score_card_modeling as sc
    # sc.function
    
### function list:
    # 日期转换函数(已删除)
        以后用 df_all[base_date+"_date"] = df_all[base_date].apply(lambda x: pd.to_datetime(str(int(x))) if type(x)!=str and np.isnan(x)==False else pd.to_datetime(x)) 处理日期解析
	# 分组转换
        value2group(x, cutoffs, mv = [])
    # 计算KS值，生成 KS曲线 ROC曲线 lift曲线
        cal_ks(df, score_name, label_name, label_1='bad', KS_pic=None, ROC_pic=None, LIFT_pic=None)
    # 计算PSI
        cal_psi(df1, df2, var1, var2, cutoffs)
    # 评分分布图
        score_histogram(df_score, score_name, cutoffs, mv_list=[], his_pic=None, title_name='Score Histogram')
	# 评分查得率
        score_count(df_score, score_name, nan_name)
    # 缺失率分析
        ana_miss(df, var_list, mv_str=['','unknow','Unknow','unknown','Unknown','Null','NONE','none','None','NAN'])
    # 频数分析
        ana_freq(df, var_list)
    # 数值分析
        ana_num(df, var_list)
    # 全部变量autobin报告
        autobin_all_report(df, var_list, label_list, outfile, label_1='bad', max_groups=20, p_value=0.05, mv_list=[], speed_lv=100)
    # 全部变量finebin报告
        finebin_all_report(df, df_cutoffs, var_list, label_list, outfile, label_1='bad', mv_list=[])
    # 单变量finebin表
        finebin_single_var(df, cutoffs, var, label, label_1='bad', mv_list=[], pic_file=None, title_name='Histogram')
    # 评分价值评估
        score_value_test(outfile_path, df_all, score_title, score_col, label_col, label1='bad')
    # 十进制精确四舍五入
        round_up(value)
    # 删除全部为0或空的行
        drop_nan_row(df, sn)
    # 删除自定义全部为空的行
        drop_empty_row(df, except_col=['序号'], list_empty_type=['0', '', 'nan', 'null'])
    # 生成随机因子
        rand_factor(df, id_name, mob_name)
    # 生成范围内的随机整数
        rand_int(df, start, stop, id_name, mob_name)

### 数据加解密工具 Class: Encrypt_tool
    # encrypt2csv(self, infile, outfile, encoding='utf-8')
    # decrypt2df(self, infile, encoding='utf-8')
"""

#依赖包，未写进函数，使用函数时需要提前调用：
import pandas as pd
import numpy as np
from datetime import datetime
startTime = datetime.now()

## 个别函数用到的包，已写进函数，不用另外调用，需保证已安装对应包：
## scipy
## matplotlib
## base64
## getpass
## cryptography
## decimal


###########################     变量值转分组     ###############################
# x: 分组的变量
# cutoffs: 切分点列表
# mv: 字符型缺失值列表，可自定义，默认为空
def value2group(x, cutoffs, mv = []):
    cutoffs = sorted(cutoffs)
    num_groups = len(cutoffs)
    if num_groups == 1:
        return cutoffs[0]
    elif x == '' or x in mv:
        return 'None'
    elif type(x) == str:
        return x
    elif np.isnan(x):
        return -9999
    elif x < cutoffs[0]:
        return -8888
    else:
        for i in range(1, num_groups):
            if cutoffs[i-1] <= x < cutoffs[i]:
                return cutoffs[i-1]
        return cutoffs[num_groups-1]
################################################################################



############ 计算 KS值,精确累计值 生成 KS曲线 ROC曲线 lift曲线 #######################
### 输入：
    # df: 要分析的数据表
    # score_name: 评分列名称
    # label: 坏标签列名
    # label_1: 默认 label中1代表bad时，label_1 = 'bad'， 
    #          label中1代表good时, label_1可定义为其他值
    # weight: 默认为None,指定weight列时，按weight权重计算。
   
### 输出：
    # 返回KS值,数值型
    # 在console中显示 KS曲线，ROC曲线，Lift曲线
    # 也可另存图片 KS曲线，ROC曲线，Lift曲线
    
def cal_ks(df, score_name, label_name, label_1 = 'bad', KS_pic=None, ROC_pic=None, LIFT_pic=None, weight=None):
    from matplotlib import pyplot as plt
    #生成ks表
    if label_1 == 'bad' and weight!=None:
        df_ks = df[[score_name, label_name, weight]]
        df_ks['bad']  = df_ks[label_name]*df_ks[weight]
        df_ks['good'] = df_ks[label_name].apply(lambda e: 1-e)*df_ks[weight]
    elif label_1 != 'bad' and weight!=None:
        df_ks = df[[score_name, label_name, weight]]
        df_ks['bad']  = df_ks[label_name].apply(lambda e: 1-e)*df_ks[weight]
        df_ks['good'] = df_ks[label_name]*df_ks[weight]
    elif label_1 == 'bad' and weight==None:
        df_ks = df[[score_name, label_name]]
        df_ks['bad']  = df_ks[label_name]
        df_ks['good'] = df_ks[label_name].apply(lambda e: 1-e)
    else:
        df_ks = df[[score_name, label_name]]
        df_ks['bad']  = df_ks[label_name].apply(lambda e: 1-e)
        df_ks['good'] = df_ks[label_name]
    df_ks = df_ks.sort_values(by = [score_name], ascending = True)
    df_ks = df_ks.drop(label_name,axis=1)
    df_ks = df_ks[np.isfinite(df_ks['bad']) & np.isfinite(df_ks['good'])]
    
    #生成ks计算结果
    df_ks_result = df_ks.groupby(score_name).sum()
    bad_sum  = df_ks_result['bad'].sum()
    good_sum = df_ks_result['good'].sum()
    
    df_ks_result['bad_vpct']  = df_ks_result['bad'].apply(lambda e: e/bad_sum)
    df_ks_result['good_vpct'] = df_ks_result['good'].apply(lambda e: e/good_sum)
    df_ks_result['bad_vcumpct']  = 100*df_ks_result['bad_vpct'].cumsum()
    df_ks_result['good_vcumpct'] = 100*df_ks_result['good_vpct'].cumsum()
    df_ks_result['cumpct_dif']  = abs(df_ks_result['bad_vcumpct'] - df_ks_result['good_vcumpct'])
    
    ks = df_ks_result['cumpct_dif'].max()
    
    # KS值
    print("KS值：" + str(round(ks, 2)))
    
    # KS曲线图
    df_ks_plot = df_ks_result[['bad_vcumpct', 'good_vcumpct']]
    fig = plt.figure()
    plt.title('KS = ' + str(round(ks, 2)))
    plt.plot(df_ks_plot)
    plt.show()
    if KS_pic!=None:
        fig.savefig(KS_pic, dpi=fig.dpi)
        
    # ROC曲线
    fig = plt.figure()
    plt.title('ROC')
    plt.scatter(df_ks_result['good_vcumpct'].tolist(), df_ks_result['bad_vcumpct'].tolist())
    plt.show()
    if ROC_pic!=None:
        fig.savefig(ROC_pic, dpi=fig.dpi)
        
    # lift曲线
    df_ks_result['total'] = df_ks_result['bad'] + df_ks_result['good']
    total_sum  = df_ks_result['total'].sum()
    df_ks_result['bad_h_pct']  = df_ks_result['bad']/df_ks_result['total']
    df_ks_result['total_pct']  = df_ks_result['total']/total_sum
    df_ks_result['bad_h_cumpct']  = df_ks_result['bad_h_pct'].cumsum()
    df_ks_result['total_cumpct']  = df_ks_result['total_pct'].cumsum()
    df_ks_result['lift'] = df_ks_result['bad_vcumpct']/df_ks_result['total_cumpct']/100
    df_lift_plot = df_ks_result[['lift']]
    fig = plt.figure()
    plt.title('LIFT')
    plt.plot(df_lift_plot)
    plt.show()
    if LIFT_pic!=None:
        fig.savefig(LIFT_pic, dpi=fig.dpi)
    
    return round(ks, 2)
######################################################################################



############################# 计算变量稳定性 PSI ############################################
### 稳定性指标 PSI 计算同一个变量在跨时间样本中的稳定性
# df1: 表1
# var1: 表1中的变量1
# df2: 表2
# var2: 表2中的变量2
# cutoffs: 列表，切分点

# 返回： PSI计算表，PSI值
        
def cal_psi(df1, df2, var1, var2, cutoffs):
    df1['v1_group'] = df1[var1].apply(lambda x: value2group(x, cutoffs, mv = []))
    s_1 = df1['v1_group'].value_counts()
    s_1 = s_1.sort_index(axis=0)
    df_his_1 = pd.DataFrame({'计数1':s_1})
    df_his_1 = df_his_1.reset_index()
    total_count_1 = df_his_1['计数1'].sum(axis=0)
    df_his_1['占比1'] = df_his_1['计数1'].apply(lambda x: round((x/total_count_1),6))
    
    df2['v2_group'] = df2[var2].apply(lambda x: value2group(x, cutoffs, mv = []))
    s_2 = df2['v2_group'].value_counts()
    s_2 = s_2.sort_index(axis=0)
    df_his_2 = pd.DataFrame({'计数2':s_2})
    df_his_2 = df_his_2.reset_index()
    total_count_2 = df_his_2['计数2'].sum(axis=0)
    df_his_2['占比2'] = df_his_2['计数2'].apply(lambda x: round((x/total_count_2),6))
    
    df_his = pd.merge(df_his_1, df_his_2, on = 'index', how = 'outer')
    df_his = df_his.rename(columns={'index':'取值区间'})
    df_his['PSI'] = (df_his['占比1'] - df_his['占比2'])*(np.log(df_his['占比1']/df_his['占比2']))
    v_psi = round(df_his['PSI'].sum(axis=0), 4)
    
    df_temp = pd.DataFrame({'占比2':'SUM', 'PSI':v_psi},index=[0])
    df_his = pd.concat([df_his,df_temp])
    df_his = df_his[['取值区间', '计数1', '计数2', '占比1', '占比2', 'PSI']]
    return df_his, v_psi
#################################################################################################



######################################    评分分布图    ##############################################
### 输入:
    # df_score：含有评分的数据表
    # score_name：评分列名称
    # cutoffs：切分点列表
    #          建议写法： cutoffs = np.arange(400,800,10)
    #                    cutoffs = np.arange(df_score[score_name].min(), df_score[score_name].max(), 10)
    # weight：权重字段名，默认值为None
    # mv_list: 字符型缺失值列表，可自定义，默认为空
    # his_pic: 图片保存途径，默认为None不保存
### 输出：
    # 评分分布图，可自定义保存图片到本地
    # 评分分布表
    
def score_histogram(df_score, score_name, cutoffs, weight=None, mv_list=[], his_pic=None, title_name='Score Histogram'):
    from matplotlib import pyplot as plt
    if weight==None:
        df_temp['weight_tmp']=1
        weight = 'weight_tmp'
    
    df_temp = df_score.copy()
    df_temp[score_name + '_group'] = df_temp[score_name].apply(lambda x: value2group(x, cutoffs, mv = mv_list))
    df_his_score = df_temp.groupby([score_name+'_group'])[weight].sum().reset_index()
    count_sum = df_his_score[weight].sum()
    df_his_score['pct'] = df_his_score[weight]/count_sum
    
    fig = plt.figure()
    plt.title(title_name)
    plt.bar(np.arange(len(df_his_score)), df_his_score[weight].tolist(), tick_label=df_his_score[score_name+'_group'].apply(lambda x: int(x)).tolist())
    plt.show()
    if his_pic!=None:
        fig.savefig(his_pic, dpi=fig.dpi)
        
    return df_his_score
#####################################################################################################



##################################  评分查得率  #################################
### 统计评分数、未查得数、和查得率
### 输入：
	# df_score：含有评分的数据表
	# score_name：评分列名称
	# nan_name: 未查得填充的值
### 输出：
	# df_log：评分查得情况统计表
	
def score_count(df_score, score_name, nan_name):
    total = len(df_score)
    df_temp = df_score.copy()
    df_temp['score_str'] = df_temp[score_name].apply(lambda x: str(x)) #如果没有未查得，那么score列为数值型，那下一行会报错
    count_nan = len(df_temp[df_temp['score_str']==nan_name]) 
    count_score = total - count_nan
    df_log = pd.DataFrame({'0产品名称':score_name,'1查得数':count_score,'2未查得':count_nan,'3查询数':total,'4查得率': str(round((count_score/total)*100,2))+'%'},index=[0])
    return df_log
#################################################################################



###########################      缺失率函数     #################################
### 输入：
    # df: 读入的数据表
    # var_list: 要分析的变量名列表
    # mv_str: 认定为缺失值的字符串,默认为'unknown'
### 输出：
    # 缺失率表

def missing_ana(df, var_list, mv_str='unknown'):
    s_null_pct = round((df.isnull().sum()/len(df))*100,2)
    s_count = df.count()

    total = len(df)
    d_zeropct = []
    d_emptypct = []
    for var in var_list:
#        print(var)
        count = df[var].tolist().count(0)
        d_zeropct.append(round(100*count/total,2))
        count = df[var].tolist().count(mv_str)
        d_emptypct.append(round(100*count/total,2))

    df_out = pd.DataFrame({'1查得记录条数':total, '2未缺失条数':s_count, '3缺失值率/查得数':s_null_pct, '4零值率/查得数':d_zeropct, '5空字符率/查得数':d_emptypct})

    print('miss analysis complete!')
    print('Time: ' + str(datetime.now() - startTime))
    return df_out
#################################################################################
    


###########################     频数分析函数     #################################
### 输入：
    # df: 读入的数据表
    # var_list: 要分析的变量名列表
### 输出:
    # 频数分布表
def ana_freq(df, var_list):
    varname = []
    varvalue = []
    varcount = []
    varpct = []
    total = len(df)
    df_freq = df.fillna('NaN')
    df_freq = df_freq.applymap(lambda x: str(x))
    
    for i in range(len(var_list)):
        freq_list=df_freq[var_list[i]].value_counts(dropna = False)
        for j in range(len(freq_list)):
            varname.append(var_list[i])
            varvalue.append(freq_list.index[j])
            varcount.append(freq_list[freq_list.index[j]])
            varpct.append(round(100*freq_list[freq_list.index[j]]/total,2))
    
    df_out = pd.DataFrame({'1标签':varname,'2取值':varvalue,'3计数':varcount,'4占比':varpct})
    df_out = df_out.sort_values(by = ['1标签','2取值'], ascending = True)
    
    print('frequency analysis complete!')
    print('Time: ' + str(datetime.now() - startTime))
    return df_out
#################################################################################
    
	

#############################   数值分布函数   ##################################
### 输入：
    # df: 读入的数据表
    # var_list: 要分析的变量名列表
### 输出:
    # 数值分布表
    
def ana_num(df, var_list):
    df_num       = df[var_list]
    df_num.index = range(len(df_num))
    df_num.columns = var_list
    df_describe = df_num.describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95])
    df_describe = round(df_describe.T, 2)
    
    print('number analysis complete!')
    print('Time: ' + str(datetime.now() - startTime))
    return df_describe
#################################################################################



###########################    卡方自动分箱     ###############################
############    卡方值计算    #############
def chi_sq(arr):
    assert(arr.ndim==2)
    R_N = arr.sum(axis=1)
    C_N = arr.sum(axis=0)
    N = arr.sum()
    E = np.ones(arr.shape)*C_N/N
    E = (E.T * R_N).T
    square = (arr-E)**2/E
    square[E==0] = 0
    v = square.sum()
    return v
###########################################
    
###### 卡方分箱，返回分箱值 左闭右开 ######
def chiMerge(df, col, target, max_groups=20, p_value=0.05):
    from scipy.stats import chi2
    freq_tab = pd.crosstab(df[col],df[target])
    freq = freq_tab.values
    freq_idx = freq_tab.index.values
    freq_idx = list(freq_idx)
    
    if len(freq_idx) == 1:
        return freq_idx
    elif type(freq_idx[0]) == str:
        return freq_idx
    else:        
        cls_num = freq.shape[-1]
        threshold = chi2.isf(p_value, df= cls_num)
        
        while True:
            minvalue = np.nan
            minidx = np.nan
            
            for i in range(len(freq)-2):
                v = chi_sq(freq[i:i+2])
                if np.isnan(minvalue) or minvalue > v :
                    minvalue = v
                    minidx = i
                    
            if len(freq_idx) <= 2:
                break
            elif (max_groups < len(freq)) or (minvalue < threshold):
                tmp = freq[minidx] + freq[minidx+1]
                freq[minidx] = tmp
                freq = np.delete(freq, minidx+1, 0)
                freq_idx = np.delete(freq_idx, minidx+1, 0)
            else:
                break
            
        while True:
            if len(freq_idx) <= 2:
                break
            elif freq[len(freq)-1].sum() < 20:
                tmp = freq[len(freq)-2] + freq[len(freq)-1]
                freq[len(freq)-2] = tmp
                freq_idx = np.delete(freq_idx, len(freq)-1, 0)
                freq = np.delete(freq, len(freq)-1, 0)
            else:
                break
        
        return freq_idx
###############################################################################

##########     autobin report 计算IV值 WoE值，并输出表格      ###################
### 输入： 
    # df: 大宽表
    # var_list: x 变量列表
    # label_list: y 变量列表
    # outfile: 输出文件路径与文件名，xlsx文件
	# label_1: label为1时代表的好坏，默认为'bad'
    # max_groups: 组数超过时进行自动合并，默认值：20
    # p_value: 卡方检验p值，一般为 0.1、0.05、0.01，默认值：0.05
    # mv_list: 额外需要认定为空值的字符列表
    # speed_lv: 用于识别部分数值型变量，当取值数超过时将按百分点位切分，提升运行速度，建议取值范围20-200，取值越小越快，但越快切分效果可能越差，默认值：100
### 输出：
    # autobin woe报告，保存到本地outfile

def autobin_all_report(df, var_list, label_list, outfile, label_1 = 'bad', max_groups=20, p_value=0.05, mv_list=[], speed_lv=100):
    writer = pd.ExcelWriter(outfile)
    
    for label in label_list:
        label_name = label
        
        if label_1 != 'bad':
            df['bad'] = df[label].apply(lambda e: 1-e)
            label = 'bad'
        
        len_true = len(df[df[label] == 1])
        len_false = len(df[df[label] == 0])
        
        df_woe_out = pd.DataFrame({})
        df_iv_out = pd.DataFrame({})
        varname = []
        varvalue = []
        varvalue_count = []
        varvalue_count_true = []
        varvalue_count_false = []
        true_vert_pct = []
        false_vert_pct = []
        woe = []
        iv = []
        total_pct = []
        true_pct = []
        false_pct = []
        odds = []
        
        ivtb_varname = []
        ivtb_iv_sum = []
        cutoff_list = []
        n=1
        
        for var in var_list:
            print(str(n) +' Autobin: ' + label + ' ' + var)
            n=n+1
            
            freq_table = pd.crosstab(df[var],df[label],dropna = False)
            if len(freq_table)==0:
                continue
            
            freq_list = freq_table[0]+freq_table[1]
            
            if len(freq_list) > speed_lv and freq_table.index.values[0]!=str:
                descrb=df[var].describe(percentiles=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99])
                cutoffs = descrb[3:-1].tolist()
                cutoffs = list(set(cutoffs))
                cutoffs.sort()
                group_20_var = var+'_20pctl'
                df[group_20_var] = df[var].apply(lambda x: value2group(x, cutoffs, mv = mv_list))
                cutoffs = chiMerge(df, group_20_var, label, max_groups=max_groups, p_value=p_value)
                group_var = var+'_group'
                df[group_var] = df[var].apply(lambda x: value2group(x, cutoffs, mv = mv_list))
                freq_table = pd.crosstab(df[group_var],df[label],dropna = False)
                freq_list = freq_table[0]+freq_table[1]
                freq_true_list  = freq_table[1]
                freq_false_list = freq_table[0]
            else:
                cutoffs = freq_table.index.values
                cutoffs = chiMerge(df, var, label, max_groups=max_groups, p_value=p_value)
                group_var = var+'_group'
                df[group_var] = df[var].apply(lambda x: value2group(x, cutoffs, mv = mv_list))
                freq_table = pd.crosstab(df[group_var],df[label],dropna = False)
                freq_list = freq_table[0]+freq_table[1]
                freq_true_list  = freq_table[1]
                freq_false_list = freq_table[0]
            
            iv_sum = 0
            for value in freq_list.index:
                freq_true_count = freq_true_list[value]
                freq_false_count = freq_false_list[value] 
                
                true_vert_pct_value = freq_true_count/len_true
                false_vert_pct_value = freq_false_count/len_false
                woe_value = (0 if (false_vert_pct_value==0 or true_vert_pct_value==0) else (np.log(true_vert_pct_value/false_vert_pct_value)))
                iv_value = (true_vert_pct_value - false_vert_pct_value)*woe_value
                iv_sum += iv_value
                odds_value = (0 if freq_true_count==0 else (freq_false_count/freq_true_count))
                
                varname.append(var)
                varvalue.append(value)
                varvalue_count.append(freq_list[value])
                varvalue_count_true.append(freq_true_count)
                varvalue_count_false.append(freq_false_count)
                true_vert_pct.append(round(true_vert_pct_value,4))
                false_vert_pct.append(round(false_vert_pct_value,4))
                woe.append(round(woe_value,4))
                iv.append(round(iv_value,4))
                total_pct.append(round(freq_list[value]/(len_true + len_false),4))
                true_pct.append(round(freq_true_count/freq_list[value],4))
                false_pct.append(round(freq_false_count/freq_list[value],4))
                odds.append(round(odds_value,4))
                
            varname.append(var)
            varvalue.append('SUM')
            varvalue_count.append(len_true + len_false)
            varvalue_count_true.append(len_true)
            varvalue_count_false.append(len_false)
            true_vert_pct.append(1)
            false_vert_pct.append(1)
            woe.append('')
            iv.append(round(iv_sum,4))
            total_pct.append(1)
            true_pct.append(round(len_true/(len_true + len_false),4))
            false_pct.append(round(len_false/(len_true + len_false),4))
            odds.append(round(len_false/len_true,4))
            
            ivtb_varname.append(var)
            ivtb_iv_sum.append(round(iv_sum,4))
            cutoff_list.append(cutoffs)
        
        global df_cutoffs
        df_cutoffs = pd.DataFrame({'变量名':ivtb_varname,'cutoffs':cutoff_list})
        df_woe_out = pd.DataFrame({'标签':varname,'取值区间':varvalue,'计数':varvalue_count,'占比':total_pct,
                                   'T计数':varvalue_count_true,'T纵比':true_vert_pct,
                                   'F计数':varvalue_count_false,'F纵比':false_vert_pct,
                                   'WoE':woe,'IV值':iv,'T横比':true_pct,'F横比':false_pct,'ODDs':odds})
        df_iv_out = pd.DataFrame({'变量名':ivtb_varname,'IV值':ivtb_iv_sum})
        df_iv_out = df_iv_out.sort_values(by = ['IV值'], ascending = False)
        
        df_woe_out = df_woe_out[['标签','取值区间','计数','占比','T计数','T纵比','F计数','F纵比',
                                 'WoE','IV值','T横比','F横比','ODDs']]
        df_iv_out = df_iv_out[['变量名', 'IV值']]
        df_woe_out.to_excel(writer, sheet_name = label_name+'_WoE')
        df_iv_out.to_excel(writer, sheet_name = label_name+'_IV')
        
        df_cutoffs= df_cutoffs.set_index('变量名')
        
        print('Label ' + str(label) + ' complete!')
        print('Time: ' + str(datetime.now() - startTime))
    writer.save()
#####################################################################################################



###########################    全部入模变量结果输出 WOE IV表    ####################################
### 注意!!! : 由于df_cutoffs用的是autobin代码中的变量，容易出错
### 调整df_cutoffs时用以下代码： 
    # 先将变量名转换为index
    # df_cutoffs_fine.loc['Var','2_cutoffs'] = np.array([0,1,2,3,5])
    # 然后用finebin_single_var看切点分布，然后再调
    # finebin_single_var(df_all, df_cutoffs_fine, 'Var', label_list, mv_list=['unknown'])
    
### 输入：
    # df: 大宽表
    # df_cutoffs：为autobin中跑出的 df_cutoffs
    # var_list: x 变量列表
    # label_list: y 变量列表
    # outfile: 输出文件名
	# label_1: label为1时代表的好坏，默认为'bad'
    # mv_list: 额外需要认定为空值的字符列表
### 输出：
    # finebin woe报告，保存到本地outfile
    
def finebin_all_report(df, df_cutoffs, var_list, label_list, outfile, label_1='bad', mv_list=[]):
    writer = pd.ExcelWriter(outfile)
    
    for label in label_list:
        label_name = label
        
        if label_1 != 'bad':
            df['bad'] = df[label].apply(lambda e: 1-e)
            label = 'bad'
        
        len_true = len(df[df[label] == 1])
        len_false = len(df[df[label] == 0])
        
        df_woe_out = pd.DataFrame({})
        varname = []
        varvalue = []
        varvalue_count = []
        varvalue_count_true = []
        varvalue_count_false = []
        true_vert_pct = []
        false_vert_pct = []
        woe = []
        iv = []
        total_pct = []
        true_pct = []
        false_pct = []
        odds = []
        
        ivtb_varname = []
        ivtb_iv_sum = []
        
        for var in var_list:
            print('Finebin: ' + label + ' ' + var)
            freq_table = pd.crosstab(df[var],df[label],dropna = False)
            if len(freq_table)==0:
                continue
            
            freq_list = freq_table[0]+freq_table[1]
            
            cutoffs = df_cutoffs.loc[var,'2_cutoffs'].tolist()
            group_var = var+'_group'
            df[group_var] = df[var].apply(lambda x: value2group(x, cutoffs, mv = mv_list))
            freq_table = pd.crosstab(df[group_var],df[label],dropna = False)
            freq_list = freq_table[0]+freq_table[1]
            freq_true_list  = freq_table[1]
            freq_false_list = freq_table[0]
    
            iv_sum = 0
    
            for value in freq_list.index:
                freq_true_count = freq_true_list[value]
                freq_false_count = freq_false_list[value] 
                
                true_vert_pct_value = freq_true_count/len_true
                false_vert_pct_value = freq_false_count/len_false
                woe_value = (0 if (false_vert_pct_value==0 or true_vert_pct_value==0) else (np.log(true_vert_pct_value/false_vert_pct_value)))
                iv_value = (true_vert_pct_value - false_vert_pct_value)*woe_value
                iv_sum += iv_value
                odds_value = (0 if freq_true_count==0 else (freq_false_count/freq_true_count))
                
                varname.append(var)
                varvalue.append(value)
                varvalue_count.append(freq_list[value])
                varvalue_count_true.append(freq_true_count)
                varvalue_count_false.append(freq_false_count)
                true_vert_pct.append(round(true_vert_pct_value, 4))
                false_vert_pct.append(round(false_vert_pct_value, 4))
                woe.append(round(woe_value, 4))
                iv.append(round(iv_value, 4))
                total_pct.append(round(freq_list[value]/(len_true + len_false), 4))
                true_pct.append(round(freq_true_count/freq_list[value], 4))
                false_pct.append(round(freq_false_count/freq_list[value],4))
                odds.append(round(odds_value, 4))
                
            varname.append(var)
            varvalue.append('SUM')
            varvalue_count.append(len_true + len_false)
            varvalue_count_true.append(len_true)
            varvalue_count_false.append(len_false)
            true_vert_pct.append(1)
            false_vert_pct.append(1)
            woe.append('')
            iv.append(round(iv_sum, 4))
            total_pct.append(1)
            true_pct.append(round(len_true/(len_true + len_false), 4))
            false_pct.append(round(len_false/(len_true + len_false),4))
            odds.append(round(len_false/len_true, 4))
            
            ivtb_varname.append(var)
            ivtb_iv_sum.append(round(iv_sum, 4))

        df_woe_out = pd.DataFrame({'标签':varname,'取值区间':varvalue,'计数':varvalue_count,
                                   '占比':total_pct,'T计数':varvalue_count_true,'T纵比':true_vert_pct,
                                   'F计数':varvalue_count_false,'F纵比':false_vert_pct,'WoE':woe,
                                   'IV值':iv,'T横比':true_pct,'F横比':false_pct,'ODDs':odds})
        df_iv_out = pd.DataFrame({'变量名':ivtb_varname,'IV值':ivtb_iv_sum})
        
        df_woe_out = df_woe_out[['标签','取值区间','计数','占比','T计数','T纵比','F计数','F纵比',
                                 'WoE','IV值','T横比','F横比','ODDs']]
        df_iv_out = df_iv_out[['变量名', 'IV值']]
        df_woe_out.to_excel(writer, sheet_name = label_name+'_WoE')
        df_iv_out.to_excel(writer, sheet_name = label_name+'_IV')
        
        print('Label ' + str(label) + ' complete!')
        print('Time: ' + str(datetime.now() - startTime))
    writer.save()
###################################################################################################



###########################    单变量分箱    ##############################################
### 该函数可用于建模时单变量调分箱
### 也可用于看评分分布
        
### 输入：
    # df: 要分析的数据表
    # cutoffs：切分点列表
    # var: 要分箱的变量
    # label: 坏标签列名
	# label_1: label为1时代表的好坏，默认为'bad'
    # mv_list：字符型缺失值列表，可自定义，默认为空
	# pic_file: 分布图保存路径文件名，默认为None
	# title_name: 输出分布图标题，默认为“Histogram”
### 输出：
    # WOE分箱表
    # console中输出 分布图
    # 自定义是否保存分布图到本地

def finebin_single_var(df, cutoffs, var, label, label_1 = 'bad', weight = None, mv_list=[], pic_file=None, title_name='Histogram'):
    from matplotlib import pyplot as plt
    
    if label_1 != 'bad':
        df['bad'] = df[label].apply(lambda e: 1-e)
        label = 'bad'
    if weight == None:
        df['weight']=1
        weight = 'weight'

    len_true = df[df[label] == 1][weight].sum()
    len_false = df[df[label] == 0][weight].sum()

    
    df_woe_out = pd.DataFrame({})
    varname = []
    varvalue = []
    varvalue_count = []
    varvalue_count_true = []
    varvalue_count_false = []
    true_vert_pct = []
    false_vert_pct = []
    woe = []
    iv = []
    total_pct = []
    true_pct = []
    false_pct = []
    odds = []

    freq_table = pd.pivot_table(df,index=[var],columns=[label],values=[weight],aggfunc=[np.sum],fill_value=0)
    freq_list = freq_table.sum(axis=1)

    group_var = var+'_group'
    df[group_var] = df[var].apply(lambda x: value2group(x, cutoffs, mv = mv_list))
    freq_table = pd.pivot_table(df,index=[group_var],columns=[label],values=[weight],aggfunc=[np.sum],fill_value=0)
    freq_list = freq_table.sum(axis=1)
    freq_true_list  = freq_table.iloc[:,1]
    freq_false_list = freq_table.iloc[:,0]

    iv_sum = 0

    for value in freq_list.index:
        freq_true_count = freq_true_list[value]
        freq_false_count = freq_false_list[value] 
        
        true_vert_pct_value = freq_true_count/len_true
        false_vert_pct_value = freq_false_count/len_false
        woe_value = (0 if (false_vert_pct_value==0 or true_vert_pct_value==0) else (np.log(true_vert_pct_value/false_vert_pct_value)))
        iv_value = (true_vert_pct_value - false_vert_pct_value)*woe_value
        iv_sum += iv_value
        odds_value = (0 if freq_true_count==0 else (freq_false_count/freq_true_count))
        
        varname.append(var)
        if type(value)==str:
            varvalue.append(value)
        else:
            varvalue.append(int(value))
        varvalue_count.append(freq_list[value])
        varvalue_count_true.append(freq_true_count)
        varvalue_count_false.append(freq_false_count)
        true_vert_pct.append(round(true_vert_pct_value, 4))
        false_vert_pct.append(round(false_vert_pct_value, 4))
        woe.append(round(woe_value, 4))
        iv.append(round(iv_value, 4))
        total_pct.append(round(freq_list[value]/(len_true+len_false), 4))
        true_pct.append(round(freq_true_count/freq_list[value], 4))
        false_pct.append(round(freq_false_count/freq_list[value], 4))
        odds.append(round(odds_value, 4))
        
    varname.append(var)
    varvalue.append('SUM')
    varvalue_count.append(len_true+len_false)
    varvalue_count_true.append(len_true)
    varvalue_count_false.append(len_false)
    true_vert_pct.append(1)
    false_vert_pct.append(1)
    woe.append('')
    iv.append(round(iv_sum, 4))
    total_pct.append(1)
    true_pct.append(round(len_true/(len_true+len_false), 4))
    false_pct.append(round(len_false/(len_true+len_false), 4))
    odds.append(round(len_false/len_true, 4))
    
    df_woe_out = pd.DataFrame({'标签':varname,'取值区间':varvalue,'计数':varvalue_count,'占比':total_pct,
                               'T计数':varvalue_count_true,'T纵比':true_vert_pct,
                               'F计数':varvalue_count_false,'F纵比':false_vert_pct,
                               'WoE':woe,'IV值':iv,'T横比':true_pct, 'F横比':false_pct,'ODDs':odds})
    
    fig = plt.figure()
    plt.title(title_name)
    plt.bar(np.arange(len(df_woe_out)), df_woe_out['T横比'].tolist(), tick_label=df_woe_out['取值区间'].tolist())
    plt.show()
    if pic_file!=None:
        fig.savefig(pic_file, dpi=fig.dpi)
    
    df_woe_out = df_woe_out[['标签','取值区间','计数','占比','T计数','T纵比','F计数','F纵比',
                             'WoE','IV值','T横比','F横比','ODDs']]
    return df_woe_out
##################################################################################################



############ 评分效果分析 包含finebin WOE report & KS ROC LIFT ####################################
### 输入：
    # outfile_path： 输出路径，英文
    # score_file_name： 评分excel文件名，需包含评分和label
    # df_all: 带评分和label的表格
    # score_title: 评分标题，用于保存文件名 及 图片标题，最好不要有中文
    # naValue： 评分中未查得用的字符
    # label_col： 标签列名
    # score_col： 评分列名
    # label1：label为1时的含义，默认为'bad'
    # scorecut_step: 分布图的切分步长，默认为10
### 输出：
    # 评分WOE表： score_title + '_WOE_KS' + str(KS_value) + '_report.xlsx'
    # 坏比例分布图：score_title+"_badpct_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".png"
    # 评分KS分布图：score_title+"_KS_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".png"
    # 评分ROC分布图： score_title+"_ROC_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".png"
    # 评分LIFT分布图： score_title+"_LIFT_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".png"

#df_all = pd.read_excel(outfile_path+score_file_name, encoding='utf-8', na_values=naValue, sheetname = 0)
#cutoffs_score = np.arange(df_all[score_col].min(), df_all[score_col].max(), scorecut_step)

def score_value_test(outfile_path, df_all, score_title, score_col, label_col, cutoffs, label1='bad'):
    # 评分WOE表 and bad_pct图
    df_out = finebin_single_var(df_all, cutoffs, score_col, label_col, mv_list=[], 
             pic_file=outfile_path+score_title+"_badpct_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".png",
             title_name=score_title+' Bad Percent')

    # KS ROC Lift
    KS_value = cal_ks(df_all, score_col, label_col, label_1=label1, 
                      KS_pic=outfile_path+score_title+"_KS_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".png", 
                      ROC_pic=outfile_path+score_title+"_ROC_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".png", 
                      LIFT_pic=outfile_path+score_title+"_LIFT_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".png")
    
    writer = pd.ExcelWriter(outfile_path + score_title + '_WOE_KS' + str(KS_value) + '_report_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.xlsx')
    df_out.to_excel(writer, index=False, na_rep = '')
    writer.save()
    return KS_value
####################################################################################################


############################# 数据加解密工具 Class: Encrypt_tool #####################################
'''
### encrypt2csv(self, infile, outfile, encoding='utf-8')
    # 输入：
        # infile : 需要加密的原始文件路径，要求为txt或csv文件
        # outfile: 输出的加密后的文件路径，一般为txt或csv
        # encoding:读取文件的编码格式，默认为utf-8
    # 输出:
        # 保存到outfile的加密好的文件
        
### decrypt2df(self, infile, encoding='utf-8')
    # 输入：
        # infile: 需要解密的原始文件路径，一般为txt或csv文件
        # encoding: 输出文件的编码格式，默认为utf-8
    # 输出:
        # 解密好的DataFrame
    
### 使用举例：
et = Encrypt_tool()
infile_name  = 'D:\\modelone\\label.csv'
outfile_name = 'D:\\modelone\\label_encrypt.csv'
et.encrypt2csv(infile_name, outfile_name)
df_label = et.decrypt2df(outfile_name)
'''

import base64
import getpass
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class Encrypt_tool(object):
    def __init__(self):
        pass
    
    @staticmethod
    def generate_key(password):
        if isinstance(password,str):
            password= password.encode('utf-8')
        salt = b'\xad\xd6&A\xad\xe65\xe3\xe9\x9c\xb7\xd8!\x93Ki'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def make_decrypt(self,cipher_data):
        if isinstance(cipher_data,str):
            cipher_data = cipher_data.encode('utf-8')
        phrase = self.input_phrase()
        token = self.generate_key(phrase)
        cipher_suit = Fernet(token)
        plain_text = cipher_suit.decrypt(cipher_data)
        return plain_text.decode('utf-8')
    
    def make_encrypt(self,data):
        if isinstance(data,str):
            data = data.encode('utf-8')
        phrase = self.input_phrase()
        token = self.generate_key(phrase)
        cipher_suit = Fernet(token)
        cipher_text = cipher_suit.encrypt(data)
        return cipher_text.decode('utf-8')
    
    def input_phrase(self):
        ipt = getpass.getpass('Input phrase: ')
        return ipt.strip()

    def generate_df(self, rs):
        ss = rs.split('\n')
        columns = ss[0].split(',')
        agg = []
        for v in ss[1:-1]:
            agg.append(v.split(','))
        dd = pd.DataFrame(agg)
        dd.columns = columns
        return dd
    
    def decrypt2df(self, infile, encoding='utf-8'):
        with open(infile, encoding=encoding) as f:
            encrypted_dd = f.read()
        decrypted_dd = self.make_decrypt(encrypted_dd)
        df_decrypted = self.generate_df(decrypted_dd)
        return df_decrypted
    
    def encrypt2csv(self, infile, outfile, encoding='utf-8'):
        with open(infile, encoding=encoding) as f:
            raw_data = f.read()
        raw_data_encrypted = self.make_encrypt(raw_data)
        with open(outfile, 'w') as txt_file:
            txt_file.write(raw_data_encrypted)
#####################################################################################################


################################# round_up 精确十进制四舍五入 #######################################
'''
### 输入：
    value: 带小数的数值
### 输出：
    精确四舍五入后的整数
'''
from decimal import Decimal, ROUND_HALF_UP

def round_up(value, dec=0):
    if np.isnan(value):
        return np.nan
    else:
        multiplier = 10**dec
        value_dec = Decimal(str(value)) * multiplier
        return int(value_dec.quantize(Decimal('0'), rounding=ROUND_HALF_UP))/multiplier
######################################################################################################

################################# drop_nan_row 删除全部为0或空的行 ###################################
'''
### 输入：
    df: DataFrame
    sn: 序号列列名
### 输出：
    删除了全部为0或空的行后的DataFrame
'''

def drop_nan_row(df, sn):
    df['sum'] = df.sum(axis=1)
    df['sum'] = df['sum'] - df[sn]
    df = df[df['sum']!=0]
    df = df.drop('sum', axis=1)
    return df
#######################################################################################################

################################# drop_empty_row 删除自定义全部为空的行 ##################################
'''
这个函数相比上一个特别耗时
### 输入：
    df: DataFrame
    except_col: list，排除不检查是否为空的列名的列表，默认为:['序号']
    list_empty_type：list，自定义认为空的值，全部转为字符串，默认为:['0', '', 'nan', 'null']
### 输出：
    删除了自定义全部为空的行后的DataFrame
'''
def drop_empty_row(df, except_col=['序号'], list_empty_type=['0', '', 'nan', 'null']):
    df_ori = df.copy()
    df_ori = df_ori.reset_index()
    
    df_check = df.copy()
    df_check = df_check.drop(except_col,axis=1)
    
    for i in range(len(df_check)):
        check_tag = 1
        list_var = list(df_check)
        for var in list_var:
            if str(df_check.loc[i,var]) not in list_empty_type:
                check_tag = 0
        if check_tag == 1:
            df_check = df_check.drop(i,axis=0)
    
    df_check = df_check.reset_index()
    df_check = df_check[['index']]
    df_out = pd.merge(df_check, df_ori, how='inner', on='index')
    df_out = df_out.drop('index',axis=1)
    return df_out
#######################################################################################################

############################################ 伪随机生成器 #############################################
'''
生成随机因子
### 输入：
    df: dataframe
    id_name: SHA256或MD5加密格式的身份证号
    mob_name: SHA256或MD5加密格式的手机号
### 输出：
    df,包含一列随机因子‘random_factor’,随机因子基于身份证号与手机号生成，不会发生变化
'''
def rand_factor(df, id_name, mob_name):
    df_temp = df.copy()
    df_temp['random_factor'] = df_temp[id_name].apply(lambda x:0 if str(x)=='nan' else int(x[-15:],16) if len(x)>=15 else 0) + df_temp[mob_name].apply(lambda x:0 if str(x)=='nan' else int(x[-15:],16) if len(x)>=15 else 0)
    df_temp['random_factor'] = df_temp['random_factor'].apply(lambda x: 0 if x==0 else int(str(x)[-10:])/(10**10))
    return df_temp

'''
生成范围内的随机整数
### 输入：
    df: dataframe
    start：随机整数的开始值
    stop: 随机整数的结束值
    id_name: SHA256或MD5加密格式的身份证号
    mob_name: SHA256或MD5加密格式的手机号
### 输出：
    df,包含一列随机整数‘rend_int’,随机整数基于身份证号与手机号生成，不会发生变化，且均匀分布
'''
def rand_int(df, start, stop, id_name, mob_name):
    df_temp = df.copy()
    df_temp = rand_factor(df_temp, id_name, mob_name)
    df_temp['rend_int'] = df_temp['random_factor'].apply(lambda x: 'ERROR:string' if isinstance(x, str) else round(start+x*(stop-start+1)-0.5,0) if 0.0<x<1.0 else 'ERROR:out of range')
    df_temp = df_temp.drop('random_factor', axis=1)
    return df_temp
#######################################################################################################
