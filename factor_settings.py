# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:01:48 2021

@author: Phoenix
"""
import numpy as np
import pandas as pd
rootpath = r'D:\lstm\\'
CLOSE_ = pd.read_csv(rootpath + 'Stock500\\close_adj.csv')
CLOSE_ = CLOSE_.set_index('DATETIME')
changable = CLOSE_.copy()
'''TSRANK,Decaylinear函数的计算较为缓慢'''
#返回n阶差分
def DELTA(delta_series, n):
    d = delta_series - delta_series.shift(n)
    return d

# 对series每个元素取自然对数
def LOG(log_series):
    L = log_series.apply(np.log)
    return L

# 对两个series计算相关系数
def CORR(series1,series2,n):
    result = series1.rolling(n).corr(series2,method='pearson')
    return result

# 排序
def RANK(rank_series):
    return rank_series.rank(0)

#返回n阶滞后项
def DELAY(delay_series,n):
    return delay_series.shift(n)

#对过去n天求和
def SUM(sum_series,n):
    return sum_series.rolling(n).sum()

#对过去n天求平均
def MEAN(mean_series,n):
    return mean_series.rolling(n).mean()

#对过去n天求标准差
def STD(std_series,n):
    return std_series.rolling(n).std()

#大于0的取1，小于等于0的取-1
def SIGN(sign_series):
    changable[sign_series>0] = 1
    changable[sign_series<=0] = -1
    return changable

#返回过去n天最小值
def TSMIN(min_s,n):
    return min_s.rolling(n).min()

#返回过去n天最大值
def TSMAX(max_s,n):
    return max_s.rolling(n).max()

#序列df的末位值在过去n天的顺序排位
def TSRANK(rank_s, n):
    return rank_s.rolling(n).apply(lambda x: get_sort_value(x)/n)

#用于TSRANK函数
def get_sort_value(s):
    return s.rank(method='min',ascending=False)[len(s)-1]

#返回协方差
def COVIANCE(df1, df2, n):
    result = df1.rolling(n).cov(df2)
    return result

#返回绝对值
def ABS(df):
    return df.abs()

#按照公式计算，公式为：Yi+1 =(dfi*m+Yi*(n-m))/n，其中Y表示最终结果
def SMA(df, n, m):
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
        y = df.iloc[0:1]
        for x in range(0, len(df) - 1):
            y = y.append((df.iloc[x] * m / n) + (y.iloc[-1] * (n - m) / n), ignore_index=True)
        y.index = df.index
        return y    

#返回线性回归的回归系数
def REGBETA(series1, series2, n):
    cov = series1.rolling(n).cov(series2)
    std1 = series1.rolling(n).std()
    std2 = series2.rolling(n).std()
    return cov/(std1*std2)

#生成 1~n 的等差序列
def SEQUENCE(n):
    return pd.Series(np.asarray(range(1, n + 1)))

#用于Decaylinear函数
def rolling_decay(df):
    weight = range(1, len(df) + 1)[::-1] / np.array(range(1, len(df) + 1)).sum()
    weight = np.array(weight).reshape(1, -1)
    return np.dot(weight, np.asarray(df))

#对 A 序列计算移动平均加权，其中权重对应 d,d-1,…,1（权重和为 1）
def DECAYLINEAR(df, window):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(value=0, inplace=True)
    return df.rolling(window).apply(rolling_decay, raw=True)

#用于wma函数
def rolling_wma(df):
    """用于wma函数"""
    weight = (SEQUENCE(len(df)) - 1)[::-1] * 0.9 * 2 / (len(df) * (len(df) + 1))
    weight = np.array(weight).reshape(1, -1)
    return np.dot(weight, np.asarray(df))

#计算A前n期样本加权平均值权重为0.9i，(i表示样本距离当前时点的间隔)
def WMA(df, window):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(value=0, inplace=True)
    result = df.rolling(window).apply(rolling_wma, raw=True)
    return result

#计算前 n 期满足条件 condition 的样本个数
def COUNT(condition,window):
        """对df前n项条件求数量，df所有数据置为1，其中condition表示选择条件"""
        changable.iloc[:, :] = 1
        return SUM(changable * condition, window)

#在 A,B 中选择最大的数
def MAX(A,B):
    changable[A>=B]=A
    changable[A<B] = B
    return changable

#在 A,B 中选择最小的数
def MIN(A,B):
    changable[A>B]=B
    changable[A<=B] = A
    return changable

def ts_argmax( df):
        """用于highday函数"""
        return len(df) - np.argmax(df) - 1

#计算 A 前 n 期时间序列中最大值距离当前时点的间隔
def HIGHDAY( df, window=10):
        return df.rolling(window).apply(ts_argmax, raw=True)

def ts_argmin(df):
        """用于lowday函数"""
        return len(df) - np.argmax(df) - 1

#计算 A 前 n 期时间序列中最小值距离当前时点的间隔
def LOWDAY(df, window=10):
        return df.rolling(window).apply(ts_argmin, raw=True)

def SUMIF(df, window=10, condition=True):
        """对df前n项条件求和，其中condition表示选择条件"""
        return SUM(df * condition, window)

#对 A 筛选出符合选择条件 condition 的样本
def FILTER(df, condition=True):
        return df * condition
    
def rolling_prod(na):
        """prod 的辅助函数"""
        return np.prod(na)

#序列 A 过去 n 天累乘
def PROD(df, window=10):
    """序列df过去n天累乘"""
    return df.rolling(window).apply(rolling_prod, raw=True)

