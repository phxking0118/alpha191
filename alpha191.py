# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 14:41:57 2022

@author: Phoenix
"""
import pandas as pd
import numpy as np
from factor_settings import *
from datetime import datetime
print('因子计算开始')
print(datetime.now())
rootpath = r'D:\lstm\\'
#导入因子计算所需变量
CLOSE = pd.read_csv(rootpath + 'Stock500\\close_adj.csv')
CLOSE = CLOSE.set_index('DATETIME')
row = CLOSE.shape[0]
column = CLOSE.shape[1]
OPEN = pd.read_csv(rootpath + 'Stock500\\open_adj.csv')
OPEN = OPEN.set_index('DATETIME')

HIGH = pd.read_csv(rootpath + 'Stock500\\high_adj.csv')
HIGH = HIGH.set_index('DATETIME')

LOW = pd.read_csv(rootpath + 'Stock500\\low_adj.csv')
LOW = LOW.set_index('DATETIME')

VOLUME = pd.read_csv(rootpath + 'Stock500\\volume_adj.csv')
VOLUME = VOLUME.set_index('DATETIME')

VWAP = pd.read_csv(rootpath + 'Stock500\\vwap_adj.csv')
VWAP = VWAP.set_index('DATETIME')
AMOUNT = VOLUME*VWAP
RET = pd.read_csv(rootpath + 'Stock500\\RawRet.csv')
RET = RET.set_index('DATETIME')
#根据公式，初步计算因子计算所需的变量LD、HD、TR、DBM和DTM
LD = DELAY(LOW,1)-LOW
HD = HIGH-DELAY(HIGH,1)
TR = MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
changable = CLOSE.copy()

condition1 = OPEN<=DELAY(OPEN,1)
changable[condition1]=MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))
changable[~condition1]=0
DBM=changable
changable[condition1]=MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))
changable[condition1]=0
DTM= changable

#初始化存储因子值的字典
p = {}
for i in range(191):
    p[i] = CLOSE.copy()

#按公式计算因子值
#%%    
j = 1
print('正在计算第%d个因子。'%j)
p[j]  = -1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
np_1 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_1)
#%%    
j = 2
print('正在计算第%d个因子。'%j)
p[j]  = -1 * DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),1)
np_2 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_2)

#%%    
j = 3
print('正在计算第%d个因子。'%j)
condition1 = CLOSE==DELAY(CLOSE,1)
condition2 = CLOSE>DELAY(CLOSE,1)
changable[condition1] = 0
changable[~condition1] = CLOSE
a = changable.copy()
changable[condition2] = MIN(LOW,DELAY(CLOSE,1))
changable[~condition2] = MAX(HIGH,DELAY(CLOSE,1))
p[j]  = SUM(a-changable,6)
np_3 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_3)
#%%    
j = 4
print('正在计算第%d个因子。'%j)
condition1 = (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))
condition2 = ((SUM(CLOSE,2)/2)<((SUM(CLOSE,8)/8)-STD(CLOSE,8)))
condition3 = ((1<(VOLUME/MEAN(VOLUME,20))) | ((VOLUME/MEAN(VOLUME,20))==1))
changable[condition1] = -1
changable[~condition1 & condition2] = 1
changable[~(condition1 | condition2) & condition3] = 1
changable[~(condition1 | condition2 | condition3)] = -1
p[j]  = changable
np_4 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_4)
#%%    
j = 5
print('正在计算第%d个因子。'%j)
p[j]  = (-1*MAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3))
np_5 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_5)
#%%    
j = 5
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(SIGN(DELTA((((OPEN*0.85)+(HIGH*0.15))),4)))*-1)
np_6 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_6)
#%%    
j = 6
print('正在计算第%d个因子。'%j)
p[j]  = ((RANK(MAX((VWAP-CLOSE),3))+RANK(MIN((VWAP-CLOSE),3)))*RANK(DELTA(VOLUME,3)))
np_7 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_7)
#%%    
j = 7
print('正在计算第%d个因子。'%j)
p[j]  = RANK(DELTA(((((HIGH+LOW)/2)*0.2)+(VWAP*0.8)),4))-1
np_8 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_8)
#%%    
j = 8
print('正在计算第%d个因子。'%j)
p[j]  = SMA((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))*(HIGH-LOW)/VOLUME,7,2)
np_9 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_9)
#%%    
j = 8
print('正在计算第%d个因子。'%j)
condition1 = RET<0
changable[condition1] = STD(RET,20)
changable[~condition1] = CLOSE**2
p[j]  = RANK(MAX(changable,5))
np_10 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_10)
#%%    
j = 11
print('正在计算第%d个因子。'%j)
p[j]  = SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)
np_11 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_11)
#%%    
j = 12
print('正在计算第%d个因子。'%j)
p[j]  = (RANK((OPEN-(SUM(VWAP,10)/10))))-1*(RANK(ABS((CLOSE-VWAP))))
np_12 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_12)
#%%    
j = 13
print('正在计算第%d个因子。'%j)
p[j]  = (((HIGH*LOW)**0.5)-VWAP)
np_13 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_13)
#%%    
j = 14
print('正在计算第%d个因子。'%j)
p[j]  = CLOSE-DELAY(CLOSE,5)
np_14 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_14)
#%%    
j = 15
print('正在计算第%d个因子。'%j)
p[j]  = OPEN/DELAY(CLOSE,1)-1
np_15 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_15)
#%%    
j = 16
print('正在计算第%d个因子。'%j)
p[j]  = (-1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))
np_16 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_16)
#%%    
j = 17
print('正在计算第%d个因子。'%j)
p[j]  = RANK((VWAP-MAX(VWAP,15)))**DELTA(CLOSE,5)
np_17 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_17)
#%%    
j = 18
print('正在计算第%d个因子。'%j)
p[j]  = CLOSE/DELAY(CLOSE,5)
np_18 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_18)

#%%    
j = 20
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
np_20 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_20)
#%%    
j = 21
print('正在计算第%d个因子。'%j)
p[j]  = REGBETA(MEAN(CLOSE,6),SEQUENCE(6),5)
p[j] = p[j].iloc[0:row,0:column]
np_21 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_21)
#%%    
j = 22
print('正在计算第%d个因子。'%j)
p[j]  = SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
np_22 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_22)
#%%    
j = 23
print('正在计算第%d个因子。'%j)
condition1 = CLOSE>DELAY(CLOSE,1)
changable[condition1] = STD(CLOSE,20)
changable[~condition1] = 0
a = changable.copy()
changable[condition1] = 0
changable[~condition1] = STD(CLOSE,20)
p[j]  = SMA(a,20,1)+SMA(changable,20,1)*100
np_23 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_23)
#%%    
j = 24
print('正在计算第%d个因子。'%j)
p[j]  = SMA(CLOSE-DELAY(CLOSE,5),5,1)
np_24 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_24)
#%%    
j = 25
print('正在计算第%d个因子。'%j)
p[j]  = ((-1*RANK((DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9))))))*(1+RANK(SUM(RET,250))))
np_25 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_25)
#%%    
j = 26
print('正在计算第%d个因子。'%j)
p[j]  = ((((SUM(CLOSE,7)/7)-CLOSE))+((CORR(VWAP,DELAY(CLOSE,5),230))))
np_26 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_26)
#%%    
j = 27
print('正在计算第%d个因子。'%j)
p[j]  = WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
np_27 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_27)
#%%    
j = 28
print('正在计算第%d个因子。'%j)
p[j]  = 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
np_28 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_28)
#%%    
j = 29
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
np_29 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_29)
#%%    
j = 31
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
np_31 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_31)
#%%    
j = 32
print('正在计算第%d个因子。'%j)
p[j]  = (-1*SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),3)),3))
np_32 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_32)
#%%    
j = 33
print('正在计算第%d个因子。'%j)
p[j]  = ((((-1*TSMIN(LOW,5))+DELAY(TSMIN(LOW,5),5))*RANK(((SUM(RET,240)-SUM(RET,20))/220)))*TSRANK(VOLUME,5))
np_33 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_33)
#%%    
j = 34
print('正在计算第%d个因子。'%j)
p[j]  = MEAN(CLOSE,12)/CLOSE
np_34 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_34)
#%%    
j = 35
print('正在计算第%d个因子。'%j)
p[j]  = (MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),RANK(DECAYLINEAR(CORR((VOLUME),((OPEN*0.65)+(OPEN*0.35)),17),7)))*-1)
np_35 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_35)
#%%    
j = 36
print('正在计算第%d个因子。'%j)
p[j]  = RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP),6),2))
np_36= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_36)
#%%    
j = 37
print('正在计算第%d个因子。'%j)
p[j]  = (-1*RANK(((SUM(OPEN,5)*SUM(RET,5))-DELAY((SUM(OPEN,5)*SUM(RET,5)),10))))
np_37= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_37)
#%%    
j = 38
print('正在计算第%d个因子。'%j)
condition1 = SUM(HIGH,20)/20<HIGH
changable[condition1] = -1*DELTA(HIGH,2)
changable[~condition1] = 0
p[j]  = changable
np_38 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_38)
#%%    
j = 39
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(DECAYLINEAR(DELTA((CLOSE),2),8))-RANK(DECAYLINEAR(CORR(((VWAP*0.3)+(OPEN*0.7)),SUM(MEAN(VOLUME,180),37),14),12)))*-1
np_39= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_39)
#%%    
j = 40
print('正在计算第%d个因子。'%j)
condition1 = CLOSE>DELAY(CLOSE,1)
changable[condition1] = VOLUME
changable[~condition1] = 0
a = changable.copy()
changable[condition1] = 0
changable[~condition1] = VOLUME
p[j]  = SUM(a,20)/SUM(changable,20)*100
np_40 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_40)
#%%    
j = 41
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(MAX(DELTA((VWAP),3),5))*-1)
np_41= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_41)
#%%    
j = 42
print('正在计算第%d个因子。'%j)
p[j]  = (-1*RANK(STD(HIGH,10)))*CORR(HIGH,VOLUME,10)
np_42= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_42)
#%%    
j = 43
print('正在计算第%d个因子。'%j)
condition1 = CLOSE>DELAY(CLOSE,1)
condition2 = CLOSE<DELAY(CLOSE,1)
changable[condition1] = VOLUME
changable[condition2] = -VOLUME
changable[~(condition1 | condition2)] = 0
p[j]  = changable
np_43 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_43)
#%%    
j = 44
print('正在计算第%d个因子。'%j)
p[j]  = (TSRANK(DECAYLINEAR(CORR(((LOW)),MEAN(VOLUME,10),7),6),4)+TSRANK(DECAYLINEAR(DELTA((VWAP),3),10),15))
np_44= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_44)
#%%    
j = 45
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(DELTA((((CLOSE*0.6)+(OPEN*0.4))),1))*RANK(CORR(VWAP,MEAN(VOLUME,150),15)))
np_45= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_45)
#%%    
j = 46
print('正在计算第%d个因子。'%j)
p[j]  = (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
np_46= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_46)
#%%    
j = 47
print('正在计算第%d个因子。'%j)
p[j]  =SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
np_47= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_47)
#%%    
j = 48
print('正在计算第%d个因子。'%j)
p[j]  =(-1*((RANK(((SIGN((CLOSE-DELAY(CLOSE,1)))+SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2))))+SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3))))))*SUM(VOLUME,5))/SUM(VOLUME,20))
np_48= np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_48)
#%%    
j = 49
print('正在计算第%d个因子。'%j)
condition1 = (HIGH+LOW)>=DELAY(HIGH,1)+DELAY(LOW,1)
changable[condition1] = 0
changable[~condition1] = MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))
p[j]  = changable
np_49 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_49)
#%%    
j = 50
print('正在计算第%d个因子。'%j)
condition1 = (HIGH+LOW)>=DELAY(HIGH,1)+DELAY(LOW,1)
changable[condition1] = 0
changable[~condition1] = MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))
HJ = changable.copy()
changable[condition1] = MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))
changable[~condition1] = 0
DJ = changable.copy()
p[j]  = SUM(DJ,12)/(SUM(DJ,12)+SUM(HJ,12))-SUM(HJ,12)/((SUM(HJ,12)+SUM(DJ,12)))
np_50 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_50)
#%%    
j = 51
print('正在计算第%d个因子。'%j)
p[j]  = SUM(DJ,12)/(SUM(DJ,12)+SUM(HJ,12))
np_51 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_51)
#%%    
j = 52
print('正在计算第%d个因子。'%j)
p[j]  = SUM(MAX(HIGH-DELAY((HIGH+LOW+CLOSE)/3,1),0),26)/SUM(MAX(DELAY((HIGH+LOW+CLOSE)/3,1)-1,0),26)*100
np_52 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_52)
#%%    
j = 53
print('正在计算第%d个因子。'%j)
p[j]= COUNT(CLOSE_>DELAY(CLOSE_,1),12)/12*100
np_53 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_53)
#%%    
j = 54
print('正在计算第%d个因子。'%j)
p[j]= (-1*RANK((STD(ABS(CLOSE-OPEN),1)+(CLOSE-OPEN))+CORR(CLOSE,OPEN,10)))
np_54 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_54)
#%%    
j = 55
print('正在计算第%d个因子。'%j)
condition1 = (16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1))) & (ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)))
condition2 = (ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))) & (ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1)))
changable[condition1]=ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4
changable[~ condition1 & condition2]=ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4
changable[~ (condition1 | condition2)]=ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
changable=SUM(changable,12)
p[j] = changable
np_55 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_55)
#%%
j=56
print('正在计算第%d个因子。'%j)
changable=(RANK((OPEN-TSMIN(OPEN,12)))<RANK((RANK(CORR(SUM(((HIGH +LOW)/2),19),SUM(MEAN(VOLUME,40),19),13))**5)))
p[j] = changable
np_56 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_56)
#%%
j=57
print('正在计算第%d个因子。'%j)
p[j] = SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
np_57 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_57)
#%%
j=58
print('正在计算第%d个因子。'%j)
p[j] = COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
np_58 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_58)
#%%
j=59
print('正在计算第%d个因子。'%j)
changable = CLOSE.copy()
condition1 = CLOSE==DELAY(CLOSE,1)
condition2 =CLOSE>DELAY(CLOSE,1)
changable[condition1]=0
changable[~ condition1]=CLOSE
a=changable.copy()
changable[condition2]=MIN(LOW,DELAY(CLOSE,1))
changable[~condition2]=MAX(HIGH,DELAY(CLOSE,1))
changable=SUM(a-changable,20)
p[j] = changable
np_59 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_59)
#%%
j=60
print('正在计算第%d个因子。'%j)
p[j] = SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)
np_60 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_60)
#%%
j=61
print('正在计算第%d个因子。'%j)
p[j] = (MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80),8)),17)))*-1)
np_61 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_61)
#%%
j=62
print('正在计算第%d个因子。'%j)
p[j] =(-1*CORR(HIGH,RANK(VOLUME),5))
np_62 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_62)
#%%
j=63
print('正在计算第%d个因子。'%j)
p[j] =SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
np_63 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_63)
#%%
j=64
print('正在计算第%d个因子。'%j)
p[j] =(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14)))*-1)
np_64 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_64)
#%%
j=65
print('正在计算第%d个因子。'%j)
p[j] =MEAN(CLOSE,6)/CLOSE
np_65 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_65)
#%%
j=66
print('正在计算第%d个因子。'%j)
p[j] =(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
np_66 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_66)
#%%
j=67
print('正在计算第%d个因子。'%j)
p[j] =SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
np_67 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_67)
#%%
j=68
print('正在计算第%d个因子。'%j)
p[j] =SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
np_68 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_68)
#%%    
j = 69
print('正在计算第%d个因子。'%j)
condition1 = SUM(DTM,20)>SUM(DBM,20)
condition2 = SUM(DTM,20)==SUM(DBM,20)
changable[condition1]=(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)
changable[~condition1 &condition2]=0
changable[~(condition1 |condition2)]=(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)
p[j]  = changable
np_69 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_69)
#%%    
j = 70
print('正在计算第%d个因子。'%j)
p[j]  = STD(AMOUNT,6)
np_70 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_70)
#%%    
j = 71
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
np_71 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_71)
#%%    
j = 72
print('正在计算第%d个因子。'%j)
p[j]  = SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
np_72 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_72)
#%%    
j = 73
print('正在计算第%d个因子。'%j)
p[j]  = ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE),VOLUME,10),16),4),5)-RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3)))*-1)
np_73 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_73)
#%%    
j = 74
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(CORR(SUM(((LOW*0.35)+(VWAP*0.65)),20),SUM(MEAN(VOLUME,40),20),7))+RANK(CORR(RANK(VWAP),RANK(VOLUME),6)))
np_74 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_74)



#%%    
j = 76
print('正在计算第%d个因子。'%j)
p[j]  = STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
np_76 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_76)
#%%    
j = 77
print('正在计算第%d个因子。'%j)
p[j]  = MIN(RANK(DECAYLINEAR(((((HIGH+LOW)/2)+HIGH)-(VWAP+HIGH)),20)),RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),3),6)))
np_77 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_77)
#%%    
j = 78
print('正在计算第%d个因子。'%j)
p[j]  = ((HIGH+LOW+CLOSE)/3-MAX((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
np_78 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_78)
#%%    
j = 79
print('正在计算第%d个因子。'%j)
p[j]  = SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
np_79 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_79)
#%%    
j = 80
print('正在计算第%d个因子。'%j)
p[j]  = (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
np_80 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_80)
#%%    
j = 81
print('正在计算第%d个因子。'%j)
p[j]  = SMA(VOLUME,21,2)
np_81 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_81)
#%%    
j = 82
print('正在计算第%d个因子。'%j)
p[j]  = SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
np_82 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_82)
#%%    
j = 83
print('正在计算第%d个因子。'%j)
p[j]  = (-1*RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),5)))
np_83 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_83)
#%%    
j = 84
print('正在计算第%d个因子。'%j)
condition1 = CLOSE>DELAY(CLOSE,1)
condition2 = CLOSE<DELAY(CLOSE,1)
changable[condition1]=VOLUME
changable[~condition1 &condition2]=-VOLUME
changable[~(condition1 |condition2)]=0
changable=SUM(changable,12)
p[j]  = changable
np_83 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_83)
#%%    
j = 85
print('正在计算第%d个因子。'%j)
p[j]  = (TSRANK((VOLUME/MEAN(VOLUME,20)),20)*TSRANK((-1*DELTA(CLOSE,7)),8))
np_85 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_85)
#%%    
j = 86
print('正在计算第%d个因子。'%j)
condition1 = (0.25<(((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10)))
condition2 = ((((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10))<0)
changable[condition1]=-1
changable[~condition1 & condition2]=1
changable[~(condition1 |condition2)]=((-1*1)*(CLOSE-DELAY(CLOSE,1)))
p[j]  = changable
np_86 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_86)
#%%    
j = 87
print('正在计算第%d个因子。'%j)
p[j]  = ((RANK(DECAYLINEAR(DELTA(VWAP,4),7))+TSRANK(DECAYLINEAR(((((LOW*0.9)+(LOW*0.1))-VWAP)/(OPEN-((HIGH+LOW)/2))),11),7))*-1)
np_87 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_87)
#%%    
j = 88
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
np_88 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_88)
#%%    
j = 89
print('正在计算第%d个因子。'%j)
p[j]  = 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
np_89 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_89)
#%%    
j = 90
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(CORR(RANK(VWAP),RANK(VOLUME),5))*-1)
np_90 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_90)
#%%    
j = 91
print('正在计算第%d个因子。'%j)
p[j]  = ((RANK((CLOSE-MAX(CLOSE,5)))*RANK(CORR((MEAN(VOLUME,40)),LOW,5)))-1)
np_91 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_91)
#%%    
j = 92
print('正在计算第%d个因子。'%j)
p[j]  = (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1)
np_92 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_92)
#%%    
j = 93
print('正在计算第%d个因子。'%j)
condition1 = OPEN>=DELAY(OPEN,1)
changable[condition1]=0
changable[~condition1]=MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))
changable=SUM(changable,20)
p[j]  = changable
np_93 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_93)
#%%    
j = 94
print('正在计算第%d个因子。'%j)
condition1 = CLOSE>DELAY(CLOSE,1)
condition2 = CLOSE<DELAY(CLOSE,1)
changable[condition1]=VOLUME
changable[~condition1 & condition2]=-VOLUME
changable[~(condition1 | condition2)]=0
changable=SUM(changable,30)
p[j]  = changable
np_94 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_94)
#%%    
j = 95
print('正在计算第%d个因子。'%j)
p[j]  = STD(AMOUNT,20)
np_95 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_95)
#%%    
j = 96
print('正在计算第%d个因子。'%j)
p[j]  = SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
np_96 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_96)
#%%    
j = 97
print('正在计算第%d个因子。'%j)
p[j]  = STD(VOLUME,10)
np_97 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_97)
#%%    
j = 98
print('正在计算第%d个因子。'%j)
condition1 = (((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05) | ((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))
changable[condition1]=(-1*(CLOSE-TSMIN(CLOSE,100)))
changable[~condition1]=-1*DELTA(CLOSE,3)
p[j]  = changable
np_98 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_98)
#%%    
j = 99
print('正在计算第%d个因子。'%j)
p[j]  = (-1*RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),5)))
np_99 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_99)
#%%    
j = 100
print('正在计算第%d个因子。'%j)
p[j]  = STD(VOLUME,20)
np_100 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_100)
#%%    
j = 101
print('正在计算第%d个因子。'%j)
p[j]  = RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))
np_101 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_101)
#%%    
j = 102
print('正在计算第%d个因子。'%j)
p[j]  = SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
np_102 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_102)
#%%    
j = 103
print('正在计算第%d个因子。'%j)
p[j]  = ((20-LOWDAY(LOW,20))/20)*100
np_103 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_103)
#%%    
j = 104
print('正在计算第%d个因子。'%j)
p[j]  = (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))))
np_104 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_104)
#%%    
j = 105
print('正在计算第%d个因子。'%j)
p[j]  = (-1*CORR(RANK(OPEN),RANK(VOLUME),10))
np_105 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_105)
#%%    
j = 106
print('正在计算第%d个因子。'%j)
p[j]  = CLOSE-DELAY(CLOSE,20)
np_106 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_106)
#%%    
j = 107
print('正在计算第%d个因子。'%j)
p[j]  = (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1))))
np_107 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_107)
#%%    
j = 108
print('正在计算第%d个因子。'%j)
p[j]  = ((RANK((HIGH-MIN(HIGH,2)))**RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1)
np_108 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_108)
#%%    
j = 109
print('正在计算第%d个因子。'%j)
p[j]  = SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
np_109 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_109)
#%%    
j = 110
print('正在计算第%d个因子。'%j)
p[j]  = SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
np_110 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_110)
#%%    
j = 111
print('正在计算第%d个因子。'%j)
p[j]  = SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
np_111 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_111)
#%%    
j = 112
print('正在计算第%d个因子。'%j)
condition1 = CLOSE>DELAY(CLOSE,1)
condition2 = CLOSE<DELAY(CLOSE,1)
changable[condition1]=CLOSE-DELAY(CLOSE,1)
changable[~condition1]=0
changable=SUM(changable,12)
a=changable.copy()
changable[condition2]=ABS(CLOSE-DELAY(CLOSE,1))
changable[~condition2]=0
changable=SUM(changable,12)
p[j]  = a-changable
np_112 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_112)
#%%    
j = 113
print('正在计算第%d个因子。'%j)
p[j]  = (-1*((RANK((SUM(DELAY(CLOSE,5),20)/20))*CORR(CLOSE,VOLUME,2))*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))))
np_113 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_113)
#%%    
j = 114
print('正在计算第%d个因子。'%j)
p[j]  = ((RANK(DELAY(((HIGH-LOW)/(SUM(CLOSE,5)/5)),2))*RANK(RANK(VOLUME)))/(((HIGH-LOW)/(SUM(CLOSE,5)/5))/(VWAP-CLOSE)))
np_114 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_114)
#%%    
j = 115
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))**RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)))
np_115 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_115)
#%%    
j = 116
print('正在计算第%d个因子。'%j)
p[j]  = REGBETA(CLOSE,SEQUENCE(6),20)
p[j] = p[j].iloc[0:row,0:column]
np_116 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_116)
#%%    
j = 117
print('正在计算第%d个因子。'%j)
p[j]  = ((TSRANK(VOLUME,32)*(1-TSRANK(((CLOSE+HIGH)-LOW),16)))*(1-TSRANK(RET,32)))
np_117 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_117)
#%%    
j = 118
print('正在计算第%d个因子。'%j)
p[j]  = SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
np_118 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_118)
#%%    
j = 119
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
np_119 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_119)
#%%    
j = 120
print('正在计算第%d个因子。'%j)
p[j]  = (RANK((VWAP-CLOSE))/RANK((VWAP+CLOSE)))
np_120 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_120)
#%%    
j = 121
print('正在计算第%d个因子。'%j)
p[j]  = ((RANK((VWAP-MIN(VWAP,12)))**TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3))*-1)
np_121 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_121)
#%%    
j = 122
print('正在计算第%d个因子。'%j)
p[j]  = (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
np_122 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_122)
#%%    
j = 123
print('正在计算第%d个因子。'%j)
p[j]  = RANK(CORR(SUM(((HIGH+LOW)/2),20),SUM(MEAN(VOLUME,60),20),9))
np_123 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_123)
#%%    
j = 124
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)
np_124 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_124)
#%%    
j = 125
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(DECAYLINEAR(CORR((VWAP),MEAN(VOLUME,80),17),20))/RANK(DECAYLINEAR(DELTA(((CLOSE*0.5)+(VWAP*0.5)),3),16)))
np_125 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_125)
#%%    
j = 126
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE+HIGH+LOW)/3
np_126 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_126)
#%%    
j = 127
print('正在计算第%d个因子。'%j)
p[j]  = (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))**2,1))**(1/2)
np_127 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_127)
#%%    
j = 128
print('正在计算第%d个因子。'%j)
'''SUM的期数没给'''
condition1 = SUM((HIGH+LOW+CLOSE)/3,1)>DELAY((HIGH+LOW+CLOSE)/3,1)
changable[condition1]=(HIGH+LOW+CLOSE)/3*VOLUME
changable[~condition1]=0
changable=1+SUM(changable,14)/SUM((HIGH+LOW+CLOSE)/3,14)
p[j]  = 100-100/changable
np_128 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_128)
#%%    
j = 129
print('正在计算第%d个因子。'%j)
condition1 = CLOSE-DELAY(CLOSE,1)<0
changable[condition1]=ABS(CLOSE-DELAY(CLOSE,1))
changable[~condition1]=0
changable=SUM(changable,12)
p[j]  = changable
np_129 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_129)
#%%    
j = 130
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),9),10))/RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3)))
np_130 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_130)
#%%    
j = 131
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(DELAY(VWAP,1))**TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))
np_131 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_131)
#%%    
j = 132
print('正在计算第%d个因子。'%j)
p[j]  = MEAN(AMOUNT,20)
np_132 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_132)
#%%    
j = 133
print('正在计算第%d个因子。'%j)
p[j]  = ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
np_133 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_133)
#%%    
j = 134
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE-DELTA(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
np_134 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_134)
#%%    
j = 135
print('正在计算第%d个因子。'%j)
p[j]  = -SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
np_135 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_135)
#%%    
j = 136
print('正在计算第%d个因子。'%j)
p[j]  = ((-1*RANK(DELTA(RET,3)))*CORR(OPEN,VOLUME,10))
np_136 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_136)
#%%    
j = 137
print('正在计算第%d个因子。'%j)
condition1 = ((CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1))) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))
condition2 = (ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))) & (ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1)))
changable[condition1]=ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4
changable[~condition1 & condition2]=ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4
changable[~(condition1 | condition2)]=ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
p[j]  = 16* changable
np_137 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_137)
#%%    
j = 138
print('正在计算第%d个因子。'%j)
p[j]  = ((RANK(DECAYLINEAR(DELTA((((LOW*0.7)+(VWAP*0.3))),3),20))-TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5),19),16),7))* -1)
np_138 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_138)
#%%    
j = 139
print('正在计算第%d个因子。'%j)
p[j]  = (-1*CORR(OPEN,VOLUME,10))
np_139 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_139)
#%%    
j = 140
print('正在计算第%d个因子。'%j)
p[j]  = MIN(RANK(DECAYLINEAR(((RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE))),8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))
np_140 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_140)
#%%    
j = 141
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))*-1)
np_141 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_141)
#%%    
j = 142
print('正在计算第%d个因子。'%j)
p[j]  = (((-1*RANK(TSRANK(CLOSE,10)))*RANK(DELTA(DELTA(CLOSE,1),1)))*RANK(TSRANK((VOLUME/MEAN(VOLUME,20)),5)))
np_142 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_142)
# #%%    
# j = 143
# print('正在计算第%d个因子。'%j)
# p[j]  = SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
# np_143 = np.array(p[j])
# np.save(rootpath + 'factors_value\\%d.npy'%j, np_143)
#%%    
j = 144
print('正在计算第%d个因子。'%j)
p[j]  = SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
np_144 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_144)
#%%    
j = 145
print('正在计算第%d个因子。'%j)
p[j]  = (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
np_145 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_145)
#%%    
j = 146
print('正在计算第%d个因子。'%j)
p[j]  = MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))**2,60,2)
np_146 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_146)
#%%    
j = 147
print('正在计算第%d个因子。'%j)
'''reg缺少窗口数据'''
p[j]  = REGBETA(MEAN(CLOSE,12),SEQUENCE(12),12)
p[j] = p[j].iloc[0:row,0:column]
np_147 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_147)
#%%    
j = 148
print('正在计算第%d个因子。'%j)
p[j]  = ((RANK(CORR((OPEN),SUM(MEAN(VOLUME,60),9),6))<RANK((OPEN-TSMIN(OPEN,14))))*-1)
np_148 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_148)
#%%    
j = 150
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE+HIGH+LOW)/3*VOLUME
np_150 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_150)
#%%    
j = 151
print('正在计算第%d个因子。'%j)
p[j]  = SMA(CLOSE-DELAY(CLOSE,20),20,1)
np_151 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_151)
#%%    
j = 152
print('正在计算第%d个因子。'%j)
p[j]  = SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY (CLOSE,9),1),9,1),1),26),9,1)
np_152 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_152)
#%%    
j = 153
print('正在计算第%d个因子。'%j)
p[j]  = (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
np_153 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_153)
#%%    
j = 154
print('正在计算第%d个因子。'%j)
p[j]  = (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18)))
np_154 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_154)
#%%    
j = 155
print('正在计算第%d个因子。'%j)
p[j]  = SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
np_155 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_155)
#%%    
j = 156
print('正在计算第%d个因子。'%j)
p[j]  = (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))-1),3)))-1)
np_156 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_156)
#%%    
j = 157
print('正在计算第%d个因子。'%j)
p[j]  = (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5) +TSRANK(DELAY((-1*RET),6),5))
np_157 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_157)
#%%    
j = 158
print('正在计算第%d个因子。'%j)
p[j]  = ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
np_158 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_158)
#%%    
j = 159
print('正在计算第%d个因子。'%j)
p[j]  = ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
np_159 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_159)
#%%    
j = 160
print('正在计算第%d个因子。'%j)
condition = CLOSE<=DELAY(CLOSE,1)
changable[condition] = STD(CLOSE,20)
changable[~condition] = 0
p[j]  = SMA(changable,20,1)
np_160 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_160)
#%%    
j = 161
print('正在计算第%d个因子。'%j)
p[j]  = MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
np_161 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_161)
#%%    
j = 162
print('正在计算第%d个因子。'%j)
p[j]  = (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12, 1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
np_162 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_162)
#%%    
j = 163
print('正在计算第%d个因子。'%j)
p[j]  = RANK(((((-1*RET)*MEAN(VOLUME,20))*VWAP)*(HIGH-CLOSE)))
np_163 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_163)
#%%    
j = 164
print('正在计算第%d个因子。'%j)
condition = CLOSE>DELAY(CLOSE,1)
changable[condition] = 1/(CLOSE-DELAY(CLOSE,1))
changable[~condition] = 1
p[j]  = SMA((changable-MIN(changable,12))/(HIGH-LOW)*100,13,2)
np_164 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_164)

#%%    
j = 166
print('正在计算第%d个因子。'%j)
p[j]  = -20*(20-1)**1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)*(SUM((CLOSE/DELAY(CLOSE,1))**2,20))**1.5)
np_166 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_166)
#%%    
j = 167
print('正在计算第%d个因子。'%j)
condition = CLOSE-DELAY(CLOSE,1)>0
changable[condition] = CLOSE-DELAY(CLOSE,1)
changable[~condition] = 0
p[j]  = SUM(changable,12)
np_167 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_167)
CLOSE-DELAY(CLOSE,1)
#%%    
j = 168
print('正在计算第%d个因子。'%j)
p[j]  = (-1*VOLUME/MEAN(VOLUME,20))
np_168 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_168)
#%%    
j = 169
print('正在计算第%d个因子。'%j)
p[j]  = SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1), 26),10,1)
np_169 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_169)
#%%    
j = 170
print('正在计算第%d个因子。'%j)
p[j]  = ((((RANK((1/CLOSE))*VOLUME)/MEAN(VOLUME,20))*((HIGH*RANK((HIGH-CLOSE)))/(SUM(HIGH,5)/5)))-RANK((VWAP-DELAY(VWAP,5))))
np_170 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_170)
#%%    
j = 171
print('正在计算第%d个因子。'%j)
p[j]  = ((-1*((LOW-CLOSE)*(OPEN**5)))/((CLOSE-HIGH)*(CLOSE**5)))
np_171 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_171)
#%%    
j = 172
print('正在计算第%d个因子。'%j)
condition1 = (LD>0) & (LD>HD)
condition2 = (HD>0) & (HD>LD)
changable[condition1] = LD
changable[~condition1] = 0
LC = changable.copy()
changable[condition2] = HD
changable[~condition2] = 0
HC = changable.copy()
p[j]  = MEAN(ABS(SUM(LC,14)*100/SUM(TR,14)-SUM(HC,14)*100/SUM(TR,14))/(SUM(LC,14)*100/SUM(TR,14)+SUM(HC,14)*100/SUM(TR,14))*100,6)
np_172 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_172)
#%%    
j = 173
print('正在计算第%d个因子。'%j)
p[j]  = 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
np_173 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_173)
#%%    
j = 174
print('正在计算第%d个因子。'%j)
condition = CLOSE>DELAY(CLOSE,1)
changable[condition] = STD(CLOSE,20)
changable[~condition] = 0
p[j]  = SMA(changable,20,1)
np_174 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_174)
#%%    
j = 175
print('正在计算第%d个因子。'%j)
p[j]  = MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
np_175 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_175)
#%%    
j = 176
print('正在计算第%d个因子。'%j)
p[j]  = CORR(RANK(((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12)))),RANK(VOLUME),6)
np_176 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_176)
#%%    
j = 177
print('正在计算第%d个因子。'%j)
p[j]  = ((20-HIGHDAY(HIGH,20))/20)*100
np_177 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_177)
#%%    
j = 178
print('正在计算第%d个因子。'%j)
p[j]  = (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
np_178 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_178)
#%%    
j = 179
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(CORR(VWAP,VOLUME,4))*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12)))
np_179 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_179)
#%%    
j = 180
print('正在计算第%d个因子。'%j)
condition = MEAN(VOLUME,20)<VOLUME
changable[condition] = (-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7))
changable[~condition] = (-1*VOLUME)
p[j]  = changable
np_180 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_180)



#%%    
j = 183
print('正在计算第%d个因子。'%j)
p[j]  = (RANK(CORR(DELAY((OPEN-CLOSE),1),CLOSE,200))+RANK((OPEN-CLOSE)))
np_184 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_184)
#%%    
j = 185
print('正在计算第%d个因子。'%j)
p[j]  = RANK((-1*((1-(OPEN/CLOSE))**2)))
np_185 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_185)
#%%    
j = 186
print('正在计算第%d个因子。'%j)
p[j]  = (MEAN(ABS(SUM(LC,14)*100/SUM(TR,14)-SUM(HC,14)*100/SUM(TR,14))/(SUM(LC,14)*100/SUM(TR,14)+SUM(HC,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM(LC,14)*100/SUM(TR,14)-SUM(HC,14)*100/SUM(TR,14))/(SUM(LC,14)*100/SUM(TR,14)+SUM(HC,14)*100/SUM(TR,14))*100,6),6))/2
np_186 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_186)
#%%    
j = 187
print('正在计算第%d个因子。'%j)
condition = OPEN<=DELAY(OPEN,1)
changable[condition] = 0
changable[~condition] = MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))
p[j]  = SUM(changable,20)
np_187 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_187)
#%%    
j = 188
print('正在计算第%d个因子。'%j)
p[j]  = ((HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
np_188 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_188)
#%%    
j = 189
print('正在计算第%d个因子。'%j)
p[j]  = MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
np_189 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_189)
#%%    
j = 190
print('正在计算第%d个因子。'%j)
p[j]  = LOG((COUNT(CLOSE/DELAY(CLOSE,1)-1>((CLOSE/DELAY(CLOSE,19))**(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE,1)-1-(CLOSE/DELAY(CLOSE,19))**(1/20)-1))**2,20,CLOSE/DELAY(CLOSE,1)-1<(CLOSE/DELAY(CLOSE,19))**(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE,1)-1<(CLOSE/DELAY(CLOSE,19))**(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE,1)-1-((CLOSE/DELAY(CLOSE,19))**(1/20)-1))**2,20,CLOSE/DELAY(CLOSE,1)-1>(CLOSE/DELAY(CLOSE,19))**(1/20)-1))))
np_190 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_190)
#%%    
j = 191
print('正在计算第%d个因子。'%j)
p[j]  = ((CORR(MEAN(VOLUME,20),LOW,5)+((HIGH+LOW)/2))-CLOSE)
np_191 = np.array(p[j])
np.save(rootpath + 'factors_value\\%d.npy'%j, np_191)

print("因子计算结束。")
print(datetime.now())
#%%因子的标准化1
'''如pdf中所说，此cell采用先drop nan后排序的方式，后一cell采取先排序，后在lstm
计算过程中drop nan的方式，两者取其一即可'''
days_span=1500
time_step = 8
lens = 5*time_step - 4
df=pd.read_csv(rootpath+r"Stock500\\RawRet.csv")
df=df.set_index('DATETIME')
#排序
df=df.rank(axis=1,ascending=False)  
#打标签
df[df<=160]=1    
df[df>160]=0 
#因为股票的收益率在两天后体现，向前移动两个单位   
df=df.shift(-2) 
df_matrix=np.array(df)   
#根据一字板对目标输出矩阵进行修正
#导入一字板数据
df_one=pd.read_csv(rootpath + r'Stock500\\yizi.csv')
df_one=df_one.set_index('DATETIME')
#因为股票是在后一天进行交易，向前移动一个单位
df_one=df_one.shift(-1)
#将为1的值柏标为0，其余标为1
df_one[df_one != 1]=2
df_one[df_one == 1]=0
df_one[df_one==2]=1
#得到修正后的目标输出矩阵
aim_matrix=df_matrix*np.array(df_one)
#对目标输出矩阵，按照输入矩阵的方法提取时间序列
aim_time_list = np.ones((days_span,aim_matrix.shape[1]))
for i in range(0,lens,5):
    aim_time_list=aim_matrix[i:days_span+i,:]*aim_time_list
a_ones = np.ones((aim_matrix.shape[0],aim_matrix.shape[1]))
for j in range(time_step+1):
    for i in range(days_span):
        a_ones[i+5*j,:]*=aim_time_list[i,:]
nan_index_ai = np.isnan(a_ones)
a_ones[nan_index_ai] = np.nan
a_ones[~nan_index_ai] = 1

#上面得到nan index，下面再进行排序
for i in range(1,192):
    a = np.load(rootpath + 'factors_value\\%d.npy'%i)
    a = pd.DataFrame(a)
    a = a.replace(np.inf,np.nan)
    a = a.replace(-np.inf,np.nan)
    a = a*a_ones
    a = a.rank(1,method = 'first')
    num = a.max(axis = 1)
    a = a.div(num,axis = 0)
    a = np.array(a)
    np.save(rootpath + 'factors_value_q\\%d.npy'%i,a)
#%%因子的标准化方法2
for i in range(1,192):
    a = np.load(rootpath + 'factors_value\\%d.npy'%i)   
    q = np.argsort(a, axis=1, kind='quicksort', order=None)
    np.save(rootpath + 'factors_value_q\\%d.npy'%i,a)

    
