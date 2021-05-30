'''
Created on 12 Jun 2017

@author: cheng_feng
'''

import pandas as pd
import numpy as np
from sklearn import mixture
from sklearn.linear_model import Lasso
from sklearn import metrics
import Util
import time

'parameters to tune'
ab_overlapping = 1 #same as in the paper
re_overlapping = 1
FP_threshold = 0.01#select actuator
#def_high_threshold = 0.9
#def_low_threshold = 0.1
eps = 0.1
dis_range = 1
sigma = 1.1 #buffer scaler
theta_value = 0.18#same as in the paper
gamma_value = 0.9 #same as in the paper
max_k=4
mode_num_a = 0
mode_num_b = 0

'data preprocessing'
training_data,test_data = [],[]
for i in range(5):
    training_data.append(pd.read_csv("../data/SWaT_Dataset_Normal_Part"+str(i)+".csv"))
    test_data.append(pd.read_csv("../data/SWaT_Dataset_Attack_Part"+str(i)+".csv"))
training_data = pd.concat(training_data)
test_data = pd.concat(test_data)
training_data = training_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


'predicate generation'
start_time_pg = time.time()

#cont_vars = []
training_data = training_data.fillna(method='ffill')
test_data = test_data.fillna(method='ffill')

print('training_data before predicate generation')
print(training_data)
#print('length of training data = ', len(training_data))

#后面有专门的max_dict={}，这里就没有必要了
'''
for entry in training_data:
    #记录max和min可以不限制在传感器，传感器执行器都要求
    if training_data[entry].dtypes == np.float64: #传感器读数入口
        max_value = training_data[entry].max()
        min_value = training_data[entry].min()
        #求传感器读数的update，没有必要：生成传感器读数有变化的entry_update列
        #if max_value != min_value:
        #    training_data[entry + '_update'] = training_data[entry].shift(-1) - training_data[entry]
        #    test_data[entry + '_update'] = test_data[entry].shift(-1) - test_data[entry]
        #    cont_vars.append(entry + '_update')
'''
#因为求了update，所以要把最后一行删掉，没有必要
#training_data = training_data[:len(training_data)-1]
#test_data = test_data[:len(test_data)-1]

anomaly_entries = []
#求高斯分布，删掉entry_update列，留下entry_update_cluster列
#anomaly_entries保存在某一高斯分布的概率小于所有训练值的异常值
'''
for entry in cont_vars:
    print('generate distribution-driven predicates for',entry)
    X = training_data[entry].values
    X = X.reshape(-1, 1)
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 6)
    cluster_num = 0
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            clf = gmm
            cluster_num = n_components
          
    Y = clf.predict(X)
    training_data[entry+'_cluster'] = Y
    cluster_num = len(training_data[entry+'_cluster'].unique() )
    scores = clf.score_samples(X)
    score_threshold = scores.min()*sigma
          
    test_X = test_data[entry].values
    test_X = test_X.reshape(-1, 1)
    test_Y = clf.predict(test_X)
    test_data[entry+'_cluster'] = test_Y
    test_scores = clf.score_samples(test_X)
    test_data.loc[test_scores<score_threshold,entry+'_cluster']=cluster_num
    if len(test_data.loc[test_data[entry+'_cluster']==cluster_num,:])>0:
        anomaly_entry = entry+'_cluster='+str(cluster_num)
        anomaly_entries.append(anomaly_entry)
           
    training_data = training_data.drop(entry,1)
    test_data = test_data.drop(entry,1)
       
'save intermediate result'     
training_data.to_csv("../data/swat_after_distribution_normal.csv", index=False)
test_data.to_csv("../data/swat_after_distribution_attack.csv", index=False)
# training_data = pd.read_csv("../../data/swat_after_distribution_normal.csv")
# test_data = pd.read_csv("../../data/swat_after_distribution_attack.csv")
'''

'derive event driven predicates'
cont_vars = []
disc_vars = []

max_dict = {}
min_dict = {}

onehot_entries = {}
dead_entries = []
FPR_entries = []
MVdf = pd.DataFrame()
#invar_dict = {}
for entry in training_data:
    if training_data[entry].dtypes == np.float64:  # 传感器入口
        max_value = training_data[entry].max()
        min_value = training_data[entry].min()
        if max_value == min_value:  # 全程只有一个值的传感器，考虑要不要保留，把不等于这个值的读数判定为异常；或者考虑这种传感器存不存在（这几句没有运行过，可以不考虑这种传感器存在）
            training_data = training_data.drop(entry, 1)
            test_data = test_data.drop(entry, 1)
        else:
            # print('training_date[', entry, '] before normalization:')
            # print(training_data[entry])
            training_data[entry] = training_data[entry].apply(lambda x: float(x - min_value) / float(max_value - min_value))  # 数据归一化，要做
            # print('training_date[', entry, '] after normalization:')
            # print(training_data[entry])
            cont_vars.append(entry)
            max_dict[entry] = max_value
            min_dict[entry] = min_value
            test_data[entry] = test_data[entry].apply(lambda x: float(x - min_value) / float(max_value - min_value))
    else:  # 执行器入口
        newdf = pd.get_dummies(training_data[entry]).rename(
            columns=lambda x: entry + '=' + str(x))  # 把表头从1,2的变成P101=1，P101=2这种形式，并且满足表头的数据为1，不满足的为0（所以不用再用loc函数赋值了）
        if len(
                newdf.columns.values.tolist()) <= 1:  # 只有一个状态的执行器，表头放进P101=1的Dead_entries中（我觉得没有必要，可以直接把这一列给删除，或者把测试集中这个执行器状态改变了，直接作为异常）
            unique_value = training_data[entry].unique()[0]
            dead_entries.append(entry + '=' + str(unique_value))
            training_data = pd.concat([training_data, newdf], axis=1)
            training_data = training_data.drop(entry, 1)

            for test_value in test_data[entry].unique():
                if test_value != unique_value and len(test_data.loc[test_data[entry] == test_value, :]) / len(
                        test_data) < eps:  # 为什么要加这个异常占比很少（没太懂原因，感觉第二个条件可以删除）
                    anomaly_entries.append(entry + '=' + str(test_value))

            testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
            test_data = pd.concat([test_data, testdf], axis=1)
            test_data = test_data.drop(entry, 1)

        elif len(newdf.columns.values.tolist()) == 2:  # 两个状态的执行器
            # disc_vars.append(entry)
            # training_data[entry + '_shift'] = training_data[entry].shift(-1).fillna(method='ffill').astype(int).astype(str) + '->' + training_data[entry].astype(int).astype(str)

            training_data = pd.concat([training_data, newdf], axis=1)
            training_data = training_data.drop(entry, 1)
            onehot_entries_to_add = newdf.columns.values.tolist()
            num_one = len(training_data.loc[training_data[onehot_entries_to_add[0]] == 1, :])
            num_two = len(training_data.loc[training_data[onehot_entries_to_add[1]] == 1, :])
            per_one = num_one / len(training_data)
            per_two = num_two / len(training_data)
            print('the percentation of ', onehot_entries_to_add[0], ' = ', per_one)
            print('the percentation of ', onehot_entries_to_add[1], ' = ', per_two)
            if (per_one > theta_value and per_two > theta_value) or (per_one < FP_threshold or per_two < FP_threshold):
                disc_vars.append(entry)
                onehot_entries[entry] = onehot_entries_to_add
            else:
                FPR_entries.append(onehot_entries_to_add[0])
                FPR_entries.append(onehot_entries_to_add[1])

            testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
            test_data = pd.concat([test_data, testdf], axis=1)
            test_data = test_data.drop(entry, 1)

        else:  # 三个状态的执行器
            # disc_vars.append(entry)
            # training_data[entry + '_shift'] = training_data[entry].shift(-1).fillna(method='ffill').astype(int).astype(str) + '->' + training_data[entry].astype(int).astype(str)

            training_data[entry + '!=1'] = 1
            training_data.loc[training_data[entry] == 1, entry + '!=1'] = 0
            # training_data.loc[training_data[entry] == 0, entry + '!=1'] = 2

            training_data[entry + '!=2'] = 1
            training_data.loc[training_data[entry] == 2, entry + '!=2'] = 0
            # training_data.loc[training_data[entry] == 0, entry + '!=1'] = 2
            training_data = training_data.drop(entry, 1)

            onehot_entries_to_cal = training_data.columns.values.tolist()[-2:]
            #print('newdf of 3 state acator')
            #print(newdf)
            onehot_entries_to_add = newdf.columns.values.tolist()
            num_one = len(training_data.loc[training_data[onehot_entries_to_cal[0]] == 1, :])
            num_two = len(training_data.loc[training_data[onehot_entries_to_cal[1]] == 1, :])
            per_one = num_one / len(training_data)
            per_two = num_two / len(training_data)
            print('the percentation of ', onehot_entries_to_cal[0], ' = ', per_one)
            print('the percentation of ', onehot_entries_to_cal[1], ' = ', per_two)
            if (per_one > theta_value and per_two > theta_value) or (per_one < FP_threshold or per_two < FP_threshold):
                disc_vars.append(entry)
                onehot_entries[entry] = onehot_entries_to_add  # onehot_entries是字典，没有append
                MVdf = pd.concat([MVdf, newdf], axis=1)
            else:
                FPR_entries.append(onehot_entries_to_cal[0])  # FPR_entries是list，可以append
                FPR_entries.append(onehot_entries_to_cal[1])

            test_data[entry + '!=1'] = 1
            test_data.loc[test_data[entry] == 1, entry + '!=1'] = 0
            # test_data.loc[test_data[entry] == 0, entry + '!=1'] = 2

            test_data[entry + '!=2'] = 1
            test_data.loc[test_data[entry] == 2, entry + '!=2'] = 0
            # test_data.loc[test_data[entry] == 0, entry + '!=2'] = 2
            test_data = test_data.drop(entry, 1)

            '''
            else: #多个状态的执行器，对异常检测有帮助（entry_shift其实也没有用，可以删掉）
                #保存P101，具体的P101=1 OR 2，在onehot_entries中有
                disc_vars.append(entry)
                #for actuator_value in training_data[entry].unique():
                #    disc_vars.append(entry + '=' + str(actuator_value))  # 把对异常判断有帮助的传感器添加到disc_vars中
                #这一行可以不要
                #training_data[entry + '_shift'] = training_data[entry].shift(-1).fillna(method='ffill').astype(int).astype(str) + '->' + training_data[entry].astype(int).astype(str)  # training_data[entry].shift(-1)是相同的索引对应下一个时刻的值，几个字符串相加，代表training_data[MV101_shift]这一列的数据是1->1，或者1->2这样（要注意这样相加的话，是下一个时刻的状态在前面）
                onehot_entries[entry] = newdf.columns.values.tolist()  # onehot_entries是类似MV101=1,MV101=2的入口
                training_data = pd.concat([training_data, newdf], axis=1)  # 这两行代表把MV101这一列删掉，加入MV101=1和MV101=2这两列
                training_data = training_data.drop(entry, 1)

                testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))  # 以下几步把test中的数据MV101换成MV101=1和MV101=2
                test_data = pd.concat([test_data, testdf], axis=1)
                test_data = test_data.drop(entry, 1)
'''
print('FPR_entries')
print(FPR_entries)
del_entries = dead_entries + FPR_entries
print('MVdf')
print(MVdf)
#del_entries = dead_entries.extend(FPR_entries)
#print('del_entries')
#print(del_entries)
#print('training_data after normalization and onehot-coding')
#print(training_data)
#print("cont_vars")
#print(cont_vars)
#print("disc_vars")
#print(disc_vars)
#print("max_dict")
#print(max_dict)
#print("min_dict")
#print(min_dict)
print("onehot_entries")
print(onehot_entries)
#print("dead_entries")
#print(dead_entries)

'''
for entry in onehot_entries:
    print("entry = ", entry)
    for onehot in onehot_entries[entry]:
        print("onehot = ", onehot)
        print("onehot_entries[entry]", onehot_entries[entry])
        print("training_data[onehot] =", training_data[onehot])
        min_value = training_data.loc[training_data[onehot] == 1, 'FIT101'].min()
        print("min_value = ", min_value)
        #max_value = training_data.loc[training_data[onehot] == 1, target_var].max()
'''
invar_dict = {}
for entry in disc_vars:
    #print('generate event-driven predicates for',entry)
    for target_var in cont_vars:
        #print(target_var, ' for ', entry)
        if target_var not in invar_dict:
            invar_dict[target_var] = [0,1]
        min_value = []
        max_value = []
        if len(onehot_entries[entry]) == 2:
            for onehot in onehot_entries[entry]:
                min_value.append(training_data.loc[training_data[onehot] == 1, target_var].min())
                max_value.append(training_data.loc[training_data[onehot] == 1, target_var].max())
            overlap, overlap_range = Util.findOverlapping(min_value, max_value, ab_overlapping, re_overlapping)
            if overlap == False:
                if overlap_range != 0:
                    num_ac_state = []
                    for onehot in onehot_entries[entry]:
                        num_ac_state.append(len(training_data.loc[(training_data[target_var] >= overlap_range[0]) & (training_data[target_var] <= overlap_range[1]) & (training_data[onehot] == 1), :]))
                    if (num_ac_state[0] / len(training_data) < theta_value) & (num_ac_state[1] / len(training_data) < theta_value):
                        invar_dict[target_var] = Util.appendIvar(invar_dict[target_var], min_value, max_value, dis_range)
        else:
            for onehot in onehot_entries[entry]:
                min_value.append(training_data.loc[MVdf[onehot] == 1, target_var].min())
                max_value.append(training_data.loc[MVdf[onehot] == 1, target_var].max())
            min_ls = min_value[-2 : ]
            max_ls = max_value[-2 : ]
            overlap, overlap_range = Util.findOverlapping(min_ls, max_ls, ab_overlapping, re_overlapping)
            if overlap == False:
                if overlap_range != 0:
                    num_ac_state = []
                    for onehot in onehot_entries[entry][-2 : ]:
                        num_ac_state.append(len(training_data.loc[(training_data[target_var] >= overlap_range[0]) & (training_data[target_var] <= overlap_range[1]) & (MVdf[onehot] != 1), :]))
                    if (num_ac_state[0] / len(training_data) < theta_value) & (num_ac_state[1] / len(training_data) < theta_value):
                        invar_dict[target_var] = Util.appendIvar(invar_dict[target_var], min_value, max_value, dis_range)

            '''
            for i in range(len(onehot_entries[entry]) - 1):
                for j in range(i+1, len(onehot_entries[entry])):
                    min_ls = [min_value[i], min_value[j]]
                    max_ls = [max_value[i], max_value[j]]
                    overlap = Util.findOverlapping(min_ls, max_ls, ab_overlapping, re_overlapping)
                    if overlap == False:
                        if overlap_range != 0:
                            num_ac_state = []
                            for onehot in onehot_entries[entry]:
                                num_ac_state.append(len(training_data.loc[(training_data[target_var] >= overlap_range[0]) & (training_data[target_var] <= overlap_range[1]) & (training_data[onehot] == 1), :]))
                            if (num_ac_state[0] / len(training_data) < theta_value) & (num_ac_state[1] / len(training_data) < theta_value):
                                invar_dict[target_var] = Util.appendIvar(invar_dict[target_var], min_ls, max_ls,dis_range)
            '''
'''
            print(len(onehot_entries[entry]))
            if len(onehot_entries[entry]) == 2:
                print("onehot = ", onehot)

                # print(target_var, " when ", onehot, ".min() = ", min_value)
                # if min_value not in invar_dict[target_var]:
                # if (min_value not in invar_dict[target_var]) & (min_value >= def_low_threshold) & (min_value < def_high_threshold):
                # invar_dict[target_var].append(min_value)
                # invar_dict[target_var].append(min_value)
                max_value = training_data.loc[training_data[onehot] == 1, target_var].max()
                # print(entry, " when ", onehot, ".max() = ", max_value)
                # if max_value not in invar_dict[target_var]:
                # if (max_value not in invar_dict[target_var]) & (max_value > def_low_threshold) & (max_value < def_high_threshold):
                # invar_dict[target_var].append(max_value)
                # invar_dict[target_var].append(max_value)
                if max_value - min_value < dis_range:
                    if min_value not in invar_dict[target_var]:
                        invar_dict[target_var].append(min_value)
                    if max_value not in invar_dict[target_var]:
                        invar_dict[target_var].append(max_value)
            #print("onehot = ", onehot)
            min_value = training_data.loc[training_data[onehot] == 1, target_var].min()
            #print(target_var, " when ", onehot, ".min() = ", min_value)
            #if min_value not in invar_dict[target_var]:
            #if (min_value not in invar_dict[target_var]) & (min_value >= def_low_threshold) & (min_value < def_high_threshold):
                #invar_dict[target_var].append(min_value)
            #invar_dict[target_var].append(min_value)
            max_value = training_data.loc[training_data[onehot] == 1, target_var].max()
            #print(entry, " when ", onehot, ".max() = ", max_value)
            #if max_value not in invar_dict[target_var]:
            #if (max_value not in invar_dict[target_var]) & (max_value > def_low_threshold) & (max_value < def_high_threshold):
                #invar_dict[target_var].append(max_value)
            #invar_dict[target_var].append(max_value)
            if max_value - min_value < dis_range:
                if min_value not in invar_dict[target_var]:
                    invar_dict[target_var].append(min_value)
                if max_value not in invar_dict[target_var]:
                    invar_dict[target_var].append(max_value)
'''

#print('invar_dict =', invar_dict)
num_of_invar_entryies = 0
for target_var in invar_dict:
    icpList = invar_dict[target_var]
    if icpList is not None and len(icpList) > 0:
        icpList = list(set(icpList))
        icpList.sort()
        #print('Before deleting adjacent values: invar_dict for', [target_var], ' = ')
        #print(icpList)
        icpList_dif = []

        for i in range(len(icpList)-1):
            icpList_dif.append(icpList[i+1]-icpList[i])
        #print('icpList_dif = ', icpList_dif)
        count = 0
        del_index = []
        for i in range(len(icpList_dif)):
            if icpList_dif[i] < eps:
                if count >= 1:
                    del_index.append(i)
                count = count + 1
            else:
                count = 0
        #print('del_index = ', del_index )
        icpList = [icpList[i] for i in range(len(icpList)) if i not in del_index ]

        print('After deleting adjacent values: invar_dict for', [target_var], ' = ')
        print(icpList)

        for i in range(len(icpList) - 1):
            if (i == 0) & ( (i+1) == len(icpList) - 1):
                break
            invar_entry = Util.conRangeEntry(target_var, icpList[i], icpList[i+1])
            # print('number of ', invar_entry, ' = ', len(training_data.loc[(training_data[target_var] >= icpList[i]) & (training_data[target_var] <= icpList[i + 1]), :] ))
            if len(training_data.loc[(training_data[target_var] >= icpList[i]) & (training_data[target_var] <= icpList[i+1]), :]) / len(training_data) >= theta_value:
                training_data[invar_entry] = 0
                training_data.loc[(training_data[target_var] >= icpList[i]) & (training_data[target_var] <= icpList[i+1]), invar_entry] = 1

                test_data[invar_entry] = 0
                test_data.loc[(test_data[target_var] >= icpList[i]) & (test_data[target_var] <= icpList[i+1]), invar_entry] = 1
                num_of_invar_entryies = num_of_invar_entryies + 1
                # print(invar_entry)
                print('num of data satisfying ', invar_entry, '= ',len(training_data.loc[training_data[invar_entry] == 1, :]))
        '''
        for i in range(len(icpList) - 1):
            for j in range(i + 1, len(icpList)):
                if (i == 0) & (j == len(icpList)-1):
                    break
                else:
                    invar_entry = Util.conRangeEntry(target_var, icpList[i], icpList[j])
                    # print('number of ', invar_entry, ' = ', len(training_data.loc[(training_data[target_var] >= icpList[i]) & (training_data[target_var] <= icpList[i + 1]), :] ))
                    if len(training_data.loc[(training_data[target_var] >= icpList[i]) & (training_data[target_var] <= icpList[j]),:]) / len(training_data) >= theta_value:
                        training_data[invar_entry] = 0
                        training_data.loc[(training_data[target_var] >= icpList[i]) & (training_data[target_var] <= icpList[j]), invar_entry] = 1

                        test_data[invar_entry] = 0
                        test_data.loc[(test_data[target_var] >= icpList[i]) & (test_data[target_var] <= icpList[j]), invar_entry] = 1
                        num_of_invar_entryies = num_of_invar_entryies + 1
                        # print(invar_entry)
                        print('num of data satisfying ', invar_entry, '= ',len(training_data.loc[training_data[invar_entry] == 1, :]))
        '''
for var_c in cont_vars:
    training_data = training_data.drop(var_c,1)
    test_data = test_data.drop(var_c,1)

end_time_pg = time.time()
time_cost = (end_time_pg-start_time_pg)*1.0/60

#print('training_data after predicate generation')
#print(training_data)
print('num_of_invar_entryies = ', num_of_invar_entryies)
print('predicate generation time cost: ' + str(time_cost))

'save intermediate result'
training_data.to_csv("../data/after_event_normal.csv", index=False)
test_data.to_csv("../data/after_event_attack.csv", index=False)
# training_data = pd.read_csv("../data/after_event_normal.csv")
# test_data = pd.read_csv("../data/after_event_attack.csv")

'Rule mining'
keyArray = [['FIT101','LIT101','MV101','P101','P102'], ['AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203','P204','P205','P206'],
         ['DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302'], ['AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401'],
         ['AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503'],['FIT601','P601','P602','P603']]

print('Start rule mining')
#print('def_high_threshold = ', def_high_threshold, ', def_low_threshold = ', def_low_threshold)
start_time_rm = time.time()
rule_list_0, item_dict_0 = Util.getRules(training_data, del_entries, keyArray, mode=mode_num_a, gamma=gamma_value, max_k=max_k, theta=theta_value)
print('finish mode', mode_num_a)
#print('rules_list_0 = ', rule_list_0)
##print('item_dict_0 = ', item_dict_0)
##mode 2 is quite costly, use mode 1 if want to save time
#rule_list_1, item_dict_1 = Util.getRules(training_data, dead_entries, keyArray, mode=mode_num_b, gamma=gamma_value, max_k=max_k, theta=theta_value)
#print('finish mode', mode_num_b)
end_time_rm = time.time()
time_cost = (end_time_rm-start_time_rm)*1.0/60
print('rule mining time cost: ' + str(time_cost))

''' 
rules = []
for rule in rule_list_1:
    valid = False
    for item in rule[0]:
        if 'cluster' in item_dict_1[item]:
            valid = True
            break
    if valid == False:
        for item in rule[1]:
            if 'cluster' in item_dict_1[item]:
                valid = True
                break
    if valid == True:
        rules.append(rule)
rule_list_1 = rules
'''
print('rule count: ' + str(len(rule_list_0)))
'''
print('Removing redundant rules')
start_time_rm = time.time()
num = 0
del_index = []
for rule in rule_list_0:
    #print('rule')
    #print(rule)
    num += 1
    #print('before: ')
    #print(test_data)
    training_data.loc[:, 'antecedent'] = 1
    training_data.loc[:, 'consequent'] = 1
    #print('after: ')
    #print(test_data)
    strPrint = ' '
    #print("rule[0] = ", rule[0])
    for item in rule[0]:
        if item_dict_0[item] in training_data:
            #print('item_dict_0[',item,'] = ',item_dict_0[item])
            training_data.loc[training_data[item_dict_0[item]] == 0, 'antecedent'] = 0
        else:
            training_data.loc[:, 'antecedent'] = 0
        strPrint += str(item_dict_0[item]) + ' '
    strPrint += '-->'
    for item in rule[1]:
        if item_dict_0[item] in training_data:
            training_data.loc[training_data[item_dict_0[item]] == 0, 'consequent'] = 0
        else:
            training_data.loc[:, 'consequent'] = 0
        strPrint += ' ' + str(item_dict_0[item])
    #print('len(antecedent == 0) & (consequent == 1)')
    #print(len(training_data.loc[(training_data['antecedent'] == 0) & (training_data['consequent'] == 1), :] ))
    #print('len(antecedent == 1) & (consequent == 0)')
    #print(len(training_data.loc[(training_data['antecedent'] == 1) & (training_data['consequent'] == 0), :]))
    #print('len(antecedent == 1) & (consequent == 1)')
    #print(len(training_data.loc[(training_data['antecedent'] == 1) & (training_data['consequent'] == 1), :]))
    #print('len(antecedent == 0) & (consequent == 0)')
    #print(len(training_data.loc[(training_data['antecedent'] == 0) & (training_data['consequent'] == 0), :]))
    if (len(training_data.loc[(training_data['antecedent'] == 1) & (training_data['consequent'] == 0), :] ) != 0) or (len(training_data.loc[(training_data['antecedent'] == 0) & (training_data['consequent'] == 1), :]) != 0) :
        del_index.append(rule_list_0.index(rule))

end_time_rm = time.time()
time_cost = (end_time_rm-start_time_rm)*1.0/60
print('reducing rule redundancy time cost: ' + str(time_cost))

print('Gamma=' + str(gamma_value) + ', theta=' + str(theta_value))
print("Eps = ", str(eps))
print('dis_range = ', dis_range)
print('rule count: ' + str(len(rule_list_0)))
print('num of rule to delete:', len(del_index))
rule_list_0 = [rule_list_0[i] for i in range(len(rule_list_0)) if i not in del_index ]
'''

print('Gamma=' + str(gamma_value) + ', theta=' + str(theta_value))
print("Eps = ", str(eps))
print('dis_range = ', dis_range)
print('max_k = ', max_k)
print('re_overlapping = ',re_overlapping)
print('ab_overlapping = ',ab_overlapping)
print('FP_threshold = ',FP_threshold)



' arrange rules according to phase '
phase_dict = {}
for i in range(1, len(keyArray) + 1):
    phase_dict[i] = []

for rule in rule_list_0:
    strPrint = ''
    first = True
    for item in rule[0]:
        strPrint += item_dict_0[item] + ' and '
        if first == True:
            first = False
            for i in range(0, len(keyArray)):
                for key in keyArray[i]:
                    if key in item_dict_0[item]:
                        phase = i + 1
                        break

    strPrint = strPrint[0:len(strPrint) - 4]
    strPrint += '---> '
    for item in rule[1]:
        strPrint += item_dict_0[item] + ' and '
    strPrint = strPrint[0:len(strPrint) - 4]
    phase_dict[phase].append(strPrint)
''' 
for rule in rule_list_1:
    strPrint = ''
    first = True
    for item in rule[0]:
        strPrint += item_dict_1[item] + ' and '
        if first == True:
            first = False
            for i in range(0,6):
                for key in keyArray[i]:
                    if key in item_dict_1[item]:
                        phase = i+1
                        break

    strPrint = strPrint[0:len(strPrint)-4] 
    strPrint += '---> '
    for item in rule[1]:
        strPrint += item_dict_1[item] + ' and '
    strPrint = strPrint[0:len(strPrint)-4]
    phase_dict[phase].append(strPrint)
'''
# print ' print rules'
invariance_file = "../data/invariants/invariants_gamma=" + str(gamma_value) + '&theta=' + str(theta_value) + ".txt"
with open(invariance_file, "w") as myfile:
    for i in range(1, len(keyArray) + 1):
        myfile.write('P' + str(i) + ':' + '\n')

        for rule in phase_dict[i]:
            myfile.write(rule + '\n')
            myfile.write('\n')

        myfile.write('--------------------------------------------------------------------------- ' + '\n')
    myfile.close()

###### use the invariants to do anomaly detection
# print 'start classification'
test_data['result'] = 0
for entry in anomaly_entries:
    test_data.loc[test_data[entry] == 1, 'result'] = 1

test_data['actual_ret'] = 0
test_data.loc[test_data['Normal/Attack'] != 'Normal', 'actual_ret'] = 1
actual_ret = list(test_data['actual_ret'].values)

start_time = time.time()
num = 0
for rule in rule_list_0:
    num += 1
    #print('before: ')
    #print(test_data)
    test_data.loc[:, 'antecedent'] = 1
    test_data.loc[:, 'consequent'] = 1
    #print('after: ')
    #print(test_data)
    strPrint = ' '
    #print("rule[0] = ", rule[0])
    for item in rule[0]:
        if item_dict_0[item] in test_data:
            #print('item_dict_0[',item,'] = ',item_dict_0[item])
            test_data.loc[test_data[item_dict_0[item]] == 0, 'antecedent'] = 0
        else:
            test_data.loc[:, 'antecedent'] = 0
        strPrint += str(item_dict_0[item]) + ' '
    strPrint += '-->'
    for item in rule[1]:
        if item_dict_0[item] in test_data:
            test_data.loc[test_data[item_dict_0[item]] == 0, 'consequent'] = 0
        else:
            test_data.loc[:, 'consequent'] = 0
        strPrint += ' ' + str(item_dict_0[item])
    test_data.loc[(test_data['antecedent'] == 1) & (test_data['consequent'] == 0), 'result'] = 1
    #test_data.loc[(test_data['antecedent'] == 0) & (test_data['consequent'] == 1), 'result'] = 1
    #test_data.loc[(test_data['antecedent'] == 1) & (test_data['consequent'] == 0), 'result'] = 0

'''
for rule in rule_list_1:
    num += 1
    test_data.loc[:,'antecedent'] = 1
    test_data.loc[:,'consequent'] = 1
    strPrint = ' '
    for item in rule[0]:
        if item_dict_1[item] in test_data:
            test_data.loc[test_data[ item_dict_1[item] ]==0,  'antecedent'] = 0
        else:
            test_data.loc[:,  'antecedent'] = 0
        strPrint += str(item_dict_1[item]) + ' '

    strPrint += '-->'

    for item in rule[1]:
        if item_dict_1[item] in test_data:
            test_data.loc[test_data[ item_dict_1[item] ]==0,  'consequent'] = 0
        else:
            test_data.loc[:,  'antecedent'] = 1
        strPrint += ' ' + str(item_dict_1[item]) 
    test_data.loc[(test_data[ 'antecedent' ]==1) & (test_data[ 'consequent' ]==0),  'result'] = 1
'''
end_time = time.time()
time_cost = (end_time - start_time) * 1.0 / 60
print('detection time cost: ' + str(time_cost))
predict_ret = list(test_data['result'].values)

Util.evaluate_prediction(actual_ret, predict_ret, verbose=1)
 
