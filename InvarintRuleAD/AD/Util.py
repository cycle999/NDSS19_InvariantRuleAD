'''
Created on 4 Sep 2017

@author: cf1510
'''

import sys
sys.path.append("..")
print(sys.path)

from sklearn.metrics import confusion_matrix
from RuleMiningUtil import MISTree
from RuleMiningUtil import RuleGenerator

def evaluate_prediction(actual_result,predict_result, verbose = 1):
    cmatrix = confusion_matrix(actual_result, predict_result)
    precision = cmatrix[1][1]*1.0/(cmatrix[1][1]+cmatrix[0][1])
    recall = cmatrix[1][1]*1.0/(cmatrix[1][1]+cmatrix[1][0])
    f1score = 2*precision*recall/(precision+recall)    
    accuracy = (cmatrix[1][1]+cmatrix[0][0])*1.0/(cmatrix[1][1]+cmatrix[0][1]+cmatrix[0][0]+cmatrix[1][0])
    FPR = cmatrix[0][1]*1.0/(cmatrix[0][1]+cmatrix[0][0])
    if(verbose == 1):
        print( 'actual 1: ' + str(actual_result.count(1)) +'  0: '+ str(actual_result.count(0)) )
        print( 'predict 1: ' + str(predict_result.count(1)) +'  0: '+ str(predict_result.count(0)) )
        print( 'precision: ' + str(precision) )
        print( 'recall: ' + str(recall) )
        print( 'f1score: ' + str(f1score) )
        print( 'accuracy: ' + str(accuracy) )
        print( 'TPR: ' + str(recall) )
        print( 'FPR: ' + str(FPR) )
    return precision, recall, f1score, accuracy, FPR


def conInvarEntry(target_var, threshold, lessOrGreater, max_dict, min_dict, coefs, active_vars):
    threshold_value = threshold*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var] 
    msg = ''
    msg += target_var + lessOrGreater 
    count = 0
    for i in range(len(coefs)):
        if(coefs[i] > 0):
            msg += ' ' + str(coefs[i]) + '*' + active_vars[i]
            count += 1
    if count > 0:
        msg += '+' 
    msg += str(threshold_value)
    return msg


def conMarginEntry(target_var, threshold, margin):
    # threshold_value = threshold*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var]

    msg = ''
    if margin == 0:
        msg += target_var + '<' + str(threshold)
    else:
        msg += target_var + '>' + str(threshold)

    return msg


def conRangeEntry(target_var, lb, ub):
    # threshold_lb = lb*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var]
    # threshold_up = ub*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var]

    msg = ''
    msg += str(lb) + '<' + target_var + '<' + str(ub)

    return msg



def getRules(training_data, dead_entries, keyArray, mode=0, gamma=0.4, max_k=4, theta=0.1):
    data = training_data.copy()
#     print(len(data))
    'drop dead entries'
    for entry in dead_entries:
        data = data.drop(entry, 1)
    
    for entry in data:
        if mode == 0:
            if 'cluster' in entry:
                data = data.drop(entry, 1)
        elif mode == 1:
            if 'cluster' not in entry:
                data = data.drop(entry, 1)
        
    
    index_dict = {}
    item_dict = {}
    minSup_dict = {}
    index = 100
    for entry in data:
        index_dict[entry] = index
        item_dict[index] = entry
        index += 1
    print (index_dict)
    min_num = len(data)*theta
#     print 'min_num: ' + str(min_num) 
    for entry in data:
        minSup_dict[ index_dict[entry]  ] = max( gamma*len(data[data[entry] == 1]), min_num )
    #     minSup_dict[ index_dict[entry]  ] = 100
#         print entry + ': ' + str(len(data[data[entry] == 1])*1.0)
        data.loc[data[entry]==1, entry] = index_dict[entry]
    df_list = data.values.tolist()
    #print('df_list = ', df_list)
    dataset = []
    for datalist in df_list:
        temptlist = filter(lambda a: a != 0, datalist)
        numbers = list(temptlist)
        dataset.append(numbers)
    #print('dataset = ')
    #print(dataset)
    item_count_dict = MISTree.count_items(dataset)
    root, MIN_freq_item_header_table, MIN, MIN_freq_item_header_dict = MISTree.genMIS_tree(dataset, item_count_dict, minSup_dict)
 
    freq_patterns, support_data = MISTree.CFP_growth(root, MIN_freq_item_header_table, minSup_dict, max_k)

    #print('freq_patterns = ')
    #print(freq_patterns)
    L = RuleGenerator.filterClosedPatterns(freq_patterns, support_data, item_count_dict, max_k, MIN)

    closedL = "../data/freq_patterns/gamma=" + str(gamma) + '&theta=' + str(theta) + ".txt"
    with open(closedL, "w") as myfile:
        #for i in range(1, len(keyArray) + 1):
            #myfile.write('P' + str(i) + ':' + '\n')
        for close_freq in L:
            myfile.write(str(close_freq) + '\n')
            myfile.write('\n')

            myfile.write('--------------------------------------------------------------------------- ' + '\n')
        myfile.close()

    support_file = "../data/support_data/gamma=" + str(gamma) + '&theta=' + str(theta) + ".txt"
    with open(support_file, "w") as myfile:
        #for i in range(1, len(keyArray) + 1):
            # myfile.write('P' + str(i) + ':' + '\n')
        for support_item in support_data:
            myfile.write(str(support_item) + '\n')
            myfile.write('\n')

            myfile.write('--------------------------------------------------------------------------- ' + '\n')
        myfile.close()


    rules = RuleGenerator.generateRules(L, support_data, MIN_freq_item_header_dict, minSup_dict, min_confidence=1)
    
    valid_rules = []
    for rule in rules:
        valid = True
        for i in range(len(keyArray)):
            for key in keyArray[i]:
                belongAnteq = False
                belongConseq = False
                for item in rule[0]:
                    if key in item_dict[item]:
                        belongAnteq = True
                        break
                for item in rule[1]:
                    if key in item_dict[item]:
                        belongConseq = True
                        break
                if belongAnteq == True and belongConseq == True:
                    valid = False
                    break
        if valid == True:
            valid_rules.append(rule)
    
    return valid_rules, item_dict