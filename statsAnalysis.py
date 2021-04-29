import numpy as np
import re
from scipy import stats
import statsmodels as sm
from statsmodels.stats import proportion
import pandas as pd 


data = pd.read_csv("C:\\Users\\talil\\OneDrive\\Documents\\GitHub\\BN-understanding\\processedData_0202_fixed.csv")

headers = ["Block", "Question", "Small - Medium All", "Medium - Large All", "Small - Large All", "Small - Medium Correct", "Medium - Large Correct", "Small - Large Correct"]
#headers2 = ["Cohort", "Question", "Chain - CC", "CC - CE", "Medium - Large All", "Small - Large All", "Small - Medium Correct", "Medium - Large Correct", "Small - Large Correct"]
headers2 = ["Block", "Question", "Small - Medium", "Medium - Large", "Small - Large"]
headers3 = ["Block", "Question", "ReadCPT - Directionality"]
headers4 = ["Block", "Question", "AVG_Small All", "SD_Small All", "AVG_Medium All", "SD_Medium All", "AVG_Large All", "SD_Large All", "Pvalue_Small - Medium All",  "Pvalue_Medium - Large All", "Pvalue_Small - Large All", "AVG_Small Correct", "SD_Small Correct", "AVG_Medium Correct", "SD_Medium Correct", "AVG_ Large Correct", "SD_ Large Correct", "Pvalue_Small - Medium Correct",  "Pvalue_Medium - Large Correct", "Pvalue_Small - Large Correct"]

chain_col = { 1: ['Chain_Q1_Cond1_P_Decreased', 'Chain_Q1_Distance', 'Chain_Q1_P_AnsWithinRange', 'Chain_Q1_ReadCPT', 'Chain_Q1_Cond2_P', 'Chain_Q1_Cond3_P', 'Chain_Q1_Distance_norm'],
              2: ['Chain_Q2_Cond1_P_Decreased', 'Chain_Q2_Distance', 'Chain_Q2_P_AnsWithinRange', 'Chain_Q2_ReadCPT', 'Chain_Q2_Cond2_P', 'Chain_Q2_Cond3_P', 'Chain_Q2_Distance_norm'],
              3: ['Chain_Q3_Cond1_P_Increased', 'Chain_Q3_Distance', 'Chain_Q3_P_AnsWithinRange', 'Chain_Q3_ReadCPT', 'Chain_Q3_Cond2_P', 'Chain_Q3_Cond3_P', 'Chain_Q3_Distance_norm'],
              4: ['Chain_Q4_Cond1_P_Increased', 'Chain_Q4_Distance', 'Chain_Q4_P_AnsWithinRange', 'Chain_Q4_ReadCPT', 'Chain_Q4_Cond2_P', 'Chain_Q4_Cond3_P', 'Chain_Q4_Distance_norm'],
              5: ['Chain_Q5_Cond1_P_The_Same', 'Chain_Q5_Distance', 'Chain_Q5_P_AnsWithinRange', 'Chain_Q5_ReadCPT', 'Chain_Q5_Cond2_P', 'Chain_Q5_Cond3_P', 'Chain_Q5_Distance_norm']}    
                  
CC_col = { 1: ['CC_Q1_Cond1_P_Increased','CC_Q1_Distance', 'CC_Q1_P_AnsWithinRange', 'CC_Q1_ReadCPT', 'CC_Q1_Cond2_P', 'CC_Q1_Cond3_P', 'CC_Q1_Distance_norm'],
              2: ['CC_Q2_Cond1_P_Increased', 'CC_Q2_Distance', 'CC_Q2_P_AnsWithinRange', 'CC_Q2_ReadCPT', 'CC_Q2_Cond2_P', 'CC_Q2_Cond3_P', 'CC_Q2_Distance_norm'],
              3: ['CC_Q3_Cond1_P_Increased', 'CC_Q3_Distance', 'CC_Q3_P_AnsWithinRange', 'CC_Q3_ReadCPT', 'CC_Q3_Cond2_P', 'CC_Q3_Cond3_P', 'CC_Q3_Distance_norm'],
              4: ['CC_Q4_Cond1_P_Decreased', 'CC_Q4_Distance', 'CC_Q4_P_AnsWithinRange', 'CC_Q4_ReadCPT', 'CC_Q4_Cond2_P', 'CC_Q4_Cond3_P', 'CC_Q4_Distance_norm'],
              5: ['CC_Q5_Cond1_P_Increased', 'CC_Q5_Distance', 'CC_Q5_P_AnsWithinRange', 'CC_Q5_ReadCPT', 'CC_Q5_Cond2_P', 'CC_Q5_Cond3_P', 'CC_Q5_Distance_norm'],
              6: ['CC_Q6_Cond1_P_The_Same',  'CC_Q6_Distance', 'CC_Q6_P_AnsWithinRange', 'CC_Q6_ReadCPT', 'CC_Q6_Cond2_P', 'CC_Q6_Cond3_P', 'CC_Q6_Distance_norm'],
              7: ['CC_Q7_Cond1_P_Increased', 'CC_Q7_Distance', 'CC_Q7_P_AnsWithinRange', 'CC_Q7_ReadCPT', 'CC_Q7_Cond2_P', 'CC_Q7_Cond3_P', 'CC_Q7_Distance_norm']}

CE_col = { 1: ['CE_Q1_Cond1_P_The_Same', 'CE_Q1_Distance', 'CE_Q1_P_AnsWithinRange', 'CE_Q1_ReadCPT', 'CE_Q1_Cond2_P', 'CE_Q1_Cond3_P', 'CE_Q1_Distance_norm'],
              2: ['CE_Q2_Cond1_P_Increased', 'CE_Q2_Distance', 'CE_Q2_P_AnsWithinRange', 'CE_Q2_ReadCPT', 'CE_Q2_Cond2_P', 'CE_Q2_Cond3_P', 'CE_Q2_Distance_norm'],
              3: ['CE_Q3_Cond1_P_Increased', 'CE_Q3_Distance', 'CE_Q3_P_AnsWithinRange', 'CE_Q3_ReadCPT', 'CE_Q3_Cond2_P', 'CE_Q3_Cond3_P', 'CE_Q3_Distance_norm'],
              4: ['CE_Q4_Cond1_P_Increased', 'CE_Q4_Distance', 'CE_Q4_P_AnsWithinRange', 'CE_Q4_ReadCPT', 'CE_Q4_Cond2_P', 'CE_Q4_Cond3_P', 'CE_Q4_Distance_norm'],
              5: ['CE_Q5_Cond1_P_Decreased', 'CE_Q5_Distance', 'CE_Q5_P_AnsWithinRange', 'CE_Q5_ReadCPT', 'CE_Q5_Cond2_P', 'CE_Q5_Cond3_P', 'CE_Q5_Distance_norm'],
              6: ['CE_Q6_Cond1_P_Increased', 'CE_Q6_Distance', 'CE_Q6_P_AnsWithinRange', 'CE_Q6_ReadCPT', 'CE_Q6_Cond2_P', 'CE_Q6_Cond3_P', 'CE_Q6_Distance_norm'],
              7: ['CE_Q7_Cond1_P_Increased', 'CE_Q7_Distance', 'CE_Q7_P_AnsWithinRange', 'CE_Q7_ReadCPT', 'CE_Q7_Cond2_P', 'CE_Q7_Cond3_P', 'CE_Q7_Distance_norm']}

"""
Checks for Distance variabilty between cohorts
"""
def DistanceTest(data):

    rows = []

    data_simple = data[data['Cohort_Type']=="simple"]
    data_medium = data[data['Cohort_Type']=="medium"]
    data_hard = data[data['Cohort_Type']=="hard"]

    small_chain_right = pd.Series([])
    medium_chain_right = pd.Series([])
    large_chain_right = pd.Series([])

    small_CC_right = pd.Series([])
    medium_CC_right = pd.Series([])
    large_CC_right = pd.Series([])

    small_CE_right = pd.Series([])
    medium_CE_right = pd.Series([])
    large_CE_right = pd.Series([])

    rows.append(headers)

    for qnum in range(1,6): #adding chain

        data_simple_right = data_simple[data_simple[chain_col[qnum][0]] == True]
        data_medium_right = data_medium[data_medium[chain_col[qnum][0]] == True]
        data_hard_right = data_hard[data_hard[chain_col[qnum][0]] == True]

        small_medium_all = stats.ranksums(data_simple[chain_col[qnum][1]], data_medium[chain_col[qnum][1]]).pvalue
        medium_large_all = stats.ranksums(data_hard[chain_col[qnum][1]], data_medium[chain_col[qnum][1]]).pvalue
        small_large_all = stats.ranksums(data_simple[chain_col[qnum][1]], data_hard[chain_col[qnum][1]]).pvalue

        small_medium_correct = stats.ranksums(data_simple_right[chain_col[qnum][1]], data_medium_right[chain_col[qnum][1]]).pvalue
        medium_large_correct = stats.ranksums(data_hard_right[chain_col[qnum][1]], data_medium_right[chain_col[qnum][1]]).pvalue
        small_large_correct = stats.ranksums(data_simple_right[chain_col[qnum][1]], data_hard_right[chain_col[qnum][1]]).pvalue

        small_chain_right = pd.concat([small_chain_right, data_simple_right[chain_col[qnum][1]]])
        medium_chain_right = pd.concat([medium_chain_right, data_medium_right[chain_col[qnum][1]]])
        large_chain_right = pd.concat([large_chain_right, data_hard_right[chain_col[qnum][1]]])

        rows.append(["chain", qnum, small_medium_all, medium_large_all, small_large_all, small_medium_correct, medium_large_correct, small_large_correct])


##    print(data_simple[chain_col[1][1]])
##    print(data_simple[chain_col[2][1]])
##    print(pd.concat([data_simple[chain_col[1][1]], data_simple[chain_col[2][1]]]))
        
    for qnum in range(1,8): #adding common cause

        data_simple_right = data_simple[data_simple[CC_col[qnum][0]] == True]
        data_medium_right = data_medium[data_medium[CC_col[qnum][0]] == True]
        data_hard_right = data_hard[data_hard[CC_col[qnum][0]] == True]

        small_medium_all = stats.ranksums(data_simple[CC_col[qnum][1]], data_medium[CC_col[qnum][1]]).pvalue
        medium_large_all = stats.ranksums(data_hard[CC_col[qnum][1]], data_medium[CC_col[qnum][1]]).pvalue
        small_large_all = stats.ranksums(data_simple[CC_col[qnum][1]], data_hard[CC_col[qnum][1]]).pvalue

        small_medium_correct = stats.ranksums(data_simple_right[CC_col[qnum][1]], data_medium_right[CC_col[qnum][1]]).pvalue
        medium_large_correct = stats.ranksums(data_hard_right[CC_col[qnum][1]], data_medium_right[CC_col[qnum][1]]).pvalue
        small_large_correct = stats.ranksums(data_simple_right[CC_col[qnum][1]], data_hard_right[CC_col[qnum][1]]).pvalue

        small_CC_right = pd.concat([small_CC_right, data_simple_right[CC_col[qnum][1]]])
        medium_CC_right = pd.concat([medium_CC_right, data_medium_right[CC_col[qnum][1]]])
        large_CC_right = pd.concat([large_CC_right, data_hard_right[CC_col[qnum][1]]])

        rows.append(["CC", qnum, small_medium_all, medium_large_all, small_large_all, small_medium_correct, medium_large_correct, small_large_correct])

    for qnum in range(1,8): #adding common effect

        data_simple_right = data_simple[data_simple[CE_col[qnum][0]] == True]
        data_medium_right = data_medium[data_medium[CE_col[qnum][0]] == True]
        data_hard_right = data_hard[data_hard[CE_col[qnum][0]] == True]

        small_medium_all = stats.ranksums(data_simple[CE_col[qnum][1]], data_medium[CE_col[qnum][1]]).pvalue
        medium_large_all = stats.ranksums(data_hard[CE_col[qnum][1]], data_medium[CE_col[qnum][1]]).pvalue
        small_large_all = stats.ranksums(data_simple[CE_col[qnum][1]], data_hard[CE_col[qnum][1]]).pvalue

        small_medium_correct = stats.ranksums(data_simple_right[CE_col[qnum][1]], data_medium_right[CE_col[qnum][1]]).pvalue
        medium_large_correct = stats.ranksums(data_hard_right[CE_col[qnum][1]], data_medium_right[CE_col[qnum][1]]).pvalue
        small_large_correct = stats.ranksums(data_simple_right[CE_col[qnum][1]], data_hard_right[CE_col[qnum][1]]).pvalue

        small_CE_right = pd.concat([small_CE_right, data_simple_right[CE_col[qnum][1]]])
        medium_CE_right = pd.concat([medium_CE_right, data_medium_right[CE_col[qnum][1]]])
        large_CE_right = pd.concat([large_CE_right, data_hard_right[CE_col[qnum][1]]])

        rows.append(["CE", qnum, small_medium_all, medium_large_all, small_large_all, small_medium_correct, medium_large_correct, small_large_correct])

     #combined
    small_chain = pd.concat([data_simple[chain_col[1][1]], data_simple[chain_col[2][1]], data_simple[chain_col[3][1]], data_simple[chain_col[4][1]], data_simple[chain_col[5][1]]])
    medium_chain = pd.concat([data_medium[chain_col[1][1]], data_medium[chain_col[2][1]], data_medium[chain_col[3][1]], data_medium[chain_col[4][1]], data_medium[chain_col[5][1]]])
    large_chain = pd.concat([data_hard[chain_col[1][1]], data_hard[chain_col[2][1]], data_hard[chain_col[3][1]], data_hard[chain_col[4][1]], data_hard[chain_col[5][1]]])

    small_CC = pd.concat([data_simple[CC_col[1][1]], data_simple[CC_col[2][1]], data_simple[CC_col[3][1]], data_simple[CC_col[4][1]], data_simple[CC_col[5][1]], data_simple[CC_col[6][1]], data_simple[CC_col[7][1]]])
    medium_CC = pd.concat([data_medium[CC_col[1][1]], data_medium[CC_col[2][1]], data_medium[CC_col[3][1]], data_medium[CC_col[4][1]], data_medium[CC_col[5][1]], data_medium[CC_col[6][1]], data_medium[CC_col[7][1]]])
    large_CC = pd.concat([data_hard[CC_col[1][1]], data_hard[CC_col[2][1]], data_hard[CC_col[3][1]], data_hard[CC_col[4][1]], data_hard[CC_col[5][1]], data_hard[CC_col[6][1]], data_hard[CC_col[7][1]]])

    small_CE = pd.concat([data_simple[CE_col[1][1]], data_simple[CE_col[2][1]], data_simple[CE_col[3][1]], data_simple[CE_col[4][1]], data_simple[CE_col[5][1]], data_simple[CE_col[6][1]], data_simple[CE_col[7][1]]])
    medium_CE = pd.concat([data_medium[CE_col[1][1]], data_medium[CE_col[2][1]], data_medium[CE_col[3][1]], data_medium[CE_col[4][1]], data_medium[CE_col[5][1]], data_medium[CE_col[6][1]], data_medium[CE_col[7][1]]])
    large_CE = pd.concat([data_hard[CE_col[1][1]], data_hard[CE_col[2][1]], data_hard[CE_col[3][1]], data_hard[CE_col[4][1]], data_hard[CE_col[5][1]], data_hard[CE_col[6][1]], data_hard[CE_col[7][1]]])


    #chain all
    small_medium_all = stats.ranksums(small_chain, medium_chain).pvalue
    medium_large_all = stats.ranksums(medium_chain, large_chain).pvalue
    small_large_all = stats.ranksums(small_chain, large_chain).pvalue

    rows.append(["chain", "small - large all", small_chain.mean(), small_chain.std(), medium_chain.mean(), medium_chain.std(), large_chain.mean(), large_chain.std(), small_medium_all, medium_large_all, small_large_all])

    #CC all
    small_medium_all = stats.ranksums(small_CC, medium_CC).pvalue
    medium_large_all = stats.ranksums(medium_CC, large_CC).pvalue
    small_large_all = stats.ranksums(small_CC, large_CC).pvalue

    rows.append(["CC", "small - large all", small_CC.mean(), small_CC.std(), medium_CC.mean(), medium_CC.std(), large_CC.mean(), large_CC.std(), small_medium_all, medium_large_all, small_large_all])

    #CE all
    small_medium_all = stats.ranksums(small_CE, medium_CE).pvalue
    medium_large_all = stats.ranksums(medium_CE, large_CE).pvalue
    small_large_all = stats.ranksums(small_CE, large_CE).pvalue

    rows.append(["CE", "small - large all", small_CE.mean(), small_CE.std(), medium_CE.mean(), medium_CE.std(), large_CE.mean(), large_CE.std(), small_medium_all, medium_large_all, small_large_all])

    #small all
    chain_CC_all = stats.ranksums(small_chain, small_CC).pvalue
    CC_CE_all = stats.ranksums(small_CC, small_CE).pvalue
    chain_CE_all = stats.ranksums(small_chain, small_CE).pvalue

    rows.append(["small", "chain - CE all", small_chain.mean(), small_chain.std(), small_CC.mean(), small_CC.std(), small_CE.mean(), small_CE.std(), chain_CC_all, CC_CE_all, chain_CE_all])

    #medium all
    chain_CC_all = stats.ranksums(medium_chain, medium_CC).pvalue
    CC_CE_all = stats.ranksums(medium_CC, medium_CE).pvalue
    chain_CE_all = stats.ranksums(medium_chain, medium_CE).pvalue

    rows.append(["medium", "chain - CE all", medium_chain.mean(), medium_chain.std(), medium_CC.mean(), medium_CC.std(), medium_CE.mean(), medium_CE.std(), chain_CC_all, CC_CE_all, chain_CE_all])

    #large all
    chain_CC_all = stats.ranksums(large_chain, large_CC).pvalue
    CC_CE_all = stats.ranksums(large_CC, large_CE).pvalue
    chain_CE_all = stats.ranksums(large_chain, large_CE).pvalue

    rows.append(["large", "chain - CE all", large_chain.mean(), large_chain.std(), large_CC.mean(), large_CC.std(), large_CE.mean(), large_CE.std(), chain_CC_all, CC_CE_all, chain_CE_all])

    #chain right
    small_medium_right = stats.ranksums(small_chain_right, medium_chain_right).pvalue
    medium_large_right = stats.ranksums(medium_chain_right, large_chain_right).pvalue
    small_large_right = stats.ranksums(small_chain_right, large_chain_right).pvalue

    rows.append(["chain", "small - large right", small_chain_right.mean(), small_chain_right.std(), medium_chain_right.mean(), medium_chain_right.std(), large_chain_right.mean(), large_chain_right.std(), small_medium_right, medium_large_right, small_large_right])

    #CC right
    small_medium_right = stats.ranksums(small_CC_right, medium_CC_right).pvalue
    medium_large_right = stats.ranksums(medium_CC_right, large_CC_right).pvalue
    small_large_right = stats.ranksums(small_CC_right, large_CC_right).pvalue

    rows.append(["CC", "small - large right", small_CC_right.mean(), small_CC_right.std(), medium_CC_right.mean(), medium_CC_right.std(), large_CC_right.mean(), large_CC_right.std(), small_medium_right, medium_large_right, small_large_right])

    #CE right
    small_medium_right = stats.ranksums(small_CE_right, medium_CE_right).pvalue
    medium_large_right = stats.ranksums(medium_CE_right, large_CE_right).pvalue
    small_large_right = stats.ranksums(small_CE_right, large_CE_right).pvalue

    rows.append(["CE", "small - large right", small_CE_right.mean(), small_CE_right.std(), medium_CE_right.mean(), medium_CE_right.std(), large_CE_right.mean(), large_CE_right.std(), small_medium_right, medium_large_right, small_large_right])

    #small right
    chain_CC_right = stats.ranksums(small_chain_right, small_CC_right).pvalue
    CC_CE_right= stats.ranksums(small_CC_right, small_CE_right).pvalue
    chain_CE_right = stats.ranksums(small_chain_right, small_CE_right).pvalue

    rows.append(["small", "chain - CE right", small_chain_right.mean(), small_chain_right.std(), small_CC_right.mean(), small_CC_right.std(), small_CE_right.mean(), small_CE_right.std(), chain_CC_right, CC_CE_right, chain_CE_right])

    #medium right
    chain_CC_right = stats.ranksums(medium_chain_right, medium_CC_right).pvalue
    CC_CE_right = stats.ranksums(medium_CC_right, medium_CE_right).pvalue
    chain_CE_right = stats.ranksums(medium_chain_right, medium_CE_right).pvalue

    rows.append(["medium", "chain - CE right", medium_chain_right.mean(), medium_chain_right.std(), medium_CC_right.mean(), medium_CC_right.std(), medium_CE_right.mean(), medium_CE_right.std(), chain_CC_right, CC_CE_right, chain_CE_right])

    #large right
    chain_CC_right = stats.ranksums(large_chain_right, large_CC_right).pvalue
    CC_CE_right = stats.ranksums(large_CC_right, large_CE_right).pvalue
    chain_CE_right = stats.ranksums(large_chain_right, large_CE_right).pvalue

    rows.append(["large", "chain - CE right", large_chain_right.mean(), large_chain_right.std(), large_CC_right.mean(), large_CC_right.std(), large_CE_right.mean(), large_CE_right.std(), chain_CC_right, CC_CE_right, chain_CE_right])

    print(type(large_chain))
    print(type(large_chain_right))
  
    table = pd.DataFrame(rows)
    table.to_csv("Stats\\Distance.csv", header = False, index = False)


"""
Checks for Directionality variabilty between cohorts
"""
def DirectionalityTest(data):

    rows = []

    data_simple = data[data['Cohort_Type']=="simple"]
    data_medium = data[data['Cohort_Type']=="medium"]
    data_hard = data[data['Cohort_Type']=="hard"]

    rows.append(headers2)

    for qnum in range(1,6): #adding chain

        data_simple_true = data_simple[data_simple[chain_col[qnum][0]] == True]
        data_medium_true = data_medium[data_medium[chain_col[qnum][0]] == True]
        data_hard_true = data_hard[data_hard[chain_col[qnum][0]] == True]
        
        small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
        medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
        small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

        rows.append(["chain", qnum, small_medium, medium_large, small_large])

        
    for qnum in range(1,8): #adding common cause

        data_simple_true = data_simple[data_simple[CC_col[qnum][0]] == True]
        data_medium_true = data_medium[data_medium[CC_col[qnum][0]] == True]
        data_hard_true = data_hard[data_hard[CC_col[qnum][0]] == True]        

        small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
        medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
        small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

        rows.append(["CC", qnum, small_medium, medium_large, small_large])

    for qnum in range(1,8): #adding common effect

        data_simple_true = data_simple[data_simple[CE_col[qnum][0]] == True]
        data_medium_true = data_medium[data_medium[CE_col[qnum][0]] == True]
        data_hard_true = data_hard[data_hard[CE_col[qnum][0]] == True]        

        small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
        medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
        small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

        rows.append(["CE", qnum, small_medium, medium_large, small_large])



    #combined
    small_chain = pd.concat([data_simple[chain_col[1][0]], data_simple[chain_col[2][0]], data_simple[chain_col[3][0]], data_simple[chain_col[4][0]], data_simple[chain_col[5][0]]])
    medium_chain = pd.concat([data_medium[chain_col[1][0]], data_medium[chain_col[2][0]], data_medium[chain_col[3][0]], data_medium[chain_col[4][0]], data_medium[chain_col[5][0]]])
    large_chain = pd.concat([data_hard[chain_col[1][0]], data_hard[chain_col[2][0]], data_hard[chain_col[3][0]], data_hard[chain_col[4][0]], data_hard[chain_col[5][0]]])

    small_CC = pd.concat([data_simple[CC_col[1][0]], data_simple[CC_col[2][0]], data_simple[CC_col[3][0]], data_simple[CC_col[4][0]], data_simple[CC_col[5][0]], data_simple[CC_col[6][0]], data_simple[CC_col[7][0]]])
    medium_CC = pd.concat([data_medium[CC_col[1][0]], data_medium[CC_col[2][0]], data_medium[CC_col[3][0]], data_medium[CC_col[4][0]], data_medium[CC_col[5][0]], data_medium[CC_col[6][0]], data_medium[CC_col[7][0]]])
    large_CC = pd.concat([data_hard[CC_col[1][0]], data_hard[CC_col[2][0]], data_hard[CC_col[3][0]], data_hard[CC_col[4][0]], data_hard[CC_col[5][0]], data_hard[CC_col[6][0]], data_hard[CC_col[7][0]]])

    small_CE = pd.concat([data_simple[CE_col[1][0]], data_simple[CE_col[2][0]], data_simple[CE_col[3][0]], data_simple[CE_col[4][0]], data_simple[CE_col[5][0]], data_simple[CE_col[6][0]], data_simple[CE_col[7][0]]])
    medium_CE = pd.concat([data_medium[CE_col[1][0]], data_medium[CE_col[2][0]], data_medium[CE_col[3][0]], data_medium[CE_col[4][0]], data_medium[CE_col[5][0]], data_medium[CE_col[6][0]], data_medium[CE_col[7][0]]])
    large_CE = pd.concat([data_hard[CE_col[1][0]], data_hard[CE_col[2][0]], data_hard[CE_col[3][0]], data_hard[CE_col[4][0]], data_hard[CE_col[5][0]], data_hard[CE_col[6][0]], data_hard[CE_col[7][0]]])

    small_chain_true = small_chain[small_chain == True]
    medium_chain_true = medium_chain[medium_chain == True]
    large_chain_true = large_chain[large_chain == True]

    small_CC_true = small_CC[small_CC == True]
    medium_CC_true = medium_CC[medium_CC == True]
    large_CC_true = large_CC[large_CC == True]

    small_CE_true = small_CE[small_CE == True]
    medium_CE_true = medium_CE[medium_CE == True]
    large_CE_true = large_CE[large_CE == True]

    #chain
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(medium_chain_true), len(medium_chain)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(medium_chain_true), len(medium_chain)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(large_chain_true), len(large_chain)).pvalue

    rows.append(["chain", "small - large", small_medium, medium_large, small_large])

    #CC
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(medium_CC_true), len(medium_CC)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(medium_CC_true), len(medium_CC)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(large_CC_true), len(large_CC)).pvalue

    rows.append(["CC", "small - large", small_medium, medium_large, small_large])

    #CE
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(medium_CE_true), len(medium_CE)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CE_true), len(large_CE), len(medium_CE_true), len(medium_CE)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["CE", "small - large", small_medium, medium_large, small_large])

    #small
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CC_true), len(small_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(small_CE_true), len(small_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CE_true), len(small_CE)).pvalue

    rows.append(["small", "chain - CE", chain_CC, CC_CE, chain_CE])

    #medium
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CC_true), len(medium_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_CC_true), len(medium_CC), len(medium_CE_true), len(medium_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CE_true), len(medium_CE)).pvalue

    rows.append(["medium", "chain - CE", chain_CC, CC_CE, chain_CE])

    #large
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CC_true), len(large_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(large_CE_true), len(large_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["large", "chain - CE", chain_CC, CC_CE, chain_CE])


    #Cause to effect questions
    small_chain = pd.concat([data_simple[chain_col[1][0]], data_simple[chain_col[2][0]], data_simple[chain_col[3][0]], data_simple[chain_col[4][0]]])
    medium_chain = pd.concat([data_medium[chain_col[1][0]], data_medium[chain_col[2][0]], data_medium[chain_col[3][0]], data_medium[chain_col[4][0]]])
    large_chain = pd.concat([data_hard[chain_col[1][0]], data_hard[chain_col[2][0]], data_hard[chain_col[3][0]], data_hard[chain_col[4][0]]])

    small_CC = pd.concat([data_simple[CC_col[4][0]], data_simple[CC_col[5][0]], data_simple[CC_col[7][0]]])
    medium_CC = pd.concat([data_medium[CC_col[4][0]], data_medium[CC_col[5][0]], data_medium[CC_col[7][0]]])
    large_CC = pd.concat([data_hard[CC_col[4][0]], data_hard[CC_col[5][0]], data_hard[CC_col[7][0]]])

    small_CE = pd.concat([data_simple[CE_col[2][0]], data_simple[CE_col[3][0]], data_simple[CE_col[4][0]]])
    medium_CE = pd.concat([data_medium[CE_col[2][0]], data_medium[CE_col[3][0]], data_medium[CE_col[4][0]]])
    large_CE = pd.concat([data_hard[CE_col[2][0]], data_hard[CE_col[3][0]], data_hard[CE_col[4][0]]])
    
    small_chain_true = small_chain[small_chain == True]
    medium_chain_true = medium_chain[medium_chain == True]
    large_chain_true = large_chain[large_chain == True]

    small_CC_true = small_CC[small_CC == True]
    medium_CC_true = medium_CC[medium_CC == True]
    large_CC_true = large_CC[large_CC == True]

    small_CE_true = small_CE[small_CE == True]
    medium_CE_true = medium_CE[medium_CE == True]
    large_CE_true = large_CE[large_CE == True]

    #chain
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(medium_chain_true), len(medium_chain)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(medium_chain_true), len(medium_chain)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(large_chain_true), len(large_chain)).pvalue

    rows.append(["chain cause to effect", "small - large", small_medium, medium_large, small_large])

    #CC
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(medium_CC_true), len(medium_CC)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(medium_CC_true), len(medium_CC)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(large_CC_true), len(large_CC)).pvalue

    rows.append(["CC  cause to effect", "small - large", small_medium, medium_large, small_large])

    #CE
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(medium_CE_true), len(medium_CE)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CE_true), len(large_CE), len(medium_CE_true), len(medium_CE)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["CE  cause to effect", "small - large", small_medium, medium_large, small_large])

    #small
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CC_true), len(small_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(small_CE_true), len(small_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CE_true), len(small_CE)).pvalue

    rows.append(["small cause to effect", "chain - CE", chain_CC, CC_CE, chain_CE])

    #medium
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CC_true), len(medium_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_CC_true), len(medium_CC), len(medium_CE_true), len(medium_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CE_true), len(medium_CE)).pvalue

    rows.append(["medium cause to effect", "chain - CE", chain_CC, CC_CE, chain_CE])

    #large
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CC_true), len(large_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(large_CE_true), len(large_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["large cause to effect", "chain - CE", chain_CC, CC_CE, chain_CE])

                         

    #Effect to cause questions
    #small_chain = pd.concat([data_simple[chain_col[1][0]], data_simple[chain_col[2][0]], data_simple[chain_col[3][0]], data_simple[chain_col[4][0]]])
    #medium_chain = pd.concat([data_medium[chain_col[1][0]], data_medium[chain_col[2][0]], data_medium[chain_col[3][0]], data_medium[chain_col[4][0]]])
    #large_chain = pd.concat([data_hard[chain_col[1][0]], data_hard[chain_col[2][0]], data_hard[chain_col[3][0]], data_hard[chain_col[4][0]]])

    small_CC = pd.concat([data_simple[CC_col[1][0]], data_simple[CC_col[2][0]], data_simple[CC_col[3][0]]])
    medium_CC = pd.concat([data_medium[CC_col[1][0]], data_medium[CC_col[2][0]], data_medium[CC_col[3][0]]])
    large_CC = pd.concat([data_hard[CC_col[1][0]], data_hard[CC_col[2][0]], data_hard[CC_col[3][0]]])

    small_CE = pd.concat([data_simple[CE_col[5][0]], data_simple[CE_col[6][0]]])
    medium_CE = pd.concat([data_medium[CE_col[5][0]], data_medium[CE_col[6][0]]])
    large_CE = pd.concat([data_hard[CE_col[5][0]], data_hard[CE_col[6][0]]])
    
    small_chain_true = small_chain[small_chain == True]
    medium_chain_true = medium_chain[medium_chain == True]
    large_chain_true = large_chain[large_chain == True]

    small_CC_true = small_CC[small_CC == True]
    medium_CC_true = medium_CC[medium_CC == True]
    large_CC_true = large_CC[large_CC == True]

    small_CE_true = small_CE[small_CE == True]
    medium_CE_true = medium_CE[medium_CE == True]
    large_CE_true = large_CE[large_CE == True]

    #chain
##    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(medium_chain_true), len(medium_chain)).pvalue
##    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(medium_chain_true), len(medium_chain)).pvalue
##    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(large_chain_true), len(large_chain)).pvalue
##
##    rows.append(["chain effect to cause", "small - large", small_medium, medium_large, small_large])
##
    #CC
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(medium_CC_true), len(medium_CC)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(medium_CC_true), len(medium_CC)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(large_CC_true), len(large_CC)).pvalue

    rows.append(["CC  effect to cause", "small - large", small_medium, medium_large, small_large])

    #CE
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(medium_CE_true), len(medium_CE)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CE_true), len(large_CE), len(medium_CE_true), len(medium_CE)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["CE  effect to cause", "small - large", small_medium, medium_large, small_large])

    #small
    #chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CC_true), len(small_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(small_CE_true), len(small_CE)).pvalue
    #chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CE_true), len(small_CE)).pvalue

    rows.append(["small effect to cause", "chain - CE", CC_CE])

    #medium
    #chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CC_true), len(medium_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_CC_true), len(medium_CC), len(medium_CE_true), len(medium_CE)).pvalue
    #chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CE_true), len(medium_CE)).pvalue

    rows.append(["medium effect to cause", "chain - CE", CC_CE])

    #large
    #chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CC_true), len(large_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(large_CE_true), len(large_CE)).pvalue
    #chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["large effect to cause", "chain - CE", CC_CE])


    #Dseparation
    small_chain = pd.concat([data_simple[chain_col[5][0]]])
    medium_chain = pd.concat([data_medium[chain_col[5][0]]])
    large_chain = pd.concat([data_hard[chain_col[5][0]]])

    small_CC = pd.concat([data_simple[CC_col[6][0]]])
    medium_CC = pd.concat([data_medium[CC_col[6][0]]])
    large_CC = pd.concat([data_hard[CC_col[6][0]]])

    small_CE = pd.concat([data_simple[CE_col[1][0]]])
    medium_CE = pd.concat([data_medium[CE_col[1][0]]])
    large_CE = pd.concat([data_hard[CE_col[1][0]]])

    small_chain_true = small_chain[small_chain == True]
    medium_chain_true = medium_chain[medium_chain == True]
    large_chain_true = large_chain[large_chain == True]

    small_CC_true = small_CC[small_CC == True]
    medium_CC_true = medium_CC[medium_CC == True]
    large_CC_true = large_CC[large_CC == True]

    small_CE_true = small_CE[small_CE == True]
    medium_CE_true = medium_CE[medium_CE == True]
    large_CE_true = large_CE[large_CE == True]

    #chain
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(medium_chain_true), len(medium_chain)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(medium_chain_true), len(medium_chain)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(large_chain_true), len(large_chain)).pvalue

    rows.append(["chain Dseparation", "small - large", small_medium, medium_large, small_large])

    #CC
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(medium_CC_true), len(medium_CC)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(medium_CC_true), len(medium_CC)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(large_CC_true), len(large_CC)).pvalue

    rows.append(["CC  Dseparation", "small - large", small_medium, medium_large, small_large])

    #CE
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(medium_CE_true), len(medium_CE)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CE_true), len(large_CE), len(medium_CE_true), len(medium_CE)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["CE  Dseparation", "small - large", small_medium, medium_large, small_large])

    #small
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CC_true), len(small_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(small_CE_true), len(small_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CE_true), len(small_CE)).pvalue

    rows.append(["small Dseparation", "chain - CE", chain_CC, CC_CE, chain_CE])

    #medium
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CC_true), len(medium_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_CC_true), len(medium_CC), len(medium_CE_true), len(medium_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CE_true), len(medium_CE)).pvalue

    rows.append(["medium Dseparation", "chain - CE", chain_CC, CC_CE, chain_CE])

    #large
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CC_true), len(large_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(large_CE_true), len(large_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["large Dseparation", "chain - CE", chain_CC, CC_CE, chain_CE])


    
    table = pd.DataFrame(rows)
    table.to_csv("Stats\\Directionality.csv", header = False, index = False)


"""
Checks for Range variabilty between cohorts
"""
def RangeTest(data):

    rows = []

    data_simple = data[data['Cohort_Type']=="simple"]
    data_medium = data[data['Cohort_Type']=="medium"]
    data_hard = data[data['Cohort_Type']=="hard"]

    rows.append(headers2)

    for qnum in range(1,6): #adding chain

        data_simple_true = data_simple[data_simple[chain_col[qnum][2]] == True]
        data_medium_true = data_medium[data_medium[chain_col[qnum][2]] == True]
        data_hard_true = data_hard[data_hard[chain_col[qnum][2]] == True]
        
        small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
        medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
        small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

        rows.append(["chain", qnum, small_medium, medium_large, small_large])

        
    for qnum in range(1,8): #adding common cause

        data_simple_true = data_simple[data_simple[CC_col[qnum][2]] == True]
        data_medium_true = data_medium[data_medium[CC_col[qnum][2]] == True]
        data_hard_true = data_hard[data_hard[CC_col[qnum][2]] == True]        

        small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
        medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
        small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

        rows.append(["CC", qnum, small_medium, medium_large, small_large])

    for qnum in range(1,8): #adding common effect

        data_simple_true = data_simple[data_simple[CE_col[qnum][2]] == True]
        data_medium_true = data_medium[data_medium[CE_col[qnum][2]] == True]
        data_hard_true = data_hard[data_hard[CE_col[qnum][2]] == True]        

        small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
        medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
        small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

        rows.append(["CE", qnum, small_medium, medium_large, small_large])


  
        #combined
    small_chain = pd.concat([data_simple[chain_col[1][2]], data_simple[chain_col[2][2]], data_simple[chain_col[3][2]], data_simple[chain_col[4][2]], data_simple[chain_col[5][2]]])
    medium_chain = pd.concat([data_medium[chain_col[1][2]], data_medium[chain_col[2][2]], data_medium[chain_col[3][2]], data_medium[chain_col[4][2]], data_medium[chain_col[5][2]]])
    large_chain = pd.concat([data_hard[chain_col[1][2]], data_hard[chain_col[2][2]], data_hard[chain_col[3][2]], data_hard[chain_col[4][2]], data_hard[chain_col[5][2]]])

    small_CC = pd.concat([data_simple[CC_col[1][2]], data_simple[CC_col[2][2]], data_simple[CC_col[3][2]], data_simple[CC_col[4][2]], data_simple[CC_col[5][2]], data_simple[CC_col[6][2]], data_simple[CC_col[7][2]]])
    medium_CC = pd.concat([data_medium[CC_col[1][2]], data_medium[CC_col[2][2]], data_medium[CC_col[3][2]], data_medium[CC_col[4][2]], data_medium[CC_col[5][2]], data_medium[CC_col[6][2]], data_medium[CC_col[7][2]]])
    large_CC = pd.concat([data_hard[CC_col[1][2]], data_hard[CC_col[2][2]], data_hard[CC_col[3][2]], data_hard[CC_col[4][2]], data_hard[CC_col[5][2]], data_hard[CC_col[6][2]], data_hard[CC_col[7][2]]])

    small_CE = pd.concat([data_simple[CE_col[1][2]], data_simple[CE_col[2][2]], data_simple[CE_col[3][2]], data_simple[CE_col[4][2]], data_simple[CE_col[5][2]], data_simple[CE_col[6][2]], data_simple[CE_col[7][2]]])
    medium_CE = pd.concat([data_medium[CE_col[1][2]], data_medium[CE_col[2][2]], data_medium[CE_col[3][2]], data_medium[CE_col[4][2]], data_medium[CE_col[5][2]], data_medium[CE_col[6][2]], data_medium[CE_col[7][2]]])
    large_CE = pd.concat([data_hard[CE_col[1][2]], data_hard[CE_col[2][2]], data_hard[CE_col[3][2]], data_hard[CE_col[4][2]], data_hard[CE_col[5][2]], data_hard[CE_col[6][2]], data_hard[CE_col[7][2]]])

    small_chain_true = small_chain[small_chain == True]
    medium_chain_true = medium_chain[medium_chain == True]
    large_chain_true = large_chain[large_chain == True]

    small_CC_true = small_CC[small_CC == True]
    medium_CC_true = medium_CC[medium_CC == True]
    large_CC_true = large_CC[large_CC == True]

    small_CE_true = small_CE[small_CE == True]
    medium_CE_true = medium_CE[medium_CE == True]
    large_CE_true = large_CE[large_CE == True]

    #chain
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(medium_chain_true), len(medium_chain)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(medium_chain_true), len(medium_chain)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(large_chain_true), len(large_chain)).pvalue

    rows.append(["chain", "small - large", small_medium, medium_large, small_large])

    #CC
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(medium_CC_true), len(medium_CC)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(medium_CC_true), len(medium_CC)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(large_CC_true), len(large_CC)).pvalue

    rows.append(["CC", "small - large", small_medium, medium_large, small_large])

    #CE
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(medium_CE_true), len(medium_CE)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_CE_true), len(large_CE), len(medium_CE_true), len(medium_CE)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_CE_true), len(small_CE), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["CE", "small - large", small_medium, medium_large, small_large])

    #small
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CC_true), len(small_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(small_CE_true), len(small_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(small_CE_true), len(small_CE)).pvalue

    rows.append(["small", "chain - CE", chain_CC, CC_CE, chain_CE])

    #medium
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CC_true), len(medium_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_CC_true), len(medium_CC), len(medium_CE_true), len(medium_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(medium_chain_true), len(medium_chain), len(medium_CE_true), len(medium_CE)).pvalue

    rows.append(["medium", "chain - CE", chain_CC, CC_CE, chain_CE])

    #large
    chain_CC = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CC_true), len(large_CC)).pvalue
    CC_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_CC_true), len(large_CC), len(large_CE_true), len(large_CE)).pvalue
    chain_CE = sm.stats.proportion.score_test_proportions_2indep(len(large_chain_true), len(large_chain), len(large_CE_true), len(large_CE)).pvalue

    rows.append(["large", "chain - CE", chain_CC, CC_CE, chain_CE])


    
    
    table = pd.DataFrame(rows)
    table.to_csv("Stats\\Range.csv", header = False, index = False)


"""
Checks for Range variabilty between cohorts
"""
def CPTTest(data):

    rows = []

    data_simple = data[data['Cohort_Type']=="simple"]

    rows.append(headers3)

    for qnum in [1, 3, 4]: #adding chain

        data_simple_true = data_simple[data_simple[chain_col[qnum][3]] == True]
        dir_data_simple_true = data_simple[data_simple[chain_col[qnum][0]] == True]
        
        cpt_dir  = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(dir_data_simple_true), len(data_simple)).pvalue

        rows.append(["chain", qnum, cpt_dir])

        
    for qnum in [4, 5, 7]: #adding common cause

        data_simple_true = data_simple[data_simple[CC_col[qnum][3]] == True]
        dir_data_simple_true = data_simple[data_simple[CC_col[qnum][0]] == True]

        cpt_dir  = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(dir_data_simple_true), len(data_simple)).pvalue

        rows.append(["CC", qnum, cpt_dir])
  
        #combined
    small_chain = pd.concat([data_simple[chain_col[1][3]], data_simple[chain_col[3][3]], data_simple[chain_col[4][3]]])
    dir_small_chain = pd.concat([data_simple[chain_col[1][0]], data_simple[chain_col[3][0]], data_simple[chain_col[4][0]]])
    
    small_CC = pd.concat([data_simple[CC_col[4][3]], data_simple[CC_col[5][3]], data_simple[CC_col[7][3]]])
    dir_small_CC = pd.concat([data_simple[CC_col[4][0]], data_simple[CC_col[5][0]], data_simple[CC_col[7][0]]])

    small_chain_true = small_chain[small_chain == True]
    dir_small_chain_true = dir_small_chain[dir_small_chain == True]

    small_CC_true = small_CC[small_CC == True]
    dir_small_CC_true = dir_small_CC[dir_small_CC == True]

    #chain
    cpt_dir = sm.stats.proportion.score_test_proportions_2indep(len(small_chain_true), len(small_chain), len(dir_small_chain_true), len(dir_small_chain)).pvalue

    rows.append(["chain", "all", cpt_dir])

    #CC
    cpt_dir = sm.stats.proportion.score_test_proportions_2indep(len(small_CC_true), len(small_CC), len(dir_small_CC_true), len(dir_small_CC)).pvalue

    rows.append(["CC", "all", cpt_dir])
    
    
    table = pd.DataFrame(rows)
    table.to_csv("Stats\\CPT.csv", header = False, index = False)

"""
Checks for Range variabilty between cohorts
"""
def Cond2(data):

    rows = []

    data_simple = data[data['Cohort_Type']=="simple"]
    data_medium = data[data['Cohort_Type']=="medium"]
    data_hard = data[data['Cohort_Type']=="hard"]

    rows.append(headers2)

    #adding chain q4

    data_simple_true = data_simple[data_simple[chain_col[4][4]] == True]
    data_medium_true = data_medium[data_medium[chain_col[4][4]] == True]
    data_hard_true = data_hard[data_hard[chain_col[4][4]] == True]
        
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

    rows.append(["chain", 4, small_medium, medium_large, small_large])

        
    for qnum in [6,7]: #adding common cause

        data_simple_true = data_simple[data_simple[CC_col[qnum][4]] == True]
        data_medium_true = data_medium[data_medium[CC_col[qnum][4]] == True]
        data_hard_true = data_hard[data_hard[CC_col[qnum][4]] == True]        

        small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
        medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
        small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

        rows.append(["CC", qnum, small_medium, medium_large, small_large])

##    for qnum in range(1,8): #adding common effect
##
##        data_simple_true = data_simple[data_simple[CE_col[qnum][4]] == True]
##        data_medium_true = data_medium[data_medium[CE_col[qnum][4]] == True]
##        data_hard_true = data_hard[data_hard[CE_col[qnum][4]] == True]        
##
##        small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
##        medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
##        small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue
##
##        rows.append(["CE", qnum, small_medium, medium_large, small_large])
  
        #combined
    small_all = pd.concat([data_simple[chain_col[4][4]], data_simple[CC_col[6][4]], data_simple[CC_col[7][4]]])
    medium_all = pd.concat([data_medium[chain_col[4][4]], data_medium[CC_col[6][4]], data_medium[CC_col[7][4]]])
    large_all = pd.concat([data_hard[chain_col[4][4]], data_hard[CC_col[6][4]], data_hard[CC_col[7][4]]])

    small_all_true = small_all[small_all == True]
    medium_all_true = medium_all[medium_all == True]
    large_all_true = large_all[large_all == True]

    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_all_true), len(small_all), len(medium_all_true), len(medium_all)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_all_true), len(large_all), len(medium_all_true), len(medium_all)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_all_true), len(small_all), len(large_all_true), len(large_all)).pvalue

    rows.append(["all", "small - large", small_medium, medium_large, small_large])
    
    table = pd.DataFrame(rows)
    table.to_csv("Stats\\Cond2.csv", header = False, index = False)


"""
Checks for Range variabilty between cohorts
"""
def Cond3(data):

    rows = []

    data_simple = data[data['Cohort_Type']=="simple"]
    data_medium = data[data['Cohort_Type']=="medium"]
    data_hard = data[data['Cohort_Type']=="hard"]

    rows.append(headers2)

    #adding CC q2

    data_simple_true = data_simple[data_simple[CC_col[2][5]] == True]
    data_medium_true = data_medium[data_medium[CC_col[2][5]] == True]
    data_hard_true = data_hard[data_hard[CC_col[2][5]] == True]
        
    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

    rows.append(["CC", 2, small_medium, medium_large, small_large])

    for qnum in [3, 4]: #adding common effect

        data_simple_true = data_simple[data_simple[CE_col[qnum][5]] == True]
        data_medium_true = data_medium[data_medium[CE_col[qnum][5]] == True]
        data_hard_true = data_hard[data_hard[CE_col[qnum][5]] == True]        

        small_medium = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_medium_true), len(data_medium)).pvalue
        medium_large = sm.stats.proportion.score_test_proportions_2indep(len(data_hard_true), len(data_hard), len(data_medium_true), len(data_medium)).pvalue
        small_large = sm.stats.proportion.score_test_proportions_2indep(len(data_simple_true), len(data_simple), len(data_hard_true), len(data_hard)).pvalue

        rows.append(["CE", qnum, small_medium, medium_large, small_large])
  
        #combined
    small_all = pd.concat([data_simple[CC_col[2][5]], data_simple[CE_col[3][5]], data_simple[CE_col[4][5]]])
    medium_all = pd.concat([data_medium[CC_col[2][5]], data_medium[CE_col[3][5]], data_medium[CE_col[4][5]]])
    large_all = pd.concat([data_hard[CC_col[2][5]], data_hard[CE_col[3][5]], data_hard[CE_col[4][5]]])

    small_all_true = small_all[small_all == True]
    medium_all_true = medium_all[medium_all == True]
    large_all_true = large_all[large_all == True]

    small_medium = sm.stats.proportion.score_test_proportions_2indep(len(small_all_true), len(small_all), len(medium_all_true), len(medium_all)).pvalue
    medium_large = sm.stats.proportion.score_test_proportions_2indep(len(large_all_true), len(large_all), len(medium_all_true), len(medium_all)).pvalue
    small_large = sm.stats.proportion.score_test_proportions_2indep(len(small_all_true), len(small_all), len(large_all_true), len(large_all)).pvalue

    rows.append(["all", "small - large", small_medium, medium_large, small_large])
    
    table = pd.DataFrame(rows)
    table.to_csv("Stats\\Cond3.csv", header = False, index = False)


"""
Checks for Distance variabilty between cohorts
"""
def DistanceNormTest(data):

    rows = []

    data_simple = data[data['Cohort_Type']=="simple"]
    data_medium = data[data['Cohort_Type']=="medium"]
    data_hard = data[data['Cohort_Type']=="hard"]

    small_chain_right = pd.Series([])
    medium_chain_right = pd.Series([])
    large_chain_right = pd.Series([])

    small_CC_right = pd.Series([])
    medium_CC_right = pd.Series([])
    large_CC_right = pd.Series([])

    small_CE_right = pd.Series([])
    medium_CE_right = pd.Series([])
    large_CE_right = pd.Series([])

    rows.append(headers4)

    for qnum in range(1,6): #adding chain

        data_simple_right = data_simple[data_simple[chain_col[qnum][0]] == True]
        data_medium_right = data_medium[data_medium[chain_col[qnum][0]] == True]
        data_hard_right = data_hard[data_hard[chain_col[qnum][0]] == True]

        small_medium_all = stats.ranksums(data_simple[chain_col[qnum][6]], data_medium[chain_col[qnum][6]]).pvalue
        medium_large_all = stats.ranksums(data_hard[chain_col[qnum][6]], data_medium[chain_col[qnum][6]]).pvalue
        small_large_all = stats.ranksums(data_simple[chain_col[qnum][6]], data_hard[chain_col[qnum][6]]).pvalue

        small_medium_correct = stats.ranksums(data_simple_right[chain_col[qnum][6]], data_medium_right[chain_col[qnum][6]]).pvalue
        medium_large_correct = stats.ranksums(data_hard_right[chain_col[qnum][6]], data_medium_right[chain_col[qnum][6]]).pvalue
        small_large_correct = stats.ranksums(data_simple_right[chain_col[qnum][6]], data_hard_right[chain_col[qnum][6]]).pvalue

        small_chain_right = pd.concat([small_chain_right, data_simple_right[chain_col[qnum][6]]])
        medium_chain_right = pd.concat([medium_chain_right, data_medium_right[chain_col[qnum][6]]])
        large_chain_right = pd.concat([large_chain_right, data_hard_right[chain_col[qnum][6]]])

        rows.append(["chain", qnum, data_simple[chain_col[qnum][6]].mean(), data_simple[chain_col[qnum][6]].std(), data_medium[chain_col[qnum][6]].mean(), data_medium[chain_col[qnum][6]].std(), data_hard[chain_col[qnum][6]].mean(), data_hard[chain_col[qnum][6]].std(), small_medium_all, medium_large_all, small_large_all, data_simple_right[chain_col[qnum][6]].mean(), data_simple_right[chain_col[qnum][6]].std(), data_medium_right[chain_col[qnum][6]].mean(), data_medium_right[chain_col[qnum][6]].std(), data_hard_right[chain_col[qnum][6]].mean(), data_hard_right[chain_col[qnum][6]].std(), small_medium_correct, medium_large_correct, small_large_correct])

        
    for qnum in range(1,8): #adding common cause

        data_simple_right = data_simple[data_simple[CC_col[qnum][0]] == True]
        data_medium_right = data_medium[data_medium[CC_col[qnum][0]] == True]
        data_hard_right = data_hard[data_hard[CC_col[qnum][0]] == True]

        small_medium_all = stats.ranksums(data_simple[CC_col[qnum][6]], data_medium[CC_col[qnum][6]]).pvalue
        medium_large_all = stats.ranksums(data_hard[CC_col[qnum][6]], data_medium[CC_col[qnum][6]]).pvalue
        small_large_all = stats.ranksums(data_simple[CC_col[qnum][6]], data_hard[CC_col[qnum][6]]).pvalue

        small_medium_correct = stats.ranksums(data_simple_right[CC_col[qnum][6]], data_medium_right[CC_col[qnum][6]]).pvalue
        medium_large_correct = stats.ranksums(data_hard_right[CC_col[qnum][6]], data_medium_right[CC_col[qnum][6]]).pvalue
        small_large_correct = stats.ranksums(data_simple_right[CC_col[qnum][6]], data_hard_right[CC_col[qnum][6]]).pvalue

        small_CC_right = pd.concat([small_CC_right, data_simple_right[CC_col[qnum][6]]])
        medium_CC_right = pd.concat([medium_CC_right, data_medium_right[CC_col[qnum][6]]])
        large_CC_right = pd.concat([large_CC_right, data_hard_right[CC_col[qnum][6]]])

        rows.append(["CC", qnum, data_simple[CC_col[qnum][6]].mean(), data_simple[CC_col[qnum][6]].std(), data_medium[CC_col[qnum][6]].mean(), data_medium[CC_col[qnum][6]].std(), data_hard[CC_col[qnum][6]].mean(), data_hard[CC_col[qnum][6]].std(), small_medium_all, medium_large_all, small_large_all, data_simple_right[CC_col[qnum][6]].mean(), data_simple_right[CC_col[qnum][6]].std(), data_medium_right[CC_col[qnum][6]].mean(), data_medium_right[CC_col[qnum][6]].std(), data_hard_right[CC_col[qnum][6]].mean(), data_hard_right[CC_col[qnum][6]].std(), small_medium_correct, medium_large_correct, small_large_correct])

    for qnum in range(1,8): #adding common effect

        data_simple_right = data_simple[data_simple[CE_col[qnum][0]] == True]
        data_medium_right = data_medium[data_medium[CE_col[qnum][0]] == True]
        data_hard_right = data_hard[data_hard[CE_col[qnum][0]] == True]

        small_medium_all = stats.ranksums(data_simple[CE_col[qnum][6]], data_medium[CE_col[qnum][6]]).pvalue
        medium_large_all = stats.ranksums(data_hard[CE_col[qnum][6]], data_medium[CE_col[qnum][6]]).pvalue
        small_large_all = stats.ranksums(data_simple[CE_col[qnum][6]], data_hard[CE_col[qnum][6]]).pvalue

        small_medium_correct = stats.ranksums(data_simple_right[CE_col[qnum][6]], data_medium_right[CE_col[qnum][6]]).pvalue
        medium_large_correct = stats.ranksums(data_hard_right[CE_col[qnum][6]], data_medium_right[CE_col[qnum][6]]).pvalue
        small_large_correct = stats.ranksums(data_simple_right[CE_col[qnum][6]], data_hard_right[CE_col[qnum][6]]).pvalue

        small_CE_right = pd.concat([small_CE_right, data_simple_right[CE_col[qnum][6]]])
        medium_CE_right = pd.concat([medium_CE_right, data_medium_right[CE_col[qnum][6]]])
        large_CE_right = pd.concat([large_CE_right, data_hard_right[CE_col[qnum][6]]])

        rows.append(["CE", qnum, data_simple[CE_col[qnum][6]].mean(), data_simple[CE_col[qnum][6]].std(), data_medium[CE_col[qnum][6]].mean(), data_medium[CE_col[qnum][6]].std(), data_hard[CE_col[qnum][6]].mean(), data_hard[CE_col[qnum][6]].std(), small_medium_all, medium_large_all, small_large_all, data_simple_right[CE_col[qnum][6]].mean(), data_simple_right[CE_col[qnum][6]].std(), data_medium_right[CE_col[qnum][6]].mean(), data_medium_right[CE_col[qnum][6]].std(), data_hard_right[CE_col[qnum][6]].mean(), data_hard_right[CE_col[qnum][6]].std(), small_medium_correct, medium_large_correct, small_large_correct])

     #combined
    small_chain = pd.concat([data_simple[chain_col[1][6]], data_simple[chain_col[2][6]], data_simple[chain_col[3][6]], data_simple[chain_col[4][6]], data_simple[chain_col[5][6]]])
    medium_chain = pd.concat([data_medium[chain_col[1][6]], data_medium[chain_col[2][6]], data_medium[chain_col[3][6]], data_medium[chain_col[4][6]], data_medium[chain_col[5][6]]])
    large_chain = pd.concat([data_hard[chain_col[1][6]], data_hard[chain_col[2][6]], data_hard[chain_col[3][6]], data_hard[chain_col[4][6]], data_hard[chain_col[5][6]]])

    small_CC = pd.concat([data_simple[CC_col[1][6]], data_simple[CC_col[2][6]], data_simple[CC_col[3][6]], data_simple[CC_col[4][6]], data_simple[CC_col[5][6]], data_simple[CC_col[6][6]], data_simple[CC_col[7][6]]])
    medium_CC = pd.concat([data_medium[CC_col[1][6]], data_medium[CC_col[2][6]], data_medium[CC_col[3][6]], data_medium[CC_col[4][6]], data_medium[CC_col[5][6]], data_medium[CC_col[6][6]], data_medium[CC_col[7][6]]])
    large_CC = pd.concat([data_hard[CC_col[1][6]], data_hard[CC_col[2][6]], data_hard[CC_col[3][6]], data_hard[CC_col[4][6]], data_hard[CC_col[5][6]], data_hard[CC_col[6][6]], data_hard[CC_col[7][6]]])

    small_CE = pd.concat([data_simple[CE_col[1][6]], data_simple[CE_col[2][6]], data_simple[CE_col[3][6]], data_simple[CE_col[4][6]], data_simple[CE_col[5][6]], data_simple[CE_col[6][6]], data_simple[CE_col[7][6]]])
    medium_CE = pd.concat([data_medium[CE_col[1][6]], data_medium[CE_col[2][6]], data_medium[CE_col[3][6]], data_medium[CE_col[4][6]], data_medium[CE_col[5][6]], data_medium[CE_col[6][6]], data_medium[CE_col[7][6]]])
    large_CE = pd.concat([data_hard[CE_col[1][6]], data_hard[CE_col[2][6]], data_hard[CE_col[3][6]], data_hard[CE_col[4][6]], data_hard[CE_col[5][6]], data_hard[CE_col[6][6]], data_hard[CE_col[7][6]]])


    #chain all
    small_medium_all = stats.ranksums(small_chain, medium_chain).pvalue
    medium_large_all = stats.ranksums(medium_chain, large_chain).pvalue
    small_large_all = stats.ranksums(small_chain, large_chain).pvalue

    rows.append(["chain", "small - large all", small_chain.mean(), small_chain.std(), medium_chain.mean(), medium_chain.std(), large_chain.mean(), large_chain.std(), small_medium_all, medium_large_all, small_large_all])

    #CC all
    small_medium_all = stats.ranksums(small_CC, medium_CC).pvalue
    medium_large_all = stats.ranksums(medium_CC, large_CC).pvalue
    small_large_all = stats.ranksums(small_CC, large_CC).pvalue

    rows.append(["CC", "small - large all", small_CC.mean(), small_CC.std(), medium_CC.mean(), medium_CC.std(), large_CC.mean(), large_CC.std(), small_medium_all, medium_large_all, small_large_all])

    #CE all
    small_medium_all = stats.ranksums(small_CE, medium_CE).pvalue
    medium_large_all = stats.ranksums(medium_CE, large_CE).pvalue
    small_large_all = stats.ranksums(small_CE, large_CE).pvalue

    rows.append(["CE", "small - large all", small_CE.mean(), small_CE.std(), medium_CE.mean(), medium_CE.std(), large_CE.mean(), large_CE.std(), small_medium_all, medium_large_all, small_large_all])

    #small all
    chain_CC_all = stats.ranksums(small_chain, small_CC).pvalue
    CC_CE_all = stats.ranksums(small_CC, small_CE).pvalue
    chain_CE_all = stats.ranksums(small_chain, small_CE).pvalue

    rows.append(["small", "chain - CE all", small_chain.mean(), small_chain.std(), small_CC.mean(), small_CC.std(), small_CE.mean(), small_CE.std(), chain_CC_all, CC_CE_all, chain_CE_all])

    #medium all
    chain_CC_all = stats.ranksums(medium_chain, medium_CC).pvalue
    CC_CE_all = stats.ranksums(medium_CC, medium_CE).pvalue
    chain_CE_all = stats.ranksums(medium_chain, medium_CE).pvalue

    rows.append(["medium", "chain - CE all", medium_chain.mean(), medium_chain.std(), medium_CC.mean(), medium_CC.std(), medium_CE.mean(), medium_CE.std(), chain_CC_all, CC_CE_all, chain_CE_all])

    #large all
    chain_CC_all = stats.ranksums(large_chain, large_CC).pvalue
    CC_CE_all = stats.ranksums(large_CC, large_CE).pvalue
    chain_CE_all = stats.ranksums(large_chain, large_CE).pvalue

    rows.append(["large", "chain - CE all", large_chain.mean(), large_chain.std(), large_CC.mean(), large_CC.std(), large_CE.mean(), large_CE.std(), chain_CC_all, CC_CE_all, chain_CE_all])

    #chain right
    small_medium_right = stats.ranksums(small_chain_right, medium_chain_right).pvalue
    medium_large_right = stats.ranksums(medium_chain_right, large_chain_right).pvalue
    small_large_right = stats.ranksums(small_chain_right, large_chain_right).pvalue

    rows.append(["chain", "small - large right", small_chain_right.mean(), small_chain_right.std(), medium_chain_right.mean(), medium_chain_right.std(), large_chain_right.mean(), large_chain_right.std(), small_medium_right, medium_large_right, small_large_right])

    #CC right
    small_medium_right = stats.ranksums(small_CC_right, medium_CC_right).pvalue
    medium_large_right = stats.ranksums(medium_CC_right, large_CC_right).pvalue
    small_large_right = stats.ranksums(small_CC_right, large_CC_right).pvalue

    rows.append(["CC", "small - large right", small_CC_right.mean(), small_CC_right.std(), medium_CC_right.mean(), medium_CC_right.std(), large_CC_right.mean(), large_CC_right.std(), small_medium_right, medium_large_right, small_large_right])

    #CE right
    small_medium_right = stats.ranksums(small_CE_right, medium_CE_right).pvalue
    medium_large_right = stats.ranksums(medium_CE_right, large_CE_right).pvalue
    small_large_right = stats.ranksums(small_CE_right, large_CE_right).pvalue

    rows.append(["CE", "small - large right", small_CE_right.mean(), small_CE_right.std(), medium_CE_right.mean(), medium_CE_right.std(), large_CE_right.mean(), large_CE_right.std(), small_medium_right, medium_large_right, small_large_right])

    #small right
    chain_CC_right = stats.ranksums(small_chain_right, small_CC_right).pvalue
    CC_CE_right= stats.ranksums(small_CC_right, small_CE_right).pvalue
    chain_CE_right = stats.ranksums(small_chain_right, small_CE_right).pvalue

    rows.append(["small", "chain - CE right", small_chain_right.mean(), small_chain_right.std(), small_CC_right.mean(), small_CC_right.std(), small_CE_right.mean(), small_CE_right.std(), chain_CC_right, CC_CE_right, chain_CE_right])

    #medium right
    chain_CC_right = stats.ranksums(medium_chain_right, medium_CC_right).pvalue
    CC_CE_right = stats.ranksums(medium_CC_right, medium_CE_right).pvalue
    chain_CE_right = stats.ranksums(medium_chain_right, medium_CE_right).pvalue

    rows.append(["medium", "chain - CE right", medium_chain_right.mean(), medium_chain_right.std(), medium_CC_right.mean(), medium_CC_right.std(), medium_CE_right.mean(), medium_CE_right.std(), chain_CC_right, CC_CE_right, chain_CE_right])

    #large right
    chain_CC_right = stats.ranksums(large_chain_right, large_CC_right).pvalue
    CC_CE_right = stats.ranksums(large_CC_right, large_CE_right).pvalue
    chain_CE_right = stats.ranksums(large_chain_right, large_CE_right).pvalue

    rows.append(["large", "chain - CE right", large_chain_right.mean(), large_chain_right.std(), large_CC_right.mean(), large_CC_right.std(), large_CE_right.mean(), large_CE_right.std(), chain_CC_right, CC_CE_right, chain_CE_right])

    table = pd.DataFrame(rows)
    table.to_csv("Stats\\Distance_norm.csv", header = False, index = False)

#Cond2(data)
#Cond3(data)
#CPTTest(data)
#DistanceTest(data)
DirectionalityTest(data)
#RangeTest(data)
#DistanceNormTest(data)
