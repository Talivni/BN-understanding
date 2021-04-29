import numpy as np
import pandas as pd 
import re
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\talil\\OneDrive\\Documents\\GitHub\\BN-understanding\\processedData_0202_fixed.csv")

headers = ["Block", "Question", "Cohort", "BestEstimate_Mean", "BestEstimate_Std", "Range_All_Mean", "Range_All_Std", "Range_Correct_Mean", "Range_Correct_Std", "Cond1_Increased_Count",  "Cond1_Increased_Perc", "Cond1_Decreased_Count",  "Cond1_Decreased_Perc", "Cond1_The_Same_Count",  "Cond1_The_Same_Perc", "Distance_All_Mean", "Distance_All_Std", "Distance_Correct_Mean", "Distance_Correct_Std", "AnsInRange_All_Count", "AnsInRange_All_Perc", "AnsInRange_Correct_Count", "AnsInRange_Correct_Perc", "readCPT_count", "readCPT_perc", "Cond2_Count", "Cond2_Perc", "Cond3_Count", "Cond3_Perc"]
#headers = ["Block", "Question", "Cohort", "BestEstimate_Mean", "BestEstimate_Std", "Within_Range_All", "Within_Range_Right", "Width_All", "Width_right", "Cond1_Sum_Increased", "Cond1_Sum_Decreased", "Cond1_Sum_Same", "Cond1_Perc_Increased", "Cond1_Perc_ Decreased", "Cond1_Perc_Same", "Cond2_Sum", "Cond2_Perc", "Cond3_Sum", "Cond3_Perc", "Cond4_Min", "Cond4_Max", "Cond4_Mean", "Cond4_Std", "readCPT_sum", "readCPT_perc"]

chain_col = { 1: ['Chain_Q1_Cond1_P_Decreased', 'Chain_Q1_P_BestEstimate', 'Chain_Q1_P_AnsWithinRange', 'Chain_Q1_P_Range', 'Chain_Q1_Cond1_P_Increased', 'Chain_Q1_Cond1_P_Decreased', 'Chain_Q1_Cond1_P_The_Same', 'Chain_Q1_Cond2_P', 'Chain_Q1_Cond3_P', 'Chain_Q1_Cond4_P', 'Chain_Q1_ReadCPT', 'Chain_Q1_Distance'],
              2: ['Chain_Q2_Cond1_P_Decreased', 'Chain_Q2_P_BestEstimate', 'Chain_Q2_P_AnsWithinRange', 'Chain_Q2_P_Range', 'Chain_Q2_Cond1_P_Increased', 'Chain_Q2_Cond1_P_Decreased', 'Chain_Q2_Cond1_P_The_Same', 'Chain_Q2_Cond2_P', 'Chain_Q2_Cond3_P', 'Chain_Q2_Cond4_P', 'Chain_Q2_ReadCPT', 'Chain_Q2_Distance'],
              3: ['Chain_Q3_Cond1_P_Increased', 'Chain_Q3_P_BestEstimate', 'Chain_Q3_P_AnsWithinRange', 'Chain_Q3_P_Range', 'Chain_Q3_Cond1_P_Increased', 'Chain_Q3_Cond1_P_Decreased', 'Chain_Q3_Cond1_P_The_Same', 'Chain_Q3_Cond2_P', 'Chain_Q3_Cond3_P', 'Chain_Q3_Cond4_P', 'Chain_Q3_ReadCPT', 'Chain_Q3_Distance'],
              4: ['Chain_Q4_Cond1_P_Increased', 'Chain_Q4_P_BestEstimate', 'Chain_Q4_P_AnsWithinRange', 'Chain_Q4_P_Range', 'Chain_Q4_Cond1_P_Increased', 'Chain_Q4_Cond1_P_Decreased', 'Chain_Q4_Cond1_P_The_Same', 'Chain_Q4_Cond2_P', 'Chain_Q4_Cond3_P', 'Chain_Q4_Cond4_P', 'Chain_Q4_ReadCPT', 'Chain_Q4_Distance'],
              5: ['Chain_Q5_Cond1_P_The_Same', 'Chain_Q5_P_BestEstimate', 'Chain_Q5_P_AnsWithinRange', 'Chain_Q5_P_Range', 'Chain_Q5_Cond1_P_Increased', 'Chain_Q5_Cond1_P_Decreased', 'Chain_Q5_Cond1_P_The_Same', 'Chain_Q5_Cond2_P', 'Chain_Q5_Cond3_P', 'Chain_Q5_Cond4_P', 'Chain_Q5_ReadCPT', 'Chain_Q5_Distance']}    
                  
CC_col = { 1: ['CC_Q1_Cond1_P_Increased', 'CC_Q1_P_BestEstimate', 'CC_Q1_P_AnsWithinRange', 'CC_Q1_P_Range', 'CC_Q1_Cond1_P_Increased', 'CC_Q1_Cond1_P_Decreased', 'CC_Q1_Cond1_P_The_Same', 'CC_Q1_Cond2_P', 'CC_Q1_Cond3_P', 'CC_Q1_Cond4_P', 'CC_Q1_ReadCPT', 'CC_Q1_Distance'],
              2: ['CC_Q2_Cond1_P_Increased', 'CC_Q2_P_BestEstimate', 'CC_Q2_P_AnsWithinRange', 'CC_Q2_P_Range', 'CC_Q2_Cond1_P_Increased', 'CC_Q2_Cond1_P_Decreased', 'CC_Q2_Cond1_P_The_Same', 'CC_Q2_Cond2_P', 'CC_Q2_Cond3_P', 'CC_Q2_Cond4_P', 'CC_Q2_ReadCPT', 'CC_Q2_Distance'],
              3: ['CC_Q3_Cond1_P_Increased', 'CC_Q3_P_BestEstimate', 'CC_Q3_P_AnsWithinRange', 'CC_Q3_P_Range', 'CC_Q3_Cond1_P_Increased', 'CC_Q3_Cond1_P_Decreased', 'CC_Q3_Cond1_P_The_Same', 'CC_Q3_Cond2_P', 'CC_Q3_Cond3_P', 'CC_Q3_Cond4_P', 'CC_Q3_ReadCPT', 'CC_Q3_Distance'],
              4: ['CC_Q4_Cond1_P_Decreased', 'CC_Q4_P_BestEstimate', 'CC_Q4_P_AnsWithinRange', 'CC_Q4_P_Range', 'CC_Q4_Cond1_P_Increased', 'CC_Q4_Cond1_P_Decreased', 'CC_Q4_Cond1_P_The_Same', 'CC_Q4_Cond2_P', 'CC_Q4_Cond3_P', 'CC_Q4_Cond4_P', 'CC_Q4_ReadCPT', 'CC_Q4_Distance'],
              5: ['CC_Q5_Cond1_P_Increased', 'CC_Q5_P_BestEstimate', 'CC_Q5_P_AnsWithinRange', 'CC_Q5_P_Range', 'CC_Q5_Cond1_P_Increased', 'CC_Q5_Cond1_P_Decreased', 'CC_Q5_Cond1_P_The_Same', 'CC_Q5_Cond2_P', 'CC_Q5_Cond3_P', 'CC_Q5_Cond4_P', 'CC_Q5_ReadCPT', 'CC_Q5_Distance'],
              6: ['CC_Q6_Cond1_P_The_Same', 'CC_Q6_P_BestEstimate', 'CC_Q6_P_AnsWithinRange', 'CC_Q6_P_Range', 'CC_Q6_Cond1_P_Increased', 'CC_Q6_Cond1_P_Decreased', 'CC_Q6_Cond1_P_The_Same', 'CC_Q6_Cond2_P', 'CC_Q6_Cond3_P', 'CC_Q6_Cond4_P', 'CC_Q6_ReadCPT', 'CC_Q6_Distance'],
              7: ['CC_Q7_Cond1_P_Increased', 'CC_Q7_P_BestEstimate', 'CC_Q7_P_AnsWithinRange', 'CC_Q7_P_Range', 'CC_Q7_Cond1_P_Increased', 'CC_Q7_Cond1_P_Decreased', 'CC_Q7_Cond1_P_The_Same', 'CC_Q7_Cond2_P', 'CC_Q7_Cond3_P', 'CC_Q7_Cond4_P', 'CC_Q7_ReadCPT', 'CC_Q7_Distance']}

CE_col = { 1: ['CE_Q1_Cond1_P_The_Same', 'CE_Q1_P_BestEstimate', 'CE_Q1_P_AnsWithinRange', 'CE_Q1_P_Range', 'CE_Q1_Cond1_P_Increased', 'CE_Q1_Cond1_P_Decreased', 'CE_Q1_Cond1_P_The_Same', 'CE_Q1_Cond2_P', 'CE_Q1_Cond3_P', 'CE_Q1_Cond4_P', 'CE_Q1_ReadCPT', 'CE_Q1_Distance'],
              2: ['CE_Q2_Cond1_P_Increased', 'CE_Q2_P_BestEstimate', 'CE_Q2_P_AnsWithinRange', 'CE_Q2_P_Range', 'CE_Q2_Cond1_P_Increased', 'CE_Q2_Cond1_P_Decreased', 'CE_Q2_Cond1_P_The_Same', 'CE_Q2_Cond2_P', 'CE_Q2_Cond3_P', 'CE_Q2_Cond4_P', 'CE_Q2_ReadCPT', 'CE_Q2_Distance'],
              3: ['CE_Q3_Cond1_P_Increased', 'CE_Q3_P_BestEstimate', 'CE_Q3_P_AnsWithinRange', 'CE_Q3_P_Range', 'CE_Q3_Cond1_P_Increased', 'CE_Q3_Cond1_P_Decreased', 'CE_Q3_Cond1_P_The_Same', 'CE_Q3_Cond2_P', 'CE_Q3_Cond3_P', 'CE_Q3_Cond4_P', 'CE_Q3_ReadCPT', 'CE_Q3_Distance'],
              4: ['CE_Q4_Cond1_P_Increased', 'CE_Q4_P_BestEstimate', 'CE_Q4_P_AnsWithinRange', 'CE_Q4_P_Range', 'CE_Q4_Cond1_P_Increased', 'CE_Q4_Cond1_P_Decreased', 'CE_Q4_Cond1_P_The_Same', 'CE_Q4_Cond2_P', 'CE_Q4_Cond3_P', 'CE_Q4_Cond4_P', 'CE_Q4_ReadCPT', 'CE_Q4_Distance'],
              5: ['CE_Q5_Cond1_P_Decreased', 'CE_Q5_P_BestEstimate', 'CE_Q5_P_AnsWithinRange', 'CE_Q5_P_Range', 'CE_Q5_Cond1_P_Increased', 'CE_Q5_Cond1_P_Decreased', 'CE_Q5_Cond1_P_The_Same', 'CE_Q5_Cond2_P', 'CE_Q5_Cond3_P', 'CE_Q5_Cond4_P', 'CE_Q5_ReadCPT', 'CE_Q5_Distance'],
              6: ['CE_Q6_Cond1_P_Increased', 'CE_Q6_P_BestEstimate', 'CE_Q6_P_AnsWithinRange', 'CE_Q6_P_Range', 'CE_Q6_Cond1_P_Increased', 'CE_Q6_Cond1_P_Decreased', 'CE_Q6_Cond1_P_The_Same', 'CE_Q6_Cond2_P', 'CE_Q6_Cond3_P', 'CE_Q6_Cond4_P', 'CE_Q6_ReadCPT', 'CE_Q6_Distance'],
              7: ['CE_Q7_Cond1_P_Increased', 'CE_Q7_P_BestEstimate', 'CE_Q7_P_AnsWithinRange', 'CE_Q7_P_Range', 'CE_Q7_Cond1_P_Increased', 'CE_Q7_Cond1_P_Decreased', 'CE_Q7_Cond1_P_The_Same', 'CE_Q7_Cond2_P', 'CE_Q7_Cond3_P', 'CE_Q7_Cond4_P', 'CE_Q7_ReadCPT', 'CE_Q7_Distance']}


"""
This function Arranges and writes the data into a CSV in the following format:
TBD
"""
def arrange_csv(data):

   rows = []

   data_simple = data[data['Cohort_Type']=="simple"]
   data_medium = data[data['Cohort_Type']=="medium"]
   data_hard = data[data['Cohort_Type']=="hard"]
   
   print("small:", len(data_simple))
   print("medium:", len(data_medium))
   print("large:", len(data_hard))

   rows.append(headers)

   for qnum in range(1,6): #adding chain

      data_simple_right = data_simple[data_simple[chain_col[qnum][0]] == True]
      data_medium_right = data_medium[data_medium[chain_col[qnum][0]] == True]
      data_hard_right = data_hard[data_hard[chain_col[qnum][0]] == True]
      
      rows.append(["Chain", qnum, "small", round(data_simple[chain_col[qnum][1]].mean(), 1), round(data_simple[chain_col[qnum][1]].std(), 1), round(data_simple[chain_col[qnum][3]].mean(), 2), round(data_simple[chain_col[qnum][3]].std(), 2), round(data_simple_right[chain_col[qnum][3]].mean() ,2), round(data_simple_right[chain_col[qnum][3]].std() ,2), len(data_simple[data_simple[chain_col[qnum][4]] == True]), round(len(data_simple[data_simple[chain_col[qnum][4]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[chain_col[qnum][5]] == True]), round(len(data_simple[data_simple[chain_col[qnum][5]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[chain_col[qnum][6]] == True]), round(len(data_simple[data_simple[chain_col[qnum][6]] == True]) / len(data_simple) * 100, 2), round(data_simple[chain_col[qnum][11]].mean(), 2), round(data_simple[chain_col[qnum][11]].std(), 2), round(data_simple_right[chain_col[qnum][11]].mean() ,2), round(data_simple_right[chain_col[qnum][11]].std() ,2), len(data_simple[data_simple[chain_col[qnum][2]] == True]), round(len(data_simple[data_simple[chain_col[qnum][2]] == True]) / len(data_simple) * 100, 2), len(data_simple_right[data_simple_right[chain_col[qnum][2]] == True]), round(len(data_simple_right[data_simple_right[chain_col[qnum][2]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[chain_col[qnum][10]] == True]), round(len(data_simple[data_simple[chain_col[qnum][10]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[chain_col[qnum][7]] == True]), round(len(data_simple[data_simple[chain_col[qnum][7]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[chain_col[qnum][8]] == True]), round(len(data_simple[data_simple[chain_col[qnum][8]] == True]) / len(data_simple) * 100, 2)])
      rows.append(["Chain", qnum, "medium", round(data_medium[chain_col[qnum][1]].mean(), 1), round(data_medium[chain_col[qnum][1]].std(), 1), round(data_medium[chain_col[qnum][3]].mean(), 2), round(data_medium[chain_col[qnum][3]].std(), 2), round(data_medium_right[chain_col[qnum][3]].mean(), 2), round(data_medium_right[chain_col[qnum][3]].std(), 2), len(data_medium[data_medium[chain_col[qnum][4]] == True]), round(len(data_medium[data_medium[chain_col[qnum][4]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[chain_col[qnum][5]] == True]), round(len(data_medium[data_medium[chain_col[qnum][5]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[chain_col[qnum][6]] == True]), round(len(data_medium[data_medium[chain_col[qnum][6]] == True]) / len(data_medium) * 100, 2), round(data_medium[chain_col[qnum][11]].mean(), 2), round(data_medium[chain_col[qnum][11]].std(), 2), round(data_medium_right[chain_col[qnum][11]].mean(), 2), round(data_medium_right[chain_col[qnum][11]].std(), 2), len(data_medium[data_medium[chain_col[qnum][2]] == True]), round(len(data_medium[data_medium[chain_col[qnum][2]] == True]) / len(data_medium) * 100, 2), len(data_medium_right[data_medium_right[chain_col[qnum][2]] == True]), round(len(data_medium_right[data_medium_right[chain_col[qnum][2]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[chain_col[qnum][10]] == True]), round(len(data_medium[data_medium[chain_col[qnum][10]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[chain_col[qnum][7]] == True]), round(len(data_medium[data_medium[chain_col[qnum][7]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[chain_col[qnum][8]] == True]), round(len(data_medium[data_medium[chain_col[qnum][8]] == True]) / len(data_medium) * 100, 2)])
      rows.append(["Chain", qnum, "large", round(data_hard[chain_col[qnum][1]].mean(), 1), round(data_hard[chain_col[qnum][1]].std(), 1), round(data_hard[chain_col[qnum][3]].mean(), 2), round(data_hard[chain_col[qnum][3]].std(), 2), round(data_hard_right[chain_col[qnum][3]].mean(), 2), round(data_hard_right[chain_col[qnum][3]].std(), 2), len(data_hard[data_hard[chain_col[qnum][4]] == True]), round(len(data_hard[data_hard[chain_col[qnum][4]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[chain_col[qnum][5]] == True]), round(len(data_hard[data_hard[chain_col[qnum][5]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[chain_col[qnum][6]] == True]), round(len(data_hard[data_hard[chain_col[qnum][6]] == True]) / len(data_hard) * 100, 2), round(data_hard[chain_col[qnum][11]].mean(), 2), round(data_hard[chain_col[qnum][11]].std(), 2), round(data_hard_right[chain_col[qnum][11]].mean(), 2), round(data_hard_right[chain_col[qnum][11]].std(), 2), len(data_hard[data_hard[chain_col[qnum][2]] == True]), round(len(data_hard[data_hard[chain_col[qnum][2]] == True]) / len(data_hard) * 100, 2), len(data_hard_right[data_hard_right[chain_col[qnum][2]] == True]), round(len(data_hard_right[data_hard_right[chain_col[qnum][2]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[chain_col[qnum][10]] == True]), round(len(data_hard[data_hard[chain_col[qnum][10]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[chain_col[qnum][7]] == True]), round(len(data_hard[data_hard[chain_col[qnum][7]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[chain_col[qnum][8]] == True]), round(len(data_hard[data_hard[chain_col[qnum][8]] == True]) / len(data_hard) * 100, 2)])

   for qnum in range(1,8): #adding common cause

      data_simple_right = data_simple[data_simple[CC_col[qnum][0]] == True]
      data_medium_right = data_medium[data_medium[CC_col[qnum][0]] == True]
      data_hard_right = data_hard[data_hard[CC_col[qnum][0]] == True]

      rows.append(["CC", qnum, "small", round(data_simple[CC_col[qnum][1]].mean(), 1), round(data_simple[CC_col[qnum][1]].std(), 1), round(data_simple[CC_col[qnum][3]].mean(), 2), round(data_simple[CC_col[qnum][3]].std(), 2), round(data_simple_right[CC_col[qnum][3]].mean(), 2), round(data_simple_right[CC_col[qnum][3]].std(), 2), len(data_simple[data_simple[CC_col[qnum][4]] == True]), round(len(data_simple[data_simple[CC_col[qnum][4]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CC_col[qnum][5]] == True]), round(len(data_simple[data_simple[CC_col[qnum][5]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CC_col[qnum][6]] == True]), round(len(data_simple[data_simple[CC_col[qnum][6]] == True]) / len(data_simple) * 100, 2), round(data_simple[CC_col[qnum][11]].mean(), 2), round(data_simple[CC_col[qnum][11]].std(), 2), round(data_simple_right[CC_col[qnum][11]].mean(), 2), round(data_simple_right[CC_col[qnum][11]].std(), 2), len(data_simple[data_simple[CC_col[qnum][2]] == True]), round(len(data_simple[data_simple[CC_col[qnum][2]] == True]) / len(data_simple) * 100, 2), len(data_simple_right[data_simple_right[CC_col[qnum][2]] == True]), round(len(data_simple_right[data_simple_right[CC_col[qnum][2]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CC_col[qnum][10]] == True]), round(len(data_simple[data_simple[CC_col[qnum][10]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CC_col[qnum][7]] == True]), round(len(data_simple[data_simple[CC_col[qnum][7]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CC_col[qnum][8]] == True]), round(len(data_simple[data_simple[CC_col[qnum][8]] == True]) / len(data_simple) * 100, 2)])
      rows.append(["CC", qnum, "medium", round(data_medium[CC_col[qnum][1]].mean(), 1), round(data_medium[CC_col[qnum][1]].std(), 1), round(data_medium[CC_col[qnum][3]].mean(), 2), round(data_medium[CC_col[qnum][3]].std(), 2), round(data_medium_right[CC_col[qnum][3]].mean(), 2), round(data_medium_right[CC_col[qnum][3]].std(), 2), len(data_medium[data_medium[CC_col[qnum][4]] == True]), round(len(data_medium[data_medium[CC_col[qnum][4]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CC_col[qnum][5]] == True]), round(len(data_medium[data_medium[CC_col[qnum][5]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CC_col[qnum][6]] == True]), round(len(data_medium[data_medium[CC_col[qnum][6]] == True]) / len(data_medium) * 100, 2), round(data_medium[CC_col[qnum][11]].mean(), 2), round(data_medium[CC_col[qnum][11]].std(), 2), round(data_medium_right[CC_col[qnum][11]].mean(), 2), round(data_medium_right[CC_col[qnum][11]].std(), 2), len(data_medium[data_medium[CC_col[qnum][2]] == True]), round(len(data_medium[data_medium[CC_col[qnum][2]] == True]) / len(data_medium) * 100, 2), len(data_medium_right[data_medium_right[CC_col[qnum][2]] == True]), round(len(data_medium_right[data_medium_right[CC_col[qnum][2]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CC_col[qnum][10]] == True]), round(len(data_medium[data_medium[CC_col[qnum][10]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CC_col[qnum][7]] == True]), round(len(data_medium[data_medium[CC_col[qnum][7]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CC_col[qnum][8]] == True]), round(len(data_medium[data_medium[CC_col[qnum][8]] == True]) / len(data_medium) * 100, 2)])
      rows.append(["CC", qnum, "large", round(data_hard[CC_col[qnum][1]].mean(), 1), round(data_hard[CC_col[qnum][1]].std(), 1), round(data_hard[CC_col[qnum][3]].mean(), 2), round(data_hard[CC_col[qnum][3]].std(), 2), round(data_hard_right[CC_col[qnum][3]].mean(), 2), round(data_hard_right[CC_col[qnum][3]].std(), 2), len(data_hard[data_hard[CC_col[qnum][4]] == True]), round(len(data_hard[data_hard[CC_col[qnum][4]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CC_col[qnum][5]] == True]), round(len(data_hard[data_hard[CC_col[qnum][5]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CC_col[qnum][6]] == True]), round(len(data_hard[data_hard[CC_col[qnum][6]] == True]) / len(data_hard) * 100, 2), round(data_hard[CC_col[qnum][11]].mean(), 2), round(data_hard[CC_col[qnum][11]].std(), 2), round(data_hard_right[CC_col[qnum][11]].mean(), 2), round(data_hard_right[CC_col[qnum][11]].std(), 2), len(data_hard[data_hard[CC_col[qnum][2]] == True]), round(len(data_hard[data_hard[CC_col[qnum][2]] == True]) / len(data_hard) * 100, 2), len(data_hard_right[data_hard_right[CC_col[qnum][2]] == True]), round(len(data_hard_right[data_hard_right[CC_col[qnum][2]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CC_col[qnum][10]] == True]), round(len(data_hard[data_hard[CC_col[qnum][10]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CC_col[qnum][7]] == True]), round(len(data_hard[data_hard[CC_col[qnum][7]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CC_col[qnum][8]] == True]), round(len(data_hard[data_hard[CC_col[qnum][8]] == True]) / len(data_hard) * 100, 2)])

   for qnum in range(1,8): #adding common effect

      data_simple_right = data_simple[data_simple[CE_col[qnum][0]] == True]
      data_medium_right = data_medium[data_medium[CE_col[qnum][0]] == True]
      data_hard_right = data_hard[data_hard[CE_col[qnum][0]] == True]

      rows.append(["CE", qnum, "small", round(data_simple[CE_col[qnum][1]].mean(), 1), round(data_simple[CE_col[qnum][1]].std(), 1), round(data_simple[CE_col[qnum][3]].mean(), 2), round(data_simple[CE_col[qnum][3]].std(), 2), round(data_simple_right[CE_col[qnum][3]].mean(), 2), round(data_simple_right[CE_col[qnum][3]].std(), 2), len(data_simple[data_simple[CE_col[qnum][4]] == True]), round(len(data_simple[data_simple[CE_col[qnum][4]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CE_col[qnum][5]] == True]), round(len(data_simple[data_simple[CE_col[qnum][5]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CE_col[qnum][6]] == True]), round(len(data_simple[data_simple[CE_col[qnum][6]] == True]) / len(data_simple) * 100, 2), round(data_simple[CE_col[qnum][11]].mean(), 2), round(data_simple[CE_col[qnum][11]].std(), 2), round(data_simple_right[CE_col[qnum][11]].mean(), 2), round(data_simple_right[CE_col[qnum][11]].std(), 2), len(data_simple[data_simple[CE_col[qnum][2]] == True]), round(len(data_simple[data_simple[CE_col[qnum][2]] == True]) / len(data_simple) * 100, 2), len(data_simple_right[data_simple_right[CE_col[qnum][2]] == True]), round(len(data_simple_right[data_simple_right[CE_col[qnum][2]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CE_col[qnum][10]] == True]), round(len(data_simple[data_simple[CE_col[qnum][10]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CE_col[qnum][7]] == True]), round(len(data_simple[data_simple[CE_col[qnum][7]] == True]) / len(data_simple) * 100, 2), len(data_simple[data_simple[CE_col[qnum][8]] == True]), round(len(data_simple[data_simple[CE_col[qnum][8]] == True]) / len(data_simple) * 100, 2)])
      rows.append(["CE", qnum, "medium", round(data_medium[CE_col[qnum][1]].mean(), 1), round(data_medium[CE_col[qnum][1]].std(), 1), round(data_medium[CE_col[qnum][3]].mean(), 2), round(data_medium[CE_col[qnum][3]].std(), 2), round(data_medium_right[CE_col[qnum][3]].mean(), 2), round(data_medium_right[CE_col[qnum][3]].std(), 2), len(data_medium[data_medium[CE_col[qnum][4]] == True]), round(len(data_medium[data_medium[CE_col[qnum][4]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CE_col[qnum][5]] == True]), round(len(data_medium[data_medium[CE_col[qnum][5]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CE_col[qnum][6]] == True]), round(len(data_medium[data_medium[CE_col[qnum][6]] == True]) / len(data_medium) * 100, 2), round(data_medium[CE_col[qnum][11]].mean(), 2), round(data_medium[CE_col[qnum][11]].std(), 2), round(data_medium_right[CE_col[qnum][11]].mean(), 2), round(data_medium_right[CE_col[qnum][11]].std(), 2), len(data_medium[data_medium[CE_col[qnum][2]] == True]), round(len(data_medium[data_medium[CE_col[qnum][2]] == True]) / len(data_medium) * 100, 2), len(data_medium_right[data_medium_right[CE_col[qnum][2]] == True]), round(len(data_medium_right[data_medium_right[CE_col[qnum][2]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CE_col[qnum][10]] == True]), round(len(data_medium[data_medium[CE_col[qnum][10]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CE_col[qnum][7]] == True]), round(len(data_medium[data_medium[CE_col[qnum][7]] == True]) / len(data_medium) * 100, 2), len(data_medium[data_medium[CE_col[qnum][8]] == True]), round(len(data_medium[data_medium[CE_col[qnum][8]] == True]) / len(data_medium) * 100, 2)])
      rows.append(["CE", qnum, "large", round(data_hard[CE_col[qnum][1]].mean(), 1), round(data_hard[CE_col[qnum][1]].std(), 1), round(data_hard[CE_col[qnum][3]].mean(), 2), round(data_hard[CE_col[qnum][3]].std(), 2), round(data_hard_right[CE_col[qnum][3]].mean(), 2), round(data_hard_right[CE_col[qnum][3]].std(), 2), len(data_hard[data_hard[CE_col[qnum][4]] == True]), round(len(data_hard[data_hard[CE_col[qnum][4]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CE_col[qnum][5]] == True]), round(len(data_hard[data_hard[CE_col[qnum][5]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CE_col[qnum][6]] == True]), round(len(data_hard[data_hard[CE_col[qnum][6]] == True]) / len(data_hard) * 100, 2), round(data_hard[CE_col[qnum][11]].mean(), 2), round(data_hard[CE_col[qnum][11]].std(), 2), round(data_hard_right[CE_col[qnum][11]].mean(), 2), round(data_hard_right[CE_col[qnum][11]].std(), 2), len(data_hard[data_hard[CE_col[qnum][2]] == True]), round(len(data_hard[data_hard[CE_col[qnum][2]] == True]) / len(data_hard) * 100, 2), len(data_hard_right[data_hard_right[CE_col[qnum][2]] == True]), round(len(data_hard_right[data_hard_right[CE_col[qnum][2]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CE_col[qnum][10]] == True]), round(len(data_hard[data_hard[CE_col[qnum][10]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CE_col[qnum][7]] == True]), round(len(data_hard[data_hard[CE_col[qnum][7]] == True]) / len(data_hard) * 100, 2), len(data_hard[data_hard[CE_col[qnum][8]] == True]), round(len(data_hard[data_hard[CE_col[qnum][8]] == True]) / len(data_hard) * 100, 2)])


   table = pd.DataFrame(rows)
   table.to_csv("Summary0302_fixed.csv", header = False, index = False)

arrange_csv(data)
#correct_rate(data)