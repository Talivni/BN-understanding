import numpy as np
import pandas as pd 
import re
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\talil\\OneDrive\\Documents\\GitHub\\BN-understanding\\raw_data\\raw_0202.csv")

headers = ["Participant_ID", "Cohort_Type", "Duration_Minutes", "Data_In_Future_Research", "Future_Experiments_Participation",
           "Experience", "APIs", "Years", "Used_BN_in_research?", "Last_year", "Gender", "Eval_Correct",  "Eval_Graph_Correct",
           "Chain_Q1_InitValue", "Chain_Q1_CorrectAnswer", "Chain_Q1_CorrectDirection", "Chain_Q1_P_Lowest", "Chain_Q1_P_Highest", "Chain_Q1_P_BestEstimate", "Chain_Q1_Distance", "Chain_Q1_Distance_norm", "Chain_Q1_P_Range",  "Chain_Q1_P_AnsWithinRange", "Chain_Q1_Cond1_P_Increased", "Chain_Q1_Cond1_P_Decreased",
           "Chain_Q1_Cond1_P_The_Same", "Chain_Q1_Cond2_P", "Chain_Q1_Cond3_P", "Chain_Q1_Cond4_CorrectPerc", "Chain_Q1_Cond4_P", "Chain_Q1_ReadCPT", 
           "Chain_Q2_InitValue", "Chain_Q2_CorrectAnswer", "Chain_Q2_CorrectDirection", "Chain_Q2_P_Lowest", "Chain_Q2_P_Highest", "Chain_Q2_P_BestEstimate", "Chain_Q2_Distance", "Chain_Q2_Distance_norm", "Chain_Q2_P_Range",  "Chain_Q2_P_AnsWithinRange", "Chain_Q2_Cond1_P_Increased", "Chain_Q2_Cond1_P_Decreased",
           "Chain_Q2_Cond1_P_The_Same", "Chain_Q2_Cond2_P", "Chain_Q2_Cond3_P", "Chain_Q2_Cond4_CorrectPerc", "Chain_Q2_Cond4_P", "Chain_Q2_ReadCPT",
           "Chain_Q3_InitValue", "Chain_Q3_CorrectAnswer", "Chain_Q3_CorrectDirection", "Chain_Q3_P_Lowest", "Chain_Q3_P_Highest", "Chain_Q3_P_BestEstimate", "Chain_Q3_Distance", "Chain_Q3_Distance_norm", "Chain_Q3_P_Range",  "Chain_Q3_P_AnsWithinRange", "Chain_Q3_Cond1_P_Increased", "Chain_Q3_Cond1_P_Decreased",
           "Chain_Q3_Cond1_P_The_Same", "Chain_Q3_Cond2_P", "Chain_Q3_Cond3_P", "Chain_Q3_Cond4_CorrectPerc", "Chain_Q3_Cond4_P", "Chain_Q3_ReadCPT",
           "Chain_Q4_InitValue", "Chain_Q4_CorrectAnswer", "Chain_Q4_CorrectDirection", "Chain_Q4_P_Lowest", "Chain_Q4_P_Highest", "Chain_Q4_P_BestEstimate", "Chain_Q4_Distance", "Chain_Q4_Distance_norm", "Chain_Q4_P_Range",  "Chain_Q4_P_AnsWithinRange", "Chain_Q4_Cond1_P_Increased", "Chain_Q4_Cond1_P_Decreased",
           "Chain_Q4_Cond1_P_The_Same", "Chain_Q4_Cond2_P", "Chain_Q4_Cond3_P", "Chain_Q4_Cond4_CorrectPerc", "Chain_Q4_Cond4_P", "Chain_Q4_ReadCPT",
           "Chain_Q5_InitValue", "Chain_Q5_CorrectAnswer", "Chain_Q5_CorrectDirection", "Chain_Q5_P_Lowest", "Chain_Q5_P_Highest", "Chain_Q5_P_BestEstimate", "Chain_Q5_Distance", "Chain_Q5_Distance_norm", "Chain_Q5_P_Range",  "Chain_Q5_P_AnsWithinRange", "Chain_Q5_Cond1_P_Increased", "Chain_Q5_Cond1_P_Decreased",
           "Chain_Q5_Cond1_P_The_Same", "Chain_Q5_Cond2_P", "Chain_Q5_Cond3_P", "Chain_Q5_Cond4_CorrectPerc", "Chain_Q5_Cond4_P", "Chain_Q5_ReadCPT",
           "CC_Q1_InitValue", "CC_Q1_CorrectAnswer", "CC_Q1_CorrectDirection", "CC_Q1_P_Lowest", "CC_Q1_P_Highest", "CC_Q1_P_BestEstimate", "CC_Q1_Distance", "CC_Q1_Distance_norm", "CC_Q1_P_Range",  "CC_Q1_P_AnsWithinRange", "CC_Q1_Cond1_P_Increased", "CC_Q1_Cond1_P_Decreased",
           "CC_Q1_Cond1_P_The_Same", "CC_Q1_Cond2_P", "CC_Q1_Cond3_P", "CC_Q1_Cond4_CorrectPerc", "CC_Q1_Cond4_P", "CC_Q1_ReadCPT",
           "CC_Q2_InitValue", "CC_Q2_CorrectAnswer", "CC_Q2_CorrectDirection", "CC_Q2_P_Lowest", "CC_Q2_P_Highest", "CC_Q2_P_BestEstimate", "CC_Q2_Distance", "CC_Q2_Distance_norm", "CC_Q2_P_Range",  "CC_Q2_P_AnsWithinRange", "CC_Q2_Cond1_P_Increased", "CC_Q2_Cond1_P_Decreased",
           "CC_Q2_Cond1_P_The_Same", "CC_Q2_Cond2_P", "CC_Q2_Cond3_P", "CC_Q2_Cond4_CorrectPerc", "CC_Q2_Cond4_P", "CC_Q2_ReadCPT",
           "CC_Q3_InitValue", "CC_Q3_CorrectAnswer", "CC_Q3_CorrectDirection", "CC_Q3_P_Lowest", "CC_Q3_P_Highest", "CC_Q3_P_BestEstimate", "CC_Q3_Distance", "CC_Q3_Distance_norm", "CC_Q3_P_Range",  "CC_Q3_P_AnsWithinRange", "CC_Q3_Cond1_P_Increased", "CC_Q3_Cond1_P_Decreased",
           "CC_Q3_Cond1_P_The_Same", "CC_Q3_Cond2_P", "CC_Q3_Cond3_P", "CC_Q3_Cond4_CorrectPerc", "CC_Q3_Cond4_P", "CC_Q3_ReadCPT",
           "CC_Q4_InitValue", "CC_Q4_CorrectAnswer", "CC_Q4_CorrectDirection", "CC_Q4_P_Lowest", "CC_Q4_P_Highest", "CC_Q4_P_BestEstimate", "CC_Q4_Distance", "CC_Q4_Distance_norm", "CC_Q4_P_Range",  "CC_Q4_P_AnsWithinRange", "CC_Q4_Cond1_P_Increased", "CC_Q4_Cond1_P_Decreased",
           "CC_Q4_Cond1_P_The_Same", "CC_Q4_Cond2_P", "CC_Q4_Cond3_P", "CC_Q4_Cond4_CorrectPerc", "CC_Q4_Cond4_P", "CC_Q4_ReadCPT",
           "CC_Q5_InitValue", "CC_Q5_CorrectAnswer", "CC_Q5_CorrectDirection", "CC_Q5_P_Lowest", "CC_Q5_P_Highest", "CC_Q5_P_BestEstimate", "CC_Q5_Distance", "CC_Q5_Distance_norm", "CC_Q5_P_Range",  "CC_Q5_P_AnsWithinRange", "CC_Q5_Cond1_P_Increased", "CC_Q5_Cond1_P_Decreased",
           "CC_Q5_Cond1_P_The_Same", "CC_Q5_Cond2_P", "CC_Q5_Cond3_P", "CC_Q5_Cond4_CorrectPerc", "CC_Q5_Cond4_P", "CC_Q5_ReadCPT",
           "CC_Q6_InitValue", "CC_Q6_CorrectAnswer", "CC_Q6_CorrectDirection", "CC_Q6_P_Lowest", "CC_Q6_P_Highest", "CC_Q6_P_BestEstimate", "CC_Q6_Distance", "CC_Q6_Distance_norm", "CC_Q6_P_Range",  "CC_Q6_P_AnsWithinRange", "CC_Q6_Cond1_P_Increased", "CC_Q6_Cond1_P_Decreased",
           "CC_Q6_Cond1_P_The_Same", "CC_Q6_Cond2_P", "CC_Q6_Cond3_P", "CC_Q6_Cond4_CorrectPerc", "CC_Q6_Cond4_P", "CC_Q6_ReadCPT",
           "CC_Q7_InitValue", "CC_Q7_CorrectAnswer", "CC_Q7_CorrectDirection", "CC_Q7_P_Lowest", "CC_Q7_P_Highest", "CC_Q7_P_BestEstimate", "CC_Q7_Distance", "CC_Q7_Distance_norm", "CC_Q7_P_Range",  "CC_Q7_P_AnsWithinRange", "CC_Q7_Cond1_P_Increased", "CC_Q7_Cond1_P_Decreased",
           "CC_Q7_Cond1_P_The_Same", "CC_Q7_Cond2_P", "CC_Q7_Cond3_P", "CC_Q7_Cond4_CorrectPerc", "CC_Q7_Cond4_P", "CC_Q7_ReadCPT",
           "CE_Q1_InitValue", "CE_Q1_CorrectAnswer", "CE_Q1_CorrectDirection", "CE_Q1_P_Lowest", "CE_Q1_P_Highest", "CE_Q1_P_BestEstimate", "CE_Q1_Distance", "CE_Q1_Distance_norm", "CE_Q1_P_Range",  "CE_Q1_P_AnsWithinRange", "CE_Q1_Cond1_P_Increased", "CE_Q1_Cond1_P_Decreased",
           "CE_Q1_Cond1_P_The_Same", "CE_Q1_Cond2_P", "CE_Q1_Cond3_P", "CE_Q1_Cond4_CorrectPerc", "CE_Q1_Cond4_P", "CE_Q1_ReadCPT",
           "CE_Q2_InitValue", "CE_Q2_CorrectAnswer", "CE_Q2_CorrectDirection", "CE_Q2_P_Lowest", "CE_Q2_P_Highest", "CE_Q2_P_BestEstimate", "CE_Q2_Distance", "CE_Q2_Distance_norm", "CE_Q2_P_Range",  "CE_Q2_P_AnsWithinRange", "CE_Q2_Cond1_P_Increased", "CE_Q2_Cond1_P_Decreased",
           "CE_Q2_Cond1_P_The_Same", "CE_Q2_Cond2_P", "CE_Q2_Cond3_P", "CE_Q2_Cond4_CorrectPerc", "CE_Q2_Cond4_P", "CE_Q2_ReadCPT",
           "CE_Q3_InitValue", "CE_Q3_CorrectAnswer", "CE_Q3_CorrectDirection", "CE_Q3_P_Lowest", "CE_Q3_P_Highest", "CE_Q3_P_BestEstimate", "CE_Q3_Distance", "CE_Q3_Distance_norm", "CE_Q3_P_Range",  "CE_Q3_P_AnsWithinRange", "CE_Q3_Cond1_P_Increased", "CE_Q3_Cond1_P_Decreased",
           "CE_Q3_Cond1_P_The_Same", "CE_Q3_Cond2_P", "CE_Q3_Cond3_P", "CE_Q3_Cond4_CorrectPerc", "CE_Q3_Cond4_P", "CE_Q3_ReadCPT",
           "CE_Q4_InitValue", "CE_Q4_CorrectAnswer", "CE_Q4_CorrectDirection", "CE_Q4_P_Lowest", "CE_Q4_P_Highest", "CE_Q4_P_BestEstimate", "CE_Q4_Distance", "CE_Q4_Distance_norm", "CE_Q4_P_Range",  "CE_Q4_P_AnsWithinRange", "CE_Q4_Cond1_P_Increased", "CE_Q4_Cond1_P_Decreased",
           "CE_Q4_Cond1_P_The_Same", "CE_Q4_Cond2_P", "CE_Q4_Cond3_P", "CE_Q4_Cond4_CorrectPerc", "CE_Q4_Cond4_P", "CE_Q4_ReadCPT",
           "CE_Q5_InitValue", "CE_Q5_CorrectAnswer", "CE_Q5_CorrectDirection", "CE_Q5_P_Lowest", "CE_Q5_P_Highest", "CE_Q5_P_BestEstimate", "CE_Q5_Distance", "CE_Q5_Distance_norm", "CE_Q5_P_Range",  "CE_Q5_P_AnsWithinRange", "CE_Q5_Cond1_P_Increased", "CE_Q5_Cond1_P_Decreased",
           "CE_Q5_Cond1_P_The_Same", "CE_Q5_Cond2_P", "CE_Q5_Cond3_P", "CE_Q5_Cond4_CorrectPerc", "CE_Q5_Cond4_P", "CE_Q5_ReadCPT",
           "CE_Q6_InitValue", "CE_Q6_CorrectAnswer", "CE_Q6_CorrectDirection", "CE_Q6_P_Lowest", "CE_Q6_P_Highest", "CE_Q6_P_BestEstimate", "CE_Q6_Distance", "CE_Q6_Distance_norm", "CE_Q6_P_Range",  "CE_Q6_P_AnsWithinRange", "CE_Q6_Cond1_P_Increased", "CE_Q6_Cond1_P_Decreased",
           "CE_Q6_Cond1_P_The_Same", "CE_Q6_Cond2_P", "CE_Q6_Cond3_P", "CE_Q6_Cond4_CorrectPerc", "CE_Q6_Cond4_P", "CE_Q6_ReadCPT",
           "CE_Q7_InitValue", "CE_Q7_CorrectAnswer", "CE_Q7_CorrectDirection", "CE_Q7_P_Lowest", "CE_Q7_P_Highest", "CE_Q7_P_BestEstimate", "CE_Q7_Distance", "CE_Q7_Distance_norm", "CE_Q7_P_Range",  "CE_Q7_P_AnsWithinRange", "CE_Q7_Cond1_P_Increased", "CE_Q7_Cond1_P_Decreased",
           "CE_Q7_Cond1_P_The_Same", "CE_Q7_Cond2_P", "CE_Q7_Cond3_P", "CE_Q7_Cond4_CorrectPerc", "CE_Q7_Cond4_P", "CE_Q7_ReadCPT", "Score_readCPT", "Score_Directionality", "avg_task_load"]

eval_col = {1: 'eval_1', 2: 'eval_2', 3: 'eval_3', 4: 'eval_4', 5: 'eval_5', 6: 'eval6_1', 7: 'eval6_2', 8: 'eval_7'}

eval_sol = {1: '8)', 2: '7)', 3: '2)', 4: '(0.07 * 0.1)/0.05 = 0.14', 5: '(0.08 * 0.1)/0.05 = 0.16', 6: 60, 7: 70, 8: 88}

chain_col = {('simple', 1): ['Chain_Simple_Q1_16', 'Chain_Simple_Q1_17', 'Chain_Simple_Q1_18'], 
             ('simple', 2): ['Chain_Simple_Q2_16', 'Chain_Simple_Q2_17', 'Chain_Simple_Q2_18'],
             ('simple', 3): ['Chain_Simple_Q3_16', 'Chain_Simple_Q3_17', 'Chain_Simple_Q3_18'],
             ('simple', 4): ['Chain_Simple_Q4_16', 'Chain_Simple_Q4_17', 'Chain_Simple_Q4_18'],
             ('simple', 5): ['Chain_Simple_Q5_16', 'Chain_Simple_Q5_17', 'Chain_Simple_Q5_18'],
             ('medium', 1): ['Chain_Medium_Q1_16', 'Chain_Medium_Q1_17', 'Chain_Medium_Q1_18'],
             ('medium', 2): ['Chain_Medium_Q2_16', 'Chain_Medium_Q2_17', 'Chain_Medium_Q2_18'],
             ('medium', 3): ['Chain_Medium_Q3_16', 'Chain_Medium_Q3_17', 'Chain_Medium_Q3_18'],
             ('medium', 4): ['Chain_Medium_Q4_16', 'Chain_Medium_Q4_17', 'Chain_Medium_Q4_18'],
             ('medium', 5): ['Chain_Medium_Q5_16', 'Chain_Medium_Q5_17', 'Chain_Medium_Q5_18'],
             ('hard', 1): ['Chain_Hard_Q1_16', 'Chain_Hard_Q1_17', 'Chain_Hard_Q1_18'],
             ('hard', 2): ['Chain_Hard_Q2_16', 'Chain_Hard_Q2_17', 'Chain_Hard_Q2_18'],
             ('hard', 3): ['Chain_Hard_Q3_16', 'Chain_Hard_Q3_17', 'Chain_Hard_Q3_18'],
             ('hard', 4): ['Chain_Hard_Q4_16', 'Chain_Hard_Q4_17', 'Chain_Hard_Q4_18'],
             ('hard', 5): ['Chain_Hard_Q5_16', 'Chain_Hard_Q5_17', 'Chain_Hard_Q5_18']}

CC_col = {('simple', 1): ['CC_Simple_Q1_19', 'CC_Simple_Q1_17', 'CC_Simple_Q1_18'],
             ('simple', 2): ['CC_Simple_Q2_16', 'CC_Simple_Q2_17', 'CC_Simple_Q2_18'],
             ('simple', 3): ['CC_Simple_Q3_16', 'CC_Simple_Q3_17', 'CC_Simple_Q3_18'],
             ('simple', 4): ['CC_Simple_Q4_16', 'CC_Simple_Q4_17', 'CC_Simple_Q4_18'],
             ('simple', 5): ['CC_Simple_Q5_16', 'CC_Simple_Q5_17', 'CC_Simple_Q5_18'],
             ('simple', 6): ['CC_Simple_Q6_16', 'CC_Simple_Q6_17', 'CC_Simple_Q6_18'],
             ('simple', 7): ['CC_Simple_Q7_16', 'CC_Simple_Q7_17', 'CC_Simple_Q7_18'],
             ('medium', 1): ['CC_Medium_Q1_16', 'CC_Medium_Q1_17', 'CC_Medium_Q1_18'],
             ('medium', 2): ['CC_Medium_Q2_16', 'CC_Medium_Q2_17', 'CC_Medium_Q2_18'],
             ('medium', 3): ['CC_Medium_Q3_16', 'CC_Medium_Q3_17', 'CC_Medium_Q3_18'],
             ('medium', 4): ['CC_Medium_Q4_16', 'CC_Medium_Q4_17', 'CC_Medium_Q4_18'],
             ('medium', 5): ['CC_Medium_Q5_16', 'CC_Medium_Q5_17', 'CC_Medium_Q5_18'],
             ('medium', 6): ['CC_Medium_Q6_16', 'CC_Medium_Q6_17', 'CC_Medium_Q6_18'],
             ('medium', 7): ['CC_Medium_Q7_16', 'CC_Medium_Q7_17', 'CC_Medium_Q7_18'],
             ('hard', 1): ['CC_Hard_Q1_16', 'CC_Hard_Q1_17', 'CC_Hard_Q1_18'],
             ('hard', 2): ['CC_Hard_Q2_16', 'CC_Hard_Q2_17', 'CC_Hard_Q2_18'],
             ('hard', 3): ['CC_Hard_Q3_16', 'CC_Hard_Q3_17', 'CC_Hard_Q3_18'],
             ('hard', 4): ['CC_Hard_Q4_16', 'CC_Hard_Q4_17', 'CC_Hard_Q4_18'],
             ('hard', 5): ['CC_Hard_Q5_16', 'CC_Hard_Q5_17', 'CC_Hard_Q5_18'],
             ('hard', 6): ['CC_Hard_Q6_16', 'CC_Hard_Q6_17', 'CC_Hard_Q6_18'],
             ('hard', 7): ['CC_Hard_Q7_16', 'CC_Hard_Q7_17', 'CC_Hard_Q7_18']}

CE_col = {('simple', 1): ['CE_Simple_Q1_16', 'CE_Simple_Q1_17' ,'CE_Simple_Q1_18'],
             ('simple', 2): ['CE_Simple_Q2_16', 'CE_Simple_Q2_17', 'CE_Simple_Q2_18'],
             ('simple', 3): ['CE_Simple_Q3_16', 'CE_Simple_Q3_17', 'CE_Simple_Q3_18'],
             ('simple', 4): ['CE_Simple_Q4_16', 'CE_Simple_Q4_17', 'CE_Simple_Q4_18'],
             ('simple', 5): ['CE_Simple_Q5_16', 'CE_Simple_Q5_17', 'CE_Simple_Q5_18'],
             ('simple', 6): ['CE_Simple_Q6_16', 'CE_Simple_Q6_17', 'CE_Simple_Q6_18'],
             ('simple', 7): ['CE_Simple_Q7_16', 'CE_Simple_Q7_17', 'CE_Simple_Q7_18'],
             ('medium', 1): ['CE_Medium_Q1_16', 'CE_Medium_Q1_17', 'CE_Medium_Q1_18'],
             ('medium', 2): ['CE_Medium_Q2_16', 'CE_Medium_Q2_17', 'CE_Medium_Q2_18'],
             ('medium', 3): ['CE_Medium_Q3_16', 'CE_Medium_Q3_17', 'CE_Medium_Q3_18'],
             ('medium', 4): ['CE_Medium_Q4_16', 'CE_Medium_Q4_17', 'CE_Medium_Q4_18'],
             ('medium', 5): ['CE_Medium_Q5_16', 'CE_Medium_Q5_17', 'CE_Medium_Q5_18'],
             ('medium', 6): ['CE_Medium_Q6_16', 'CE_Medium_Q6_17', 'CE_Medium_Q6_18'],
             ('medium', 7): ['CE_Medium_Q7_16', 'CE_Medium_Q7_17', 'CE_Medium_Q7_18'],
             ('hard', 1): ['CE_Hard_Q1_16', 'CE_Hard_Q1_17', 'CE_Hard_Q1_18'],
             ('hard', 2): ['CE_Hard_Q2_16', 'CE_Hard_Q2_17', 'CE_Hard_Q2_18'],
             ('hard', 3): ['CE_Hard_Q3_16', 'CE_Hard_Q3_17', 'CE_Hard_Q3_18'],
             ('hard', 4): ['CE_Hard_Q4_16', 'CE_Hard_Q4_17', 'CE_Hard_Q4_18'],
             ('hard', 5): ['CE_Hard_Q5_16', 'CE_Hard_Q5_17', 'CE_Hard_Q5_18'],
             ('hard', 6): ['CE_Hard_Q6_16', 'CE_Hard_Q6_17', 'CE_Hard_Q6_18'],
             ('hard', 7): ['CE_Hard_Q7_16', 'CE_Hard_Q7_17', 'CE_Hard_Q7_18']}
             

chain = {('simple', 1): [38, 10, 'Decreased', 'N/A', 'N/A', 73.7, True],
         ('simple', 2): [40.4, 18, 'Decreased', 'N/A', 'N/A', 55.4, 'N/A'],
         ('simple', 3): [18, 90, 'Increased', 'N/A', 'N/A', 400, True],
         ('simple', 4): [40.4, 90, 'Increased', True, 'N/A', 122.8, True],
         ('simple', 5): [90, 90, 'Remained the Same', 'N/A', 'N/A', 0, 'N/A'],
         ('medium', 1): [40.4, 18, 'Decreased', 'N/A', 'N/A', 55.4, 'N/A'],
         ('medium', 2): [43.1, 38.6, 'Decreased', 'N/A', 'N/A', 10.4, 'N/A'],
         ('medium', 3): [38.6, 55, 'Increased', 'N/A', 'N/A', 42.5, 'N/A'],
         ('medium', 4): [43.1, 55, 'Increased', True, 'N/A', 27.6, 'N/A'],
         ('medium', 5): [55, 55, 'Remained the Same', 'N/A', 'N/A', 0, 'N/A'],
         ('hard', 1): [43.1, 38.6, 'Decreased', 'N/A', 'N/A', 10.4, 'N/A'],
         ('hard', 2): [61.8, 61.2, 'Decreased', 'N/A', 'N/A', 1, 'N/A'],
         ('hard', 3): [61.2, 70, 'Increased', 'N/A', 'N/A', 14.4, 'N/A'],
         ('hard', 4): [61.8, 70, 'Increased', True, 'N/A', 13.3, 'N/A'],
         ('hard', 5): [70, 70, 'Remained the Same', 'N/A', 'N/A', 0, 'N/A']}

CC = {('simple', 1): [20, 52.9, 'Increased', 'N/A', 'N/A', 164.5, 'N/A'],
      ('simple', 2): [20, 63.6, 'Increased', 'N/A', True, 218, 'N/A'],
      ('simple', 3): [63.6, 88.7, 'Increased', 'N/A', 'N/A', 39.5, 'N/A'],
      ('simple', 4): [22, 10, 'Decreased', 'N/A', 'N/A', 54.5, True],
      ('simple', 5): [22, 70, 'Increased', 'N/A', 'N/A', 218.2, True],
      ('simple', 6): [70, 70, 'Remained the Same', True, 'N/A', 0, 'N/A'],
      ('simple', 7): [41.8, 70, 'Increased', True, 'N/A', 67.5, True],
      ('medium', 1): [17, 33.3, 'Increased', 'N/A', 'N/A', 95.9, 'N/A'],
      ('medium', 2): [17, 40, 'Increased', 'N/A', True, 135.3, 'N/A'],
      ('medium', 3): [40, 61.9, 'Increased', 'N/A', 'N/A', 54.8, 'N/A'],
      ('medium', 4): [22.1, 16, 'Decreased', 'N/A', 'N/A', 27.6, 'N/A'],
      ('medium', 5): [22.1, 52, 'Increased', 'N/A', 'N/A', 135.3, 'N/A'],
      ('medium', 6): [52, 52, 'Remained the Same', True, 'N/A', 0, 'N/A'],
      ('medium', 7): [28, 52, 'Increased', True, 'N/A',85.7, 'N/A'],
      ('hard', 1): [17, 23.2, 'Increased', 'N/A', 'N/A', 36.5, 'N/A'],
      ('hard', 2): [17, 24.6, 'Increased', 'N/A', True, 44.7, 'N/A'],
      ('hard', 3): [24.6, 32.5, 'Increased', 'N/A', 'N/A', 32.1, 'N/A'],
      ('hard', 4): [24, 21.8, 'Decreased', 'N/A', 'N/A', 9.2, 'N/A'],
      ('hard', 5): [24, 34.7, 'Increased', 'N/A', 'N/A', 44.6, 'N/A'],
      ('hard', 6): [34.7, 34.7, 'Remained the Same', True, 'N/A', 0, 'N/A'],
      ('hard', 7): [24.8, 34.7, 'Increased', True, 'N/A', 39.9, 'N/A']}

CE = {('simple', 1): [20, 20, 'Remained the Same', 'N/A', 'N/A', 0, 'N/A'],
      ('simple', 2): [32, 88, 'Increased', 'N/A', 'N/A', 175, 'N/A'],
      ('simple', 3): [32, 34, 'Increased', 'N/A', True, 6.3, 'N/A'],
      ('simple', 4): [34, 90, 'Increased', 'N/A', True, 164.7, 'N/A'],
      ('simple', 5): [20, 3.53, 'Decreased', 'N/A', 'N/A', 82.4, 'N/A'],
      ('simple', 6): [20, 55, 'Increased', 'N/A', 'N/A', 175, 'N/A'],
      ('simple', 7): [55, 66.7, 'Increased', 'N/A', 'N/A', 21.3, 'N/A'],
      ('medium', 1): [20, 20, 'Remained the Same', 'N/A', 'N/A', 0, 'N/A'],
      ('medium', 2): [37, 65, 'Increased', 'N/A', 'N/A', 75.7, 'N/A'],
      ('medium', 3): [37, 38, 'Increased', 'N/A', True, 2.7, 'N/A'],
      ('medium', 4): [38, 66, 'Increased', 'N/A', True, 73.7, 'N/A'],
      ('medium', 5): [20, 11.1, 'Decreased', 'N/A', 'N/A', 44.5, 'N/A'],
      ('medium', 6): [20, 35.1, 'Increased', 'N/A', 'N/A', 75.5, 'N/A'],
      ('medium', 7): [35.1, 37, 'Increased', 'N/A', 'N/A', 5.4, 'N/A'],
      ('hard', 1): [20, 20, 'Remained the Same', 'N/A', 'N/A', 0, 'N/A'],
      ('hard', 2): [46.1, 54.5, 'Increased', 'N/A', 'N/A', 18.2, 'N/A'],
      ('hard', 3): [46.1, 46.4, 'Increased', 'N/A', True, 0.7, 'N/A'],
      ('hard', 4): [46.4, 54.8, 'Increased', 'N/A', True, 18.1, 'N/A'],
      ('hard', 5): [20, 16.9, 'Decreased', 'N/A', 'N/A', 15.5, 'N/A'],
      ('hard', 6): [20, 23.6, 'Increased', 'N/A', 'N/A', 18, 'N/A'],
      ('hard', 7): [23.6, 23.7, 'Increased', 'N/A', 'N/A', 0.4, 'N/A']}


##data = data.query("Finished == True & Status != 'Survey Preview'")
##print(data.head(10))
##
##"""
##Prints the result of each participant in each game and whether it's true or not. 
##"""
##def correct_rate(data):
##   for i in range(len(data)):
##      row = data.iloc[i, : ]
##      case = str(row['cohort_num'])
##      print(row['Prolific_ver'], " (",case,")")
##      for qnum in range(1, 11):
##          answer = [int(row[cohort_game_col[(case, qnum)][i]]) for i in range(8)]
##          print(qnum, ": ", answer, sol_game[qnum] == answer)

"""
This function Arranges and writes the data into a CSV in the following format:
TBD
"""
def arrange_csv(data):
   rows = []
   rows.append(headers)

   for i in range(len(data)):
      row = data.iloc[i, :]
      cohort = str(row['cohortType'])

      eval_results = 0 #evaluation questions success
      eval_graph_results = 0
      score_directionality = 0
      score_readCPT = 0

      for qnum in range(1, 9):
         if row[eval_col[qnum]] == eval_sol[qnum]:
            eval_results += 1

      for qnum in range(6, 9):
         if row[eval_col[qnum]] == eval_sol[qnum]:
            eval_graph_results += 1

      new_row = [row['participant_ID'], row['cohortType'], row['Duration (in seconds)'] / 60, row['consent2'], row['consent3'], row['experience_1'], row['experience_2'], row['experience_3'], row['experience_4'], row['experience_5'], row['gender'], eval_results, eval_graph_results]

      for qnum in range(1, 6): #adding chain

         correct_answer = chain[(cohort, qnum)]
         new_row = new_row + [correct_answer[0], correct_answer[1], correct_answer[2]]
         p_best_estimate = row[chain_col[(cohort, qnum)][2]]

         lowest_highest = [row[chain_col[(cohort, qnum)][1]], row[chain_col[(cohort, qnum)][0]]] 
         lowest_highest.sort()
         new_row = new_row + lowest_highest
         new_row.append(p_best_estimate)
         new_row.append(abs(p_best_estimate - correct_answer[1]))
         if ((correct_answer[1] - correct_answer[0]) != 0):
            new_row.append((abs(p_best_estimate - correct_answer[1])) / (abs(correct_answer[1] - correct_answer[0])))
         else:
            new_row.append(abs(p_best_estimate - correct_answer[1]))

         new_row.append(abs(row[chain_col[(cohort, qnum)][1]] - row[chain_col[(cohort, qnum)][0]])) #width
         new_row.append((correct_answer[1] >= lowest_highest[0]) and (correct_answer[1] <= lowest_highest[1])) #is between lowest and highest?
         
            
         if correct_answer[0] < p_best_estimate:
            new_row = new_row + [True, False, False]
            if correct_answer[2] == 'Increased':
               score_directionality += 1
         elif correct_answer[0] > p_best_estimate:
            new_row = new_row + [False, True, False]
            if correct_answer[2] == 'Decreased':
               score_directionality += 1
         else:
            new_row = new_row + [False, False, True]
            if correct_answer[2] == 'Remained the Same':
               score_directionality += 1

         if correct_answer[3] == 'N/A' :
            new_row = new_row + ['N/A', 'N/A']
         else :
           new_row = new_row + [p_best_estimate == row[chain_col[(cohort, qnum - 1)][2]], 'N/A']

         new_row.append(correct_answer[5])
         new_row.append(abs(correct_answer[0] - p_best_estimate) / correct_answer[0] * 100)

         if correct_answer[6] == 'N/A' : #read off CPT
            new_row.append('N/A')
         else :
            new_row.append(p_best_estimate == correct_answer[1])
            if (p_best_estimate == correct_answer[1]):
               score_readCPT += 1


      for qnum in range(1, 8): #adding common cause

         correct_answer = CC[(cohort, qnum)]
         new_row = new_row + [correct_answer[0], correct_answer[1], correct_answer[2]]
         p_best_estimate = row[CC_col[(cohort, qnum)][2]]

         lowest_highest = [row[CC_col[(cohort, qnum)][1]], row[CC_col[(cohort, qnum)][0]]] 
         lowest_highest.sort()
         new_row = new_row + lowest_highest
         new_row.append(p_best_estimate)
         new_row.append(abs(p_best_estimate - correct_answer[1]))
         if ((correct_answer[1] - correct_answer[0]) != 0):
            new_row.append((abs(p_best_estimate - correct_answer[1])) / (abs(correct_answer[1] - correct_answer[0])))
         else:
            new_row.append(abs(p_best_estimate - correct_answer[1]))


         new_row.append(abs(row[CC_col[(cohort, qnum)][1]] - row[CC_col[(cohort, qnum)][0]])) #width
         new_row.append((correct_answer[1] >= lowest_highest[0]) and (correct_answer[1] <= lowest_highest[1])) #is between lowest and highest?

         if correct_answer[0] < p_best_estimate:
            new_row = new_row + [True, False, False]
            if correct_answer[2] == 'Increased':
               score_directionality += 1
         elif correct_answer[0] > p_best_estimate:
            new_row = new_row + [False, True, False]
            if correct_answer[2] == 'Decreased':
               score_directionality += 1
         else:
            new_row = new_row + [False, False, True]
            if correct_answer[2] == 'Remained the Same':
               score_directionality += 1

         if correct_answer[3] == 'N/A': # cond 2
            new_row.append('N/A')
         else:
           new_row.append(p_best_estimate == (row[CC_col[(cohort, qnum - 1)][2]]))

         if correct_answer[4] == 'N/A':
            new_row.append('N/A')
         else:
           previous_initValue = CC[(cohort, qnum - 1)][0] 
           previous_bestEst = row[CC_col[(cohort, qnum - 1)][2]]
           new_row.append((abs(correct_answer[0] - p_best_estimate) / correct_answer[0]) > (abs(previous_initValue - previous_bestEst) / previous_initValue))

         new_row.append(correct_answer[5])
         new_row.append(abs(correct_answer[0] - p_best_estimate) / correct_answer[0] * 100)

         if correct_answer[6] == 'N/A' : #read off CPT
            new_row.append('N/A')
         else :
            new_row.append(p_best_estimate == correct_answer[1])
            if (p_best_estimate == correct_answer[1]):
               score_readCPT += 1
         
      for qnum in range(1, 8): #adding common effect

         correct_answer = CE[(cohort, qnum)]
         new_row = new_row + [correct_answer[0], correct_answer[1], correct_answer[2]]
         p_best_estimate = row[CE_col[(cohort, qnum)][2]]

         lowest_highest = [row[CE_col[(cohort, qnum)][1]], row[CE_col[(cohort, qnum)][0]]] 
         lowest_highest.sort()
         new_row = new_row + lowest_highest
         new_row.append(p_best_estimate)
         new_row.append(abs(p_best_estimate - correct_answer[1]))
         if ((correct_answer[1] - correct_answer[0]) != 0):
            new_row.append((abs(p_best_estimate - correct_answer[1])) / (abs(correct_answer[1] - correct_answer[0])))
         else:
            new_row.append(abs(p_best_estimate - correct_answer[1]))


         new_row.append(abs(row[CE_col[(cohort, qnum)][1]] - row[CE_col[(cohort, qnum)][0]])) #width
         new_row.append((correct_answer[1] >= lowest_highest[0]) and (correct_answer[1] <= lowest_highest[1])) #is between lowest and highest?

         if correct_answer[0] < p_best_estimate:
            new_row = new_row + [True, False, False]
            if correct_answer[2] == 'Increased':
               score_directionality += 1
         elif correct_answer[0] > p_best_estimate:
            new_row = new_row + [False, True, False]
            if correct_answer[2] == 'Decreased':
               score_directionality += 1
         else:
            new_row = new_row + [False, False, True]
            if correct_answer[2] == 'Remained the Same':
               score_directionality += 1

         new_row.append('N/A')

         if correct_answer[4] == 'N/A':
            new_row.append('N/A')
         elif qnum == 3:
           previous_initValue = CE[(cohort, qnum - 1)][0] 
           previous_bestEst = row[CE_col[(cohort, qnum - 1)][2]]
           new_row.append((abs(correct_answer[0] - p_best_estimate) / correct_answer[0]) < (abs(previous_initValue - previous_bestEst) / previous_initValue))
         else: #Q4
            previous_initValue = CE[(cohort, qnum - 1)][0]
            previous_bestEst = row[CE_col[(cohort, qnum - 1)][2]]
            previous_previous_initValue = CE[(cohort, qnum - 2)][0]
            previous_previous_bestEst = row[CE_col[(cohort, qnum - 2)][2]]
            new_row.append(((abs(correct_answer[0] - p_best_estimate) / correct_answer[0]) > (abs(previous_previous_initValue - previous_previous_bestEst) / previous_previous_initValue)) & ((abs(correct_answer[0] - p_best_estimate) / correct_answer[0]) > (abs(previous_initValue - previous_bestEst) / previous_initValue)))

         new_row.append(correct_answer[5])
         new_row.append(abs(correct_answer[0] - p_best_estimate) / correct_answer[0] * 100)

         if correct_answer[6] == 'N/A' : #read off CPT
            new_row.append('N/A')
         else :
            new_row.append(p_best_estimate == correct_answer[1])
            if (p_best_estimate == correct_answer[1]):
               score_readCPT += 1

      new_row.append(score_readCPT)
      new_row.append(score_directionality)

      avg_row = [row['task_load_1_1'], row['task_load_2_1'], row['task_load_3_1'], row['task_load_4_1'], row['task_load_5_1']]
      new_row.append(sum(avg_row) / len(avg_row))

      rows.append(new_row)

   table = pd.DataFrame(rows)
   table.to_csv("processedData_0202_fixed.csv", header = False, index = False)

arrange_csv(data)
#correct_rate(data)
