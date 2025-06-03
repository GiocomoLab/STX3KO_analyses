import numpy as np


ko_mice = ('4467975.1', '4467975.2', '4467975.3', '4467975.4', '4467975.5',
           'Cre7',  'Cre9') #, 'CA3-1')
ko_vr_mice = ('4467975.1', '4467975.2', '4467975.3', '4467975.4', '4467975.5',
           'Cre7',  'Cre9', 'CA3-1')
ctrl_mice = ('4467331.1', '4467331.2', '4467332.1', '4467332.2', '4467333.1',
             'mCherry6', 'mCherry7', 'mCherry8', 'mCherry9')
ctrl_vr_mice = ('4467331.1', '4467331.2', '4467332.1', '4467332.2', '4467333.1',
             'mCherry6', 'mCherry7', 'mCherry8', 'mCherry9')
sparse_mice = ('SparseKO_05','SparseKO_02','SparseKO_03','SparseKO_13')

exclude_list = {
    '4467975.4':{'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 0},
    '4467975.1': ({'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel': -1, 'ravel_ind': 0},
                  {'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel': -1, 'ravel_ind': 7},
                  {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 10, 'novel': -1,
                   'ravel_ind': 8},),
    '4467332.2': (
        {'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 14, 'novel': 1, 'ravel_ind': 0},)
}

SparseKO_sessions = {
    'SparseKO_05': (
        #{'date': '30_10_2024', 'scene': 'RunningTraining_scan', 'session': 2, 'scan': 3, 'novel_arm': -1, 'exp_day': 0},
                    {'date': '16_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 22, 'novel_arm': -1, 'exp_day': 1},
                    {'date': '17_11_2024', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 13, 'novel_arm': -1, 'exp_day': 2},
                    {'date': '18_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 10, 'novel_arm': -1, 'exp_day': 3},
                    {'date': '19_11_2024', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 16, 'novel_arm': -1, 'exp_day': 4},
                    {'date': '20_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 20, 'novel_arm': -1, 'exp_day': 5},
                    {'date': '21_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 16, 'novel_arm': -1, 'exp_day': 6},
                    {'date': '22_11_2024', 'scene': 'YMaze_LNovel_LongTimeout', 'session': 1, 'scan': 23, 'novel_arm': -1, 'exp_day': 7},
                  ),
    
    'SparseKO_02': ({'date': '16_11_2024', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 14, 'novel_arm': -1, 'exp_day': 1},
                    {'date': '17_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 25, 'novel_arm': -1, 'exp_day': 2},
                    {'date': '18_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 18, 'novel_arm': -1,  'exp_day': 3},
                    {'date': '19_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'exp_day': 4},
                    {'date': '20_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 12, 'novel_arm': -1, 'exp_day': 5},
                    {'date': '21_11_2024', 'scene': 'YMaze_LNovel', 'session':2, 'scan': 8, 'novel_arm': -1, 'exp_day': 6},
                    {'date': '22_11_2024', 'scene': 'YMaze_LNovel_LongTimeout', 'session': 2, 'scan': 15, 'novel_arm': -1, 'exp_day': 7}
                   
                  ),
    'SparseKO_03': ({'date': '16_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 17, 'novel_arm': 1, 'exp_day': 1},
                    {'date': '17_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'exp_day': 2},
                    {'date': '18_11_2024', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 6, 'novel_arm': 1, 'exp_day': 3},
                    {'date': '19_11_2024', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 7, 'novel_arm': 1, 'exp_day': 4},
                    {'date': '20_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 16, 'novel_arm': 1, 'exp_day': 5},
                    {'date': '21_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 12, 'novel_arm': 1, 'exp_day': 6},
                    {'date': '22_11_2024', 'scene': 'YMaze_LNovel_LongTimeout', 'session': 1, 'scan': 19, 'novel_arm': 1, 'exp_day': 7},
                   
                  ),
    
    'SparseKO_06': (
                    {'date': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 7, 'novel_arm': 1, 'exp_day': 1},
                    {'date': '27_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 23, 'novel_arm': 1, 'exp_day': 2},
        
    ),
    
    'SparseKO_08': (
                    {'date': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 11, 'novel_arm': 0, 'exp_day': 1},
                    {'date': '27_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 26, 'novel_arm': 0, 'exp_day': 2},
        
    ),
    
    'SparseKO_09': (
                    {'date': '24_05_2025', 'scene': 'TrainingYMaze', 'session': 2, 'scan': 20, 'exp_day': '0_mux'},
                    {'date': '25_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 0, 'exp_day': 1},
                    {'date': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 14, 'novel_arm': 0, 'exp_day': 2},
                    {'date': '27_05_2025', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 1, 'novel_arm': 0, 'exp_day': 3}, # ran out of disk space while scanning
        
    ),
    
    'SparseKO_10': (
                    {'date': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 17, 'novel_arm': 1, 'exp_day': 1},
                    {'date': '27_05_2025', 'scene': 'YMaze_LNovel', 'session': 7, 'scan': 19, 'novel_arm': 1, 'exp_day': 2},
        
    ),
        
    'SparseKO_13': (
                    {'date': '22_05_2025', 'scene': '980test', 'session': 0, 'scan': 2, 'exp_day': '-1_980'},
                    {'date': '23_05_2025', 'scene': 'TrainingYMaze_BlockScan', 'session': 2, 'scan': 10, 'exp_day': '0_block'},
                    {'date': '23_05_2025', 'scene': 'TrainingYMaze', 'session': 1, 'scan': 15, 'exp_day': '0_nomux'},
                    {'date': '23_05_2025', 'scene': 'TrainingYMaze', 'session': 2, 'scan': 17, 'exp_day': '0_mux'},
                    {'date': '24_05_2025', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 27, 'novel_arm': 1, 'exp_day': 1},
                    {'date': '25_05_2025', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 8, 'novel_arm': 1, 'exp_day': 2},
                    {'date': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 20, 'novel_arm': 1, 'exp_day': 3},
                    {'date': '27_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 22, 'novel_arm': 1, 'exp_day': 4},
        
        
        
                    
                  ),
}
KO_sessions = {
    '4467975.1':   (
        {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 6, 'scan': 14, 'novel_arm': -1, 'ravel_ind': 1}, #indexing error
        #(({'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
        #  {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 6, 'scan': 14, 'novel_arm': -1, 'ravel_ind': 1}),
         {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 2},
         {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 3},
         {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 4},
         {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 5},
         {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 6},
         #({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 7},
         # {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 10, 'novel_arm': -1,
         #'ravel_ind': 8}),

                    
                    '''
         {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 9},
         (
             {'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1,
              'ravel_ind': 10},
             {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 11}),
             '''
                    
         ),

    '4467975.2': (
        {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 0},
        {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1}, # indexing err
        {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2}, # indexing err
        {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 3}, # indexing err
        {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 9, 'novel_arm': 1, 'ravel_ind': 4}, # indexing err
        {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 5}, # indexing err

        ({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 6},
         {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': 1,
          'ravel_ind': 7}),
        {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': 1,
         'ravel_ind': 8},
        ({'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': 1,
          'ravel_ind': 9},
         {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 10}),

        
    ),

    '4467975.3': (
        {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 0},
        {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 1},
        {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 2},
        {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 3},
        {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
        {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 5},
        ({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 6},
         {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 6, 'novel_arm': -1,
          'ravel_ind': 7},),
        {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1,
         'ravel_ind': 8},
        ({'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': -1,
          'ravel_ind': 9},
         {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': -1,
          'ravel_ind': 10},),
         
        
    ),

    '4467975.4': (
        {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 0},
        {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
        {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 2},
        {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
        {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 4},
        {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 5},
        ({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 6},
         {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': 1,
          'ravel_ind': 7},),
        {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': 1,
         'ravel_ind': 8},
        ({'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1,
          'ravel_ind': 9},
         {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 10},),
    ),

    '4467975.5': (
        {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 0},
        {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 1},
        {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 2},
        {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 3},
        {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
        {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 5},
        ({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 6},
         {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 8, 'novel_arm': -1,
          'ravel_ind': 7},),
        {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1,
         'ravel_ind': 8},
        ({'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': -1,
          'ravel_ind': 9},
         {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 9, 'novel_arm': -1,
          'ravel_ind': 10},),
    ),

# RIP Cre6 :'(

    'Cre7': ({'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
             {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 1},
             {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': -1, 'ravel_ind': 2},
             {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 3},
             {'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
             {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
            #({'date': '24_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 6},
             #{'date': '24_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 7},),
             #{'date': '25_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 8},
            #({'date': '26_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 9},
             #{'date': '26_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 10},),
             ),

    # 'Cre8': ({'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': 1, 'ravel_ind': 0},
    #          {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 1},
    #          {'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 2},
    #          {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 6, 'novel_arm': 1, 'ravel_ind': 3},
    #          {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 4},
    #          {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
    #         ({'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 6},
    #          {'date': '23_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 7},),
    #          {'date': '23_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 8},
    #         # ({'date': '24_10_2020', 'scene': 'YMaze_RewardReversal', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 9},
    #         #  {'date': '24_10_2020', 'scene': 'YMaze_LNovel', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 10},),
    #          ),

    'Cre9': ({'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 0},
             {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 1},
             {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
             {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
             {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 4},
             {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
           # ({'date': '24_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 6},
            # {'date': '24_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 7},),
            # {'date': '25_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 8},
            #({'date': '26_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 9},
            # {'date': '26_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 10},),
             ),

    # 'CA3-1' : ({'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': -1},
    #            {'date': '24_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': -1},
    #            {'date': '25_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 0},
    #            {'date': '26_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 1},
    #            {'date': '27_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 2},
    #            {'date': '28_10_2021', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 3},
    #            ),


}


CTRL_sessions = {
    '4467331.1': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
                  {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 1},
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 2},
                  {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 3},
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': -1, 'ravel_ind': 4},
                  {'date': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
                  # (
                  #     {'date': '05_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1,
                  #      'ravel_ind': 6},
                  #     {'date': '05_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': -1,
                  #      'ravel_ind': 7},),
                  {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 1, 'novel_arm': -1,
                   'ravel_ind': 8},
                  ({'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1,
                    'ravel_ind': 9},
                   {'date': '07_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1,
                    'ravel_ind': 10}),
                  ),

    '4467331.2': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 0},
                  {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 1},
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
                  ({'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': 1, 'ravel_ind': 3},
                   
                   {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 12, 'novel_arm': 1,
                    'ravel_ind': 4},),## ELLA COMMENTED OUT FOR DOWNSAMPLE ANALYSIS!!!! 
                  
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 5},
                  {'date': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 6},
                  ({'date': '05_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 7},
                   {'date': '05_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1,
                    'ravel_ind': 8},),
                  {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1,
                   'ravel_ind': 9},
                  ({'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1,
                    'ravel_ind': 10},
                   {'date': '07_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1,
                    'ravel_ind': 11}),
                  ),

    '4467332.1': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': -1, 'ravel_ind': 0},
                  {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 1},
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 2},
                  {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 3},
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
                  {'date': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
                  (
                      {'date': '05_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1,
                       'ravel_ind': 6},
                      {'date': '05_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': -1,
                       'ravel_ind': 7},),
                  {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 1, 'novel_arm': -1,
                   'ravel_ind': 8},
                  ({'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1,
                    'ravel_ind': 9},
                   {'date': '07_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1,
                    'ravel_ind': 10}),
                  ),

    '4467332.2': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 18, 'novel_arm': 1, 'ravel_ind': 1},
                  # ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 14, 'novel_arm': 1, 'ravel_ind': 0},
                  #  {'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 18, 'novel_arm': 1, 'ravel_ind': 1}),
                  
                  ({'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
                   {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 9, 'novel_arm': 1,
                    'ravel_ind': 3}), 
                  
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 4},
                  {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 5}, # indexing error
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 6},
                  {'date': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 7},
                  ({'date': '06_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 8},
                   {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 6, 'novel_arm': 1,
                    'ravel_ind': 9},),
                  {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 2, 'novel_arm': 1,
                   'ravel_ind': 10},
                  ({'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1,
                    'ravel_ind': 11},
                   {'date': '07_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': 1,
                    'ravel_ind': 12}),
                  ),

    '4467333.1': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 0},
                  {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 1},
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 2},
                  {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 3},
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 4},
                  {'date': '05_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
                  (
                      {'date': '06_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1,
                       'ravel_ind': 6},
                      {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': -1,
                       'ravel_ind': 7},),
                  {'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1,
                   'ravel_ind': 8},
                  ({'date': '08_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1,
                    'ravel_ind': 9},
                   {'date': '08_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1,
                    'ravel_ind': 10}),
                  ),


    'mCherry6': ({'date': '14_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 0},
                 {'date': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
                 {'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 2},
                 {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
                 {'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 4},
                 {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 5},
                ({'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 6},
                 {'date': '20_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 7}),
                 {'date': '21_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 8},
                ({'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 9},
                 {'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 10}),
                ),

    'mCherry7': ({'date': '14_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 0},
                 {'date': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 1},
                 {'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 2},
                 {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 3},
                 {'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 4},
                 {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
                ({'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 6},
                 {'date': '20_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 7},),
                 {'date': '21_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 8},
                ({'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 9},
                 {'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 10},),
                ),

    'mCherry8': ({'date': '14_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 0},
                 {'date': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
                 {'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 2},
                 {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 3},
                 {'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 4},
                 {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
                ({'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 6},
                 {'date': '21_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 7}),
                 {'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 8},
                ({'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 9},
                 {'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 10}),
                ),

    'mCherry9': ({'date': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
                 {'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 1},
                 {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 2},
                 {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 3},
                 {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
                 {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
                ({'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 6},
                 {'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 7}),
                 {'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 8},
                ({'date': '23_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 9},
                 {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 10}),
                ),




}





KO_behavior_sessions = {
    '4467975.1':   #({'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 6, 'scan': 14, 'novel_arm': -1, 'ravel_ind': 1},
         (({'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
          {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 6, 'scan': 14, 'novel_arm': -1, 'ravel_ind': 1}),
         {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 2},
         {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 3},
         {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 4},
         {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 5},
         {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 6},
         ({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 7},
          {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 10, 'novel_arm': -1,
         'ravel_ind': 8}),
         {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 9},
         (
             {'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1,
              'ravel_ind': 10},
             {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 11}),
         ),

    '4467975.2': (
        {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 0},
        {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
        {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
        {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 3},
        {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 9, 'novel_arm': 1, 'ravel_ind': 4},
        {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 5},
        ({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 6},
         {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': 1,
          'ravel_ind': 7}),
        {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': 1,
         'ravel_ind': 8},
        ({'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': 1,
          'ravel_ind': 9},
         {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 10}),
    ),

    '4467975.3': (
        {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 0},
        {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 1},
        {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 2},
        {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 3},
        {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
        {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 5},
        ({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 6},
         {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 6, 'novel_arm': -1,
          'ravel_ind': 7},),
        {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1,
         'ravel_ind': 8},
        ({'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': -1,
          'ravel_ind': 9},
         {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': -1,
          'ravel_ind': 10},),
    ),

    '4467975.4': (
        {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 0},
        {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
        {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 2},
        {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
        {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 4},
        {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 5},
        ({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 6},
         {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': 1,
          'ravel_ind': 7},),
        {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': 1,
         'ravel_ind': 8},
        ({'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1,
          'ravel_ind': 9},
         {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 10},),
    ),

    '4467975.5': (
        {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 0},
        {'date': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 1},
        {'date': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 2},
        {'date': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 3},
        {'date': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
        {'date': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 5},
        ({'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 6},
         {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 8, 'novel_arm': -1,
          'ravel_ind': 7},),
        {'date': '05_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1,
         'ravel_ind': 8},
        ({'date': '06_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': -1,
          'ravel_ind': 9},
         {'date': '06_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 9, 'novel_arm': -1,
          'ravel_ind': 10},),
    ),

# RIP Cre6 :'(

    'Cre7': ({'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
             {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 1},
             {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': -1, 'ravel_ind': 2},
             {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 3},
             {'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
             {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
            ({'date': '24_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 6},
             {'date': '24_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 7},),
             {'date': '25_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 8},
            ({'date': '26_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 9},
             {'date': '26_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 10},),
             ),

    # 'Cre8': ({'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': 1, 'ravel_ind': 0},
    #          {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 1},
    #          {'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 2},
    #          {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 6, 'novel_arm': 1, 'ravel_ind': 3},
    #          {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 4},
    #          {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
    #         ({'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 6},
    #          {'date': '23_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 7},),
    #          {'date': '23_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 8},
    #         # ({'date': '24_10_2020', 'scene': 'YMaze_RewardReversal', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 9},
    #         #  {'date': '24_10_2020', 'scene': 'YMaze_LNovel', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 10},),
    #          ),

    'Cre9': ({'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 0},
             {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 1},
             {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
             {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
             {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 4},
             {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
            ({'date': '24_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 6},
             {'date': '24_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 7},),
             {'date': '25_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': -1, 'novel_arm': 1, 'ravel_ind': 8},
            ({'date': '26_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': -1, 'novel_arm': 1, 'ravel_ind': 9},
             {'date': '26_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': 1, 'ravel_ind': 10},),
             ),

    'CA3-1': ({'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 0},
              {'date': '24_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 1},
              {'date': '25_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 2},
              {'date': '26_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 3},
              {'date': '27_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 4},
              {'date': '28_10_2021', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': -1, 'novel_arm': -1, 'ravel_ind': 5},
              ),


    #'CA3-2': (),


}


CTRL_behavior_sessions = {
    '4467331.1': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
                  {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 1},
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 2},
                  {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 3},
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': -1, 'ravel_ind': 4},
                  {'date': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
                  (
                      {'date': '05_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1,
                       'ravel_ind': 6},
                      {'date': '05_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': -1,
                       'ravel_ind': 7},),
                  {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 1, 'novel_arm': -1,
                   'ravel_ind': 8},
                  ({'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1,
                    'ravel_ind': 9},
                   {'date': '07_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1,
                    'ravel_ind': 10}),
                  ),

    '4467331.2': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 0},
                  {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 1},
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
                  ({'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': 1, 'ravel_ind': 3},
                   {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 12, 'novel_arm': 1,
                    'ravel_ind': 4},),
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 5},
                  {'date': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 6},
                  ({'date': '05_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 7},
                   {'date': '05_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1,
                    'ravel_ind': 8},),
                  {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1,
                   'ravel_ind': 9},
                  ({'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1,
                    'ravel_ind': 10},
                   {'date': '07_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1,
                    'ravel_ind': 11}),
                  ),

    '4467332.1': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': -1, 'ravel_ind': 0},
                  {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 1},
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 2},
                  {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 3},
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
                  {'date': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
                  (
                      {'date': '05_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1,
                       'ravel_ind': 6},
                      {'date': '05_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': -1,
                       'ravel_ind': 7},),
                  {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 1, 'novel_arm': -1,
                   'ravel_ind': 8},
                  ({'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1,
                    'ravel_ind': 9},
                   {'date': '07_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1,
                    'ravel_ind': 10}),
                  ),

    '4467332.2': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 18, 'novel_arm': 1, 'ravel_ind': 1},
                  # ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 14, 'novel_arm': 1, 'ravel_ind': 0},
                  #  {'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 18, 'novel_arm': 1, 'ravel_ind': 1}),
                  ({'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
                   {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 9, 'novel_arm': 1,
                    'ravel_ind': 3}),
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 4},
                  {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 5},
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 6},
                  {'date': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 7},
                  ({'date': '06_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 8},
                   {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 6, 'novel_arm': 1,
                    'ravel_ind': 9},),
                  {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 2, 'novel_arm': 1,
                   'ravel_ind': 10},
                  ({'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1,
                    'ravel_ind': 11},
                   {'date': '07_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': 1,
                    'ravel_ind': 12}),
                  ),

    '4467333.1': ({'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 0},
                  {'date': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 1},
                  {'date': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 2},
                  {'date': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 3},
                  {'date': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 4},
                  {'date': '05_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
                  (
                      {'date': '06_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1,
                       'ravel_ind': 6},
                      {'date': '06_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': -1,
                       'ravel_ind': 7},),
                  {'date': '07_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1,
                   'ravel_ind': 8},
                  ({'date': '08_12_2020', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': -1,
                    'ravel_ind': 9},
                   {'date': '08_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1,
                    'ravel_ind': 10}),
                  ),


    'mCherry6': ({'date': '14_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 0},
                 {'date': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
                 {'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 2},
                 {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
                 {'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 4},
                 {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 5},
                ({'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 6},
                 {'date': '20_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 7}),
                 {'date': '21_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 8},
                ({'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 9},
                 {'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 10}),
                ),

    'mCherry7': ({'date': '14_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 0},
                 {'date': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 1},
                 {'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 2},
                 {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 3},
                 {'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 4},
                 {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
                ({'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 6},
                 {'date': '20_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 7},),
                 {'date': '21_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 8},
                ({'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 9},
                 {'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 10},),
                ),

    'mCherry8': ({'date': '14_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 0},
                 {'date': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
                 {'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 2},
                 {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 3},
                 {'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 4},
                 {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
                ({'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 6},
                 {'date': '21_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 7}),
                 {'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 8},
                ({'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 9},
                 {'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 10}),
                ),

    'mCherry9': ({'date': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 0},
                 {'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
                 {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 2},
                 {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 3},
                 {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 4},
                 {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
                ({'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 6},
                 {'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 7}),
                 {'date': '22_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 8},
                ({'date': '23_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 9},
                 {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 10}),
                ),




}

SparseKO_behavior_sessions = {
    'SparseKO_05': (
        # {'date': '30_10_2024', 'scene': 'RunningTraining_scan', 'session': 2, 'scan': 3, 'novel_arm': -1, 'exp_day': 0},
                    {'date': '16_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 22, 'novel_arm': -1, 'exp_day': 1},
                    {'date': '17_11_2024', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 13, 'novel_arm': -1, 'exp_day': 2},
                    {'date': '18_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 10, 'novel_arm': -1, 'exp_day': 3},
                    {'date': '19_11_2024', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 16, 'novel_arm': -1, 'exp_day': 4},
                    {'date': '20_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 20, 'novel_arm': -1, 'exp_day': 5},
                    {'date': '21_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 16, 'novel_arm': -1, 'exp_day': 6},
                    {'date': '22_11_2024', 'scene': 'YMaze_LNovel_LongTimeout', 'session': 1, 'scan': 23, 'novel_arm': -1, 'exp_day': 7},
                  ),
    
    'SparseKO_02': ({'date': '16_11_2024', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 14, 'novel_arm': -1, 'exp_day': 1},
                    {'date': '17_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 25, 'novel_arm': -1, 'exp_day': 2},
                    {'date': '18_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 18, 'novel_arm': -1,  'exp_day': 3},
                    {'date': '19_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'exp_day': 4},
                    {'date': '20_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 12, 'novel_arm': -1, 'exp_day': 5},
                    {'date': '21_11_2024', 'scene': 'YMaze_LNovel', 'session':2, 'scan': 8, 'novel_arm': -1, 'exp_day': 6},
                    {'date': '22_11_2024', 'scene': 'YMaze_LNovel_LongTimeout', 'session': 2, 'scan': 15, 'novel_arm': -1, 'exp_day': 7}
                   
                  ),
    'SparseKO_03': ({'date': '16_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 17, 'novel_arm': 1, 'exp_day': 1},
                    {'date': '17_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'exp_day': 2},
                    {'date': '18_11_2024', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 6, 'novel_arm': 1, 'exp_day': 3},
                    {'date': '19_11_2024', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 7, 'novel_arm': 1, 'exp_day': 4},
                    {'date': '20_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 16, 'novel_arm': 1, 'exp_day': 5},
                    {'date': '21_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 12, 'novel_arm': 1, 'exp_day': 6},
                    {'date': '22_11_2024', 'scene': 'YMaze_LNovel_LongTimeout', 'session': 1, 'scan': 19, 'novel_arm': 1, 'exp_day': 7},
                   
                  )
}

