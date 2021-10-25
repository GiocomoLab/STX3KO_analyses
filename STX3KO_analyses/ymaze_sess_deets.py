import numpy as np


ko_mice = ('4467975.1', '4467975.2', '4467975.3', '4467975.4', '4467975.5',)
ctrl_mice = ('4467331.1', '4467331.2', '4467332.1', '4467332.2', '4467333.1',)

exclude_list = {
    '4467975.1': ({'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel': -1, 'ravel_ind': 0},
                  {'date': '04_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel': -1, 'ravel_ind': 7},
                  {'date': '04_10_2020', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 10, 'novel': -1,
                   'ravel_ind': 8},),
    '4467332.2': (
        {'date': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 14, 'novel': 1, 'ravel_ind': 0},)
}

KO_sessions = {
    '4467975.1':   ({'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 6, 'scan': 14, 'novel_arm': -1, 'ravel_ind': 1},
        #(({'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
        #  {'date': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 6, 'scan': 14, 'novel_arm': -1, 'ravel_ind': 1}),
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

    'Cre7': ({'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 0},
             {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 1},
             {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 2},
             {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
             {'date': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 4},
             {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
            ({'date': '24_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 6},
             {'date': '24_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 2, 'scan': 6, 'novel_arm': 1, 'ravel_ind': 7},),
             # {'date': '25_10_2021', 'scene': 'YMaze_RewardReversal', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 8},
            # ({'date': '26_10_2021', 'scene': 'YMaze_RewardReversal', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 9},
            #  {'date': '26_10_2021', 'scene': 'YMaze_LNovel', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 10},),
             ),

    'Cre8': ({'date': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': 1, 'ravel_ind': 0},
             {'date': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 1},
             {'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 2},
             {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 6, 'novel_arm': 1, 'ravel_ind': 3},
             {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 4},
             {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
            ({'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 6},
             {'date': '23_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 7},),
             {'date': '23_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 8},
            # ({'date': '24_10_2020', 'scene': 'YMaze_RewardReversal', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 9},
            #  {'date': '24_10_2020', 'scene': 'YMaze_LNovel', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 10},),
             ),

    'Cre9': ({'date': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 0},
             {'date': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 1},
             {'date': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 2},
             {'date': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 3},
             {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 4},
             {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
            ({'date': '24_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 6},
             {'date': '24_10_2021', 'scene': 'YMaze_RewardReversal', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 7},),
             # {'date': '25_10_2021', 'scene': 'YMaze_RewardReversal', 'session': np.nan, 'scan': np.nan, 'novel_arm': -1, 'ravel_ind': 8},
            # ({'date': '26_10_2021', 'scene': 'YMaze_RewardReversal', 'session': np.nan, 'scan': np.nan, 'novel_arm': -1, 'ravel_ind': 9},
            #  {'date': '26_10_2021', 'scene': 'YMaze_LNovel', 'session': np.nan, 'scan': np.nan, 'novel_arm': -1, 'ravel_ind': 10},),
             ),


}


CTRL_sessions = {
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
                ({'date': '23_10_2021', 'scene': 'YMaze_RewardReversal', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 9},
                 {'date': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': np.nan, 'scan': np.nan, 'novel_arm': 1, 'ravel_ind': 10}),
                ),




}

