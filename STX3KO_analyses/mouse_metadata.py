import pandas as pd
import datetime







ctrl_sessions = {
    '4467331.1': {
        'alias': 'Ctrl_1',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,8,1),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-hSyn-mCherry',
        'sessions': (
            {'date_str': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
            {'date_str': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 1},
            {'date_str': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 2},
            {'date_str': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 3},
            {'date_str': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': -1, 'ravel_ind': 4},
            ),
        },
    '4467331.2': {
        'alias': 'Ctrl_2',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,8,1),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-hSyn-mCherry',
        'sessions': (
            {'date_str': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 0},
            {'date_str': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 1},
            {'date_str': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
            ({'date_str': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': 1, 'ravel_ind': 3}, 
            {'date_str': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 12, 'novel_arm': 1, 'ravel_ind': 4},),
            {'date_str': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 5},
            {'date_str': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 6},
        ),
    },
    '4467332.1': {
        'alias': 'Ctrl_3',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,7,26),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-hSyn-mCherry',
        'sessions': (
            {'date_str': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': -1, 'ravel_ind': 0},
            {'date_str': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 1},
            {'date_str': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 2},
            {'date_str': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 3},
            {'date_str': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
            {'date_str': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
        ),
    },
    '4467332.2': {
        'alias': 'Ctrl_4',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,7,26),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-hSyn-mCherry',
        'sessions': (
            {'date_str': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 18, 'novel_arm': 1, 'ravel_ind': 1},
            ({'date_str': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
            {'date_str': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 9, 'novel_arm': 1,'ravel_ind': 3}), 
            {'date_str': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 4},
            {'date_str': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 5}, # indexing error
            {'date_str': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 6},
            {'date_str': '04_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 7},
        )
    },
    '4467333.1': {
        'alias': 'Ctrl_5',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,8,1),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-hSyn-mCherry',
        'sessions': (
            {'date_str': '29_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 0},
            {'date_str': '30_11_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 1},
            {'date_str': '01_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 2},
            {'date_str': '02_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 3},
            {'date_str': '03_12_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 4},
            {'date_str': '05_12_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
        )
    },
    'mCherry6': {
        'alias': 'Ctrl_6',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2021,6,13),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 920.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-hSyn-mCherry',
        'sessions': (
            {'date_str': '14_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 0},
            {'date_str': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
            {'date_str': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 2},
            {'date_str': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
            {'date_str': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': 1, 'ravel_ind': 4},
            {'date_str': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 5},
        )
    },
    'mCherry7': {
        'alias': 'Ctrl_7',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2021,6,13),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 920.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-hSyn-mCherry',
        'sessions': (
            {'date_str': '14_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 0},
            {'date_str': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 1},
            {'date_str': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 2},
            {'date_str': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 3},
            {'date_str': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 4},
            {'date_str': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
        )
    },
    'mCherry8': {
        'alias': 'Ctrl_8',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2021,6,17),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 920.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-hSyn-mCherry',
        'sessions': (
            {'date_str': '14_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 0},
            {'date_str': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
            {'date_str': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 2},
            {'date_str': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 3},
            {'date_str': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 4},
            {'date_str': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5},
        )
    },
    'mCherry9': {
        'alias': 'Ctrl_9',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2021,6,17),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 920.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-hSyn-mCherry',
        'sessions': (
            {'date_str': '15_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
            {'date_str': '16_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 1},
            {'date_str': '17_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 2},
            {'date_str': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 3},
            {'date_str': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
            {'date_str': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
        )
    }                    
}

cre_sessions = {
    '4467975.1': {
        'alias': 'Cre_1',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,3,12),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-CaMKII-mCherry-IRES-cre',
        'sessions': (
            {'date_str': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 6, 'scan': 14, 'novel_arm': -1, 'ravel_ind': 1}, #indexing error
            {'date_str': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 2},
            {'date_str': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 3},
            {'date_str': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 4},
            {'date_str': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 5},
            {'date_str': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 6},   
        )
    },
    '4467975.2': {
        'alias': 'Cre_2',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,3,12),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-CaMKII-mCherry-IRES-cre',
        'sessions': (
            {'date_str': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 0},
            {'date_str': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1}, # indexing err
            {'date_str': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2}, # indexing err
            {'date_str': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 3}, # indexing err
            {'date_str': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 9, 'novel_arm': 1, 'ravel_ind': 4}, # indexing err
            {'date_str': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 8, 'novel_arm': 1, 'ravel_ind': 5}, # indexing err
        )
    },
    '4467975.3': {
        'alias': 'Cre_3',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,3,12),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-CaMKII-mCherry-IRES-cre',
        'sessions': (
            {'date_str': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 0},
            {'date_str': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 1},
            {'date_str': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 2},
            {'date_str': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 3},
            {'date_str': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
            {'date_str': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': -1, 'ravel_ind': 5},
        )
    },
    '4467975.4': {
        'alias': 'Cre_4',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,3,12),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-CaMKII-mCherry-IRES-cre',
        'sessions': (
            {'date_str': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': 1, 'ravel_ind': 0},
            {'date_str': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 1},
            {'date_str': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': 1, 'ravel_ind': 2},
            {'date_str': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
            {'date_str': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 4},
            {'date_str': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 5},
        )
    },
    '4467975.5': {
        'alias': 'Cre_5',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2020,3,12),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-CaMKII-mCherry-IRES-cre',
        'sessions': (
            {'date_str': '28_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'ravel_ind': 0},
            {'date_str': '29_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 1},
            {'date_str': '30_09_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 2},
            {'date_str': '01_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': -1, 'ravel_ind': 3},
            {'date_str': '02_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
            {'date_str': '03_10_2020', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 5, 'novel_arm': -1, 'ravel_ind': 5},
        )
    },
    'Cre7': {
        'alias': 'Cre_6',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2021,6,13),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-CaMKII-mCherry-IRES-cre',
        'sessions': (
            {'date_str': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 0},
            {'date_str': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 1},
            {'date_str': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': -1, 'ravel_ind': 2},
            {'date_str': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 3},
            {'date_str': '22_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'ravel_ind': 4},
            {'date_str': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'ravel_ind': 5},
        )
    },
    'Cre9': {
        'alias': 'Cre_7',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2021,6,17),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': 980.,
        'functional_indicator': 'AAV1-hSyn-jGCaMP7f',
        'static_indicator': 'AAVDJ-CaMKII-mCherry-IRES-cre',
        'sessions': (
            {'date_str': '18_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 0},
            {'date_str': '19_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 1},
            {'date_str': '20_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': 1, 'ravel_ind': 2},
            {'date_str': '21_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 3},
            {'date_str': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 4},
            {'date_str': '23_10_2021', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 2, 'novel_arm': 1, 'ravel_ind': 5}, 
        )
    }
}


sparse_sessions = {
    'SparseKO_02': {
        'alias': 'SparseKO_1',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2024, 5,  11),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': (920., 1040.),
        'functional_indicator': ('AAV8-CreON/FlpOFF-2.0-GCaMP6f', 'AAV8-Ef1a-FlpON/CreOFF-sRGECO'),
        'recombinase_viruses': ('AAV8-Ef1a-Cre-WPRE', 'AAV8-Ef1a-FLPo-WPRE'),
        'notes': 'multiplexed imaging: 920nm laser power high on odd frames & low on even frames; 1040nm laser power low on odd frames & high on even frames',
        'sessions': (
            {'date_str': '16_11_2024', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 14, 'novel_arm': -1, 'exp_day': 1, 'ravel_ind': 0},
            {'date_str': '17_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 25, 'novel_arm': -1, 'exp_day': 2, 'ravel_ind': 1},
            {'date_str': '18_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 18, 'novel_arm': -1,  'exp_day': 3, 'ravel_ind': 2},
            {'date_str': '19_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'exp_day': 4, 'ravel_ind': 3},
            {'date_str': '20_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 12, 'novel_arm': -1, 'exp_day': 5, 'ravel_ind': 4},
            {'date_str': '21_11_2024', 'scene': 'YMaze_LNovel', 'session':2, 'scan': 8, 'novel_arm': -1, 'exp_day': 6, 'ravel_ind': 5},        
        )
    },
    'SparseKO_06': {
        'alias': 'SparseKO_2',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2024, 12, 28),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': (920., 1040.),
        'functional_indicator': ('AAV8-CreON/FlpOFF-2.0-GCaMP6f', 'AAV8-Ef1a-FlpON/CreOFF-sRGECO'),
        'recombinase_viruses': ('AAV8-Ef1a-Cre-WPRE', 'AAV8-Ef1a-FLPo-WPRE'),
        'notes': 'multiplexed imaging: 920nm laser power high on odd frames & low on even frames; 1040nm laser power low on odd frames & high on even frames',
        'sessions': (
            {'date_str': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 7, 'novel_arm': 1, 'exp_day': 1, 'ravel_ind': 0},
            {'date_str': '27_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 23, 'novel_arm': 1, 'exp_day': 2, 'ravel_ind': 1},
            {'date_str': '28_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 25, 'novel_arm': 1, 'exp_day': 3, 'ravel_ind': 2},
            {'date_str': '29_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 1, 'novel_arm': 1, 'exp_day': 4, 'ravel_ind': 3},
            {'date_str': '30_05_2025', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 29, 'novel_arm': 1, 'exp_day': 5, 'ravel_ind': 4},
            {'date_str': '31_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': 1, 'exp_day': 6, 'ravel_ind': 5},
        )
    },
    'SparseKO_08': {
        'alias': 'SparseKO_3',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2025, 1, 18),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': (920., 1040.),
        'functional_indicator': ('AAV8-CreON/FlpOFF-2.0-GCaMP6f', 'AAV8-Ef1a-FlpON/CreOFF-sRGECO'),
        'recombinase_viruses': ('AAV8-Ef1a-Cre-WPRE', 'AAV8-Ef1a-FLPo-WPRE'),
        'notes': 'multiplexed imaging: 920nm laser power high on odd frames & low on even frames; 1040nm laser power low on odd frames & high on even frames',
        'sessions': (
            {'date_str': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 11, 'novel_arm': -1, 'exp_day': 1, 'ravel_ind': 0},
            {'date_str': '27_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 26, 'novel_arm': -1, 'exp_day': 2, 'ravel_ind': 1},
            {'date_str': '28_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 3, 'novel_arm': -1, 'exp_day': 3, 'ravel_ind': 2},
            {'date_str': '29_05_2025', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 6, 'novel_arm': -1, 'exp_day': 4, 'ravel_ind': 3},
            {'date_str': '30_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 32, 'novel_arm': -1, 'exp_day': 5, 'ravel_ind': 4},
            {'date_str': '31_05_2025', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 6, 'novel_arm': -1, 'exp_day': 6, 'ravel_ind': 5},
        )
    },
    'SparseKO_09': {
        'alias': 'SparseKO_4',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2025,1,18),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': (920., 1040.),
        'functional_indicator': ('AAV8-CreON/FlpOFF-2.0-GCaMP6f', 'AAV8-Ef1a-FlpON/CreOFF-sRGECO'),
        'recombinase_viruses': ('AAV8-Ef1a-Cre-WPRE', 'AAV8-Ef1a-FLPo-WPRE'),
        'notes': 'multiplexed imaging: 920nm laser power high on odd frames & low on even frames; 1040nm laser power low on odd frames & high on even frames \n \
            Session 3 (27_05_2025) data lost due to disk space issue.',
        'sessions': (
            {'date_str': '25_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'exp_day': 1, 'ravel_ind': 0},
            {'date_str': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 14, 'novel_arm': -1, 'exp_day': 2, 'ravel_ind': 1},
            None,
            {'date_str': '28_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 4, 'novel_arm': -1, 'exp_day': 4, 'ravel_ind': 3},
            {'date_str': '29_05_2025', 'scene': 'YMaze_LNovel', 'session': 3, 'scan': 14, 'novel_arm': -1, 'exp_day': 5, 'ravel_ind': 4},
            {'date_str': '30_05_2025', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 37, 'novel_arm': -1, 'exp_day': 6, 'ravel_ind': 5},
        )
    },           
    'SparseKO_10': {
        'alias': 'SparseKO_5',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2025,1,18),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': (920., 1040.),
        'functional_indicator': ('AAV8-CreON/FlpOFF-2.0-GCaMP6f', 'AAV8-Ef1a-FlpON/CreOFF-sRGECO'),
        'recombinase_viruses': ('AAV8-Ef1a-Cre-WPRE', 'AAV8-Ef1a-FLPo-WPRE'),
        'notes': 'multiplexed imaging: 920nm laser power high on odd frames & low on even frames; 1040nm laser power low on odd frames & high on even frames',
        'sessions': (
            {'date_str': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 17, 'novel_arm': 1, 'exp_day': 1, 'ravel_ind': 0},
            {'date_str': '27_05_2025', 'scene': 'YMaze_LNovel', 'session': 7, 'scan': 19, 'novel_arm': 1, 'exp_day': 2, 'ravel_ind': 1},
            {'date_str': '28_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 7, 'novel_arm': 1, 'exp_day': 3, 'ravel_ind': 2},
            {'date_str': '29_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 17, 'novel_arm': 1, 'exp_day': 4, 'ravel_ind': 3},
            {'date_str': '30_05_2025', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 5, 'novel_arm': 1, 'exp_day': 5, 'ravel_ind': 4},
            {'date_str': '31_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 6, 'novel_arm': 1, 'exp_day': 6, 'ravel_ind': 5},        
        )
    },
    'SparseKO_11': {
        'alias': 'SparseKO_6',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2025, 1, 18),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': (920., 1040.),
        'functional_indicator': ('AAV8-CreON/FlpOFF-2.0-GCaMP6f', 'AAV8-Ef1a-FlpON/CreOFF-sRGECO'),
        'recombinase_viruses': ('AAV8-Ef1a-Cre-WPRE', 'AAV8-Ef1a-FLPo-WPRE'),
        'notes': 'multiplexed imaging: 920nm laser power high on odd frames & low on even frames; 1040nm laser power low on odd frames & high on even frames',
        'sessions': (
            {'date_str': '29_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 23, 'novel_arm': -1, 'exp_day': 1, 'ravel_ind': 0},
            {'date_str': '30_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 12, 'novel_arm': -1, 'exp_day': 2, 'ravel_ind': 1},
            {'date_str': '31_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 9, 'novel_arm': -1, 'exp_day': 3, 'ravel_ind': 2},
            {'date_str': '01_06_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 31, 'novel_arm': -1, 'exp_day': 4, 'ravel_ind': 3},
            {'date_str': '02_06_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 2, 'novel_arm': -1, 'exp_day': 5, 'ravel_ind': 4},
            {'date_str': '03_06_2025', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 10, 'novel_arm': -1, 'exp_day': 6, 'ravel_ind': 5},
        )
    },
    'SparseKO_13': {
        'alias': 'SparseKO_7',               
        'sex': 'M',
        'date_of_birth': datetime.datetime(2025,1,26),
        'genotype': 'Stx3 flox/flox',
        'imaging_lambda': (920., 1040.),
        'functional_indicator': ('AAV8-CreON/FlpOFF-2.0-GCaMP6f', 'AAV8-Ef1a-FlpON/CreOFF-sRGECO'),
        'recombinase_viruses': ('AAV8-Ef1a-Cre-WPRE', 'AAV8-Ef1a-FLPo-WPRE'),
        'notes': 'multiplexed imaging: 920nm laser power high on odd frames & low on even frames; 1040nm laser power low on odd frames & high on even frames',
        'sessions': (
            {'date_str': '24_05_2025', 'scene': 'YMaze_LNovel', 'session': 4, 'scan': 27, 'novel_arm': 1, 'exp_day': 1, 'ravel_ind': 0},
            {'date_str': '25_05_2025', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 8, 'novel_arm': 1, 'exp_day': 2, 'ravel_ind': 1},
            {'date_str': '26_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 20, 'novel_arm': 1, 'exp_day': 3, 'ravel_ind': 2},
            {'date_str': '27_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 22, 'novel_arm': 1, 'exp_day': 4, 'ravel_ind': 3},
            {'date_str': '28_05_2025', 'scene': 'YMaze_LNovel', 'session': 2, 'scan': 16, 'novel_arm': 1, 'exp_day': 5, 'ravel_ind': 4},
            {'date_str': '29_05_2025', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 20, 'novel_arm': 1, 'exp_day': 6, 'ravel_ind': 5},
        )
    }       
}


# add parsed datetime (12:00) to each session dict (uses 'date_str' or fallback 'date')
def _add_session_datetimes(container):
    def process_item(item):
        if item is None:
            return
        if isinstance(item, dict):
            date_str = item.get('date_str') or item.get('date')
            if date_str:
                try:
                    parsed = datetime.datetime.strptime(date_str, '%d_%m_%Y')
                    item['datetime'] = datetime.datetime(parsed.year, parsed.month, parsed.day)
                except Exception:
                    pass
        elif isinstance(item, (list, tuple)):
            for it in item:
                process_item(it)

    for mouse in container.values():
        sessions = mouse.get('sessions', ())
        for entry in sessions:
            process_item(entry)


_add_session_datetimes(ctrl_sessions)
_add_session_datetimes(cre_sessions)
_add_session_datetimes(sparse_sessions)


