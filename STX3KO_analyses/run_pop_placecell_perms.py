import os

import numpy as np
from matplotlib import pyplot as plt
import dill

import TwoPUtils as tpu
import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u


ko_mice = stx.ymaze_sess_deets.ko_mice
ctrl_mice = stx.ymaze_sess_deets.ctrl_mice


def load_single_sess(mouse, deets, pkl_basedir = '/home/mplitt/YMazeSessPkls'):
    pkldir = os.path.join(pkl_basedir, mouse)            
    
    sess = stx.session.YMazeSession.from_file(
            os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session'])),
            verbose=False, novel_arm=deets['novel_arm'])
    return sess

if __name__=="__main__":
    # for session_dict in (stx.ymaze_sess_deets.CTRL_sessions, stx.ymaze_sess_deets.KO_sessions):

    #     for mouse, sessions in session_dict.items():
    #         print(mouse)
    #         for deets in sessions:
    #             if isinstance(deets,tuple):
    #                 for _deets in deets:
    #                     print(_deets)
                        
    #                     sess.place_cells_calc(Fkey='F_dff', nperms=1000, use_tank_method=False)
    #                     sess._abc_impl = None
    #                     tpu.sess.save_session(sess,'/home/mplitt/YMazeSessPkls')
                        
    #             else:
    #                 print(deets)
    #                 sess = load_single_sess(mouse,deets)
                    
    #                 sess.place_cells_calc(Fkey='F_dff', nperms=1000, use_tank_method=False)
    #                 sess._abc_impl = None
    #                 tpu.sess.save_session(sess,'/home/mplitt/YMazeSessPkls')


    

    for m, (mouse, sessions) in enumerate(stx.ymaze_sess_deets.SparseKO_sessions.items()):
        print(mouse)
        
        for i, deets in enumerate(sessions):
            if mouse == 'SparseKO_09' and i==2:
                continue

            if deets is not None:
                sess = load_single_sess(mouse,deets)
                
                sess.place_cells_calc(Fkey='channel_0_F_dff', out_key='channel_0_F_dff', nperms=1000, mux=True, use_tank_method=False)
                sess.place_cells_calc(Fkey='channel_1_F_dff', out_key='channel_1_F_dff', nperms=1000, mux=True, use_tank_method=False)
            
                sess._abc_impl = None
                tpu.sess.save_session(sess,'/home/mplitt/YMazeSessPkls')