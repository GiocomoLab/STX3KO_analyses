import STX3KO_analyses as stx


def run_single_sess(mouse, metadata, session, day, 
                    scan=0, sub_notes=''):
    
    print(mouse, day)
    nwb_maker = stx.nwb_conversion.SessNWBConverter_Sparse(
        mouse, 
        metadata, 
        session, 
        day,
        scan=scan,
        sub_notes=sub_notes, 
        )
    # if not nwb_maker.out_path.is_file():
    nwb_maker.build_file()
    nwb_maker.write_file()



def loop_sessions(mice_dict):
    for mouse, metadata in mice_dict.items():
        sess_tup = metadata.get('sessions')
        
        for day, session in enumerate(sess_tup):
            if session is None:
                continue
            
            
            run_single_sess(
                mouse,
                metadata,
                session,
                day,
            )

if __name__=="__main__":
    
    loop_sessions(stx.mouse_metadata.sparse_sessions)
    
    