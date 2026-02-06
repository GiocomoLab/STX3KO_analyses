import STX3KO_analyses as stx


def run_single_sess(mouse, metadata, session, day, oak_pwd,
                    scan=1, sub_notes=''):
    
    nwb_maker = stx.nwb_conversion.SessNWBConverter_Dense(
        mouse, 
        metadata, 
        session, 
        day,
        oak_pwd,
        scan=scan,
        sub_notes=sub_notes, 
        )
    if not nwb_maker.out_path.is_file():
        nwb_maker.build_file()
        nwb_maker.write_file()
        nwb_maker.remove_sbx_data()


def loop_sessions(mice_dict):
    for mouse, metadata in mice_dict.items():
        sess_tup = metadata.get('sessions')
        
        for day, session in enumerate(sess_tup):
            
            if isinstance(session, tuple):
                for _scan, _session in enumerate(session):
                    run_single_sess(
                        mouse,
                        metadata,
                        _session, 
                        day, 
                        oak_pwd,
                        scan=_scan,
                        sub_notes=f"Data must be combined with other scan from day {day}",
                    )
            else:
                run_single_sess(
                    mouse,
                    metadata,
                    session,
                    day,
                    oak_pwd,
                )

if __name__=="__main__":
    with open('.oak_pwd', 'r') as f:
        oak_pwd = f.read().strip()
        
    loop_sessions(stx.mouse_metadata.ctrl_sessions)
    loop_sessions(stx.mouse_metadata.cre_sessions)
    