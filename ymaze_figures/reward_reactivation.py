# 1) Find some examples of reactivation in both groups for novel and familiar

# 2) Show no difference in proportion of cells recruited to reactivation
#     a) in novel or familiar
#     b) in familiar when restricting to last block

# 3) Show that reactivated cells tend to have higher firing rates than non reactivated cells in novel environments
#     a) show that reactivated cells also distinguish between novel and familiar environments by firing rate
#     b) make sure to exclude post reward reactivation in calculation
    
# 4) Show higher trial x trial correlation in reactivated vs non-reactivated cells
#     a) only in novel and effect bigger in controls
#     b) find some examples 
#     c) exclude reward consumption times/cells who's peak is within the reward zone/spatial bins within the reward zone
    
# 5) Show higher across day correlation for cells that are reactivated vs non-reactivated
#     a) effect present in both familiar and novel
#     b) effect bigger in controls 
#     c) exclude reward consumption times/cells who's peak is within the reward zone/spatial bins within the reward zone
#     d) find examples 
    
# 6) Find all cells reactivated during reward consumption
#     a) plot their activity during a reactivation event
#     b) are they highly synchronous
    

class RewardReactivation:
    
    def __init__(self, sess):
        
        if isinstance(sess.scan_info,list):
            self.frame_rate = sess.scan_info[0]['frame_rate']
        else:
            self.frame_rate = sess.scan_info['frame_rate']
        
        
        
        
        self.spks = sess.timeseries['spks']
        self.trial_mat = sess.trial_matrices['spks_norm']
        self.speed = sess.timeseries['speed']
        self.lr = sess.timeseries['LR']
        self.t = sess.timeseries['t']
        self.reward = sess.timeseries['reward']
        
        self.novel_arm = sess.novel_arm
        self.trial_info = sess.trial_info
        self.trial_start_inds = sess.trial_start_inds
        self.teleport_inds = sess.teleport_inds
        
        
        # mask for post reward epochs
        self.post_reward_epochs = None
        self._post_reward_mask(sess)
        
        # threshold spks timeseries to get "significant" transients
        self.thresh_spks = .2*np.nanmax(sess.timeseries['spks'], axis=-1, keepdims=True)
        self.spks_th = 1.*np.copy(self.spks)
        self.spks_th[self.spks_th<self.thresh_spks]=0
        
        
        
        # all activations during stopping, including periods when not near reward zone
        self.activation_mask = None
        # activations during reward consumption
        self.post_reward_activation_mask = None
        # all activations during running
        self.trial_activation_mask = None
        self.get_activations()
        
        # active during trial and during reward consumption
        self.post_reward_reactivation_mask = None
        # active during trial but not during reward consumption
        self.non_reactivation_mask = None
        self.get_post_reward_reactivations()
        
        
    def _post_reward_mask(self, sess, max_post_reward_time = 5, max_post_reward_dist = 3):
        post_reward_bool = 0.*self.t.ravel()
        for start,stop in zip(self.trial_start_inds, self.teleport_inds):

            cum_reward = np.cumsum(self.reward.ravel()[start:stop])>0
            post_reward_inds = np.argwhere(cum_reward).ravel()
            if post_reward_inds.shape[0]>0:
                post_reward_bool[post_reward_inds[0]+start:start+np.minimum(post_reward_inds[0]+int(max_post_reward_time*self.frame_rate),stop-start)] = 1


        nov_post_rzone_mask = (sess.timeseries['LR']==sess.novel_arm)*(sess.timeseries['t']>sess.rzone_nov['tfront']+max_post_reward_dist)
        fam_post_rzone_mask = (sess.timeseries['LR']==-1*sess.novel_arm)*(sess.timeseries['t']>sess.rzone_fam['tfront']+max_post_reward_dist)
        post_reward_bool[nov_post_rzone_mask.ravel()]=0
        post_reward_bool[fam_post_rzone_mask.ravel()]=0
        self.post_reward_epochs = post_reward_bool
        
        
    def get_activations(self):
    
        spks_ledge = 1*(np.diff(1.*(self.spks_th>0),prepend=0,axis=-1)>0)

        self.activation_mask = 1*(spks_ledge>0)*(self.speed<2)
        self.post_reward_activation_mask = 1*(spks_ledge>0)*(self.speed<2)*(self.post_reward_epochs[np.newaxis,:])
        self.trial_activation_mask = 1*(spks_ledge>0)*(1-self.post_reward_epochs[np.newaxis,:])
    
    def get_post_reward_reactivations(self):
        post_reward_reactivation_mask = np.zeros((self.spks.shape[0],self.trial_start_inds.shape[0]))
        non_reactivation_mask = np.zeros((self.spks.shape[0],self.trial_start_inds.shape[0]))
        for trial, (start,stop) in enumerate(zip(self.trial_start_inds,self.teleport_inds)):
            post_reward_reactivation_mask[:,trial] = 1*(self.post_reward_activation_mask[:,start:stop].sum(axis=-1)>0)*(self.trial_activation_mask[:,start:stop].sum(axis=-1)>0)
            non_reactivation_mask[:,trial] = 1*(self.post_reward_activation_mask[:,start:stop].sum(axis=-1)==0)*(self.trial_activation_mask[:,start:stop].sum(axis=-1)>0)
        
        self.post_reward_reactivation_mask = post_reward_reactivation_mask
        self.non_reactivation_mask = non_reactivation_mask
        
    

        

    
def post_reward_mask(sess,fr):
    post_reward_bool = 0.*sess.timeseries['t'].ravel()


    for start,stop in zip(sess.trial_start_inds, sess.teleport_inds):
    
        cum_reward = np.cumsum(sess.timeseries['reward'].ravel()[start:stop])>0
        post_reward_inds = np.argwhere(cum_reward).ravel()
        if post_reward_inds.shape[0]>0:
            
            post_reward_bool[post_reward_inds[0]+start:start+np.minimum(post_reward_inds[0]+int(5*fr),stop-start)] = 1
    

    nov_post_rzone_mask = (sess.timeseries['LR']==sess.novel_arm)*(sess.timeseries['t']>sess.rzone_nov['tfront']+3)
    fam_post_rzone_mask = (sess.timeseries['LR']==-1*sess.novel_arm)*(sess.timeseries['t']>sess.rzone_fam['tfront']+3)
    post_reward_bool[nov_post_rzone_mask.ravel()]=0
    post_reward_bool[fam_post_rzone_mask.ravel()]=0
    return post_reward_bool


def get_activation_masks(sess):
    if isinstance(sess.scan_info,list):
        fr = sess.scan_info[0]['frame_rate']
    else:
        fr = sess.scan_info['frame_rate']
    post_reward_bool = post_reward_mask(sess,fr)
    spks = fr*np.copy(sess.timeseries['spks'])
    # spks[sess.timeseries['F_dff']<.1]=0
    spks[sess.timeseries['spks']<.2*np.nanmax(sess.timeseries['spks'],axis=-1,keepdims=True)]=0
    speed = sess.timeseries['speed']
    
    spks_ledge = 1*(np.diff(1.*(spks>0),prepend=0,axis=-1)>0)
    
    reactivation_mask = 1*(spks_ledge>0)*(speed<2)
    post_reward_activation_mask = 1*(spks_ledge>0)*(speed<2)*(post_reward_bool[np.newaxis,:])#*(sess.vr_data['LR'].to_numpy()==sess.novel_arm)
    trial_activation_mask = 1*(spks_ledge>0)*(1-post_reward_bool[np.newaxis,:])
    
    return spks, speed, reactivation_mask, post_reward_activation_mask, trial_activation_mask
    

def get_post_reward_reactivation_masks(spks, trial_starts, teleports, post_reward_activation_mask, trial_activation_mask):
    post_reward_reactivation_mask = np.zeros((spks.shape[0],trial_starts.shape[0]))
    non_reactivation_mask = np.zeros((spks.shape[0],trial_starts.shape[0]))
    for trial, (start,stop) in enumerate(zip(trial_starts,teleports)):
        post_reward_reactivation_mask[:,trial] = 1*(post_reward_activation_mask[:,start:stop].sum(axis=-1)>0)*(trial_activation_mask[:,start:stop].sum(axis=-1)>0)
        non_reactivation_mask[:,trial] = 1*(post_reward_activation_mask[:,start:stop].sum(axis=-1)==0)*(trial_activation_mask[:,start:stop].sum(axis=-1)>0)
        
    return post_reward_reactivation_mask, non_reactivation_mask
    
def fam_reactivation_masks(sess):
    
    spks, speed, reactivation_mask, post_reward_activation_mask, trial_activation_mask = get_activation_masks(sess)
    
    
    trial_starts = sess.trial_start_inds[sess.trial_info['LR']==-1*sess.novel_arm]
    teleports = sess.teleport_inds[sess.trial_info['LR']==-1*sess.novel_arm]
    
    return get_post_reward_reactivation_masks(spks, trial_starts, teleports, post_reward_activation_mask, trial_activation_mask)
    

def nov_reactivation_masks(sess):
    
    spks, speed, reactivation_mask, post_reward_activation_mask, trial_activation_mask = get_activation_masks(sess)
    
    
    trial_starts = sess.trial_start_inds[sess.trial_info['LR']==sess.novel_arm]
    teleports = sess.teleport_inds[sess.trial_info['LR']==sess.novel_arm]
    
    return get_post_reward_reactivation_masks(spks, trial_starts, teleports, post_reward_activation_mask, trial_activation_mask)