def read_raw_sound_files(filename):
            
        import soundfile as sf
        #import sounddevice as sd
        #import scikits.audiolab
        
        
        if filename=='':
                filename='/am/roxy/home/hsadeghi/Downloads/an4/wav/an4_clstk/fash/cen1-fash-b.raw';
                    
        data, samplerate = sf.read(filename, channels=1, samplerate=16000, subtype='FLOAT',  endian='BIG')
        
        #scikits.audiolab.play(data, fs=16000)
        
        return data, samplerate


#===========================================  
def database_file_names(folder_name):
    
    import os
    
    if folder_name=='':
        folder_name='/home/hsadeghi/Downloads/an4/wav/an4_clstk';
    
    
    # getting subfolder names
    subfolder_names=list();
    for name in os.listdir(folder_name):
        if os.path.isdir(os.path.join(folder_name, name)):
            subfolder_names.append(name)
                   
    #Getting all file names
    file_names = list()
    for name1 in subfolder_names:
        for root, dirs, files in os.walk(os.path.join(folder_name,name1), topdown=False):
            for name2 in files:
                file_names.append(os.path.join(folder_name, os.path.join(name1, name2)))
    
    return file_names


    #========================================
#def apply_channel(X,H, group_size):
#    
#    
#            
#            
      
    


#========================================
def channel_generator(num_channels):
    
    import numpy as np
    import scipy.io
    
    mat = scipy.io.loadmat('/am/roxy/home/hsadeghi/Downloads/RIR-Generator-master/impulse_responses.mat')   ;
    fs=np.asscalar(mat['fs']);
    
    H_total=mat['H'];  # 1000 by 4096                          Each column is a channel IR vector
    
    H_total=np.array(H_total);
    
    num_impulse_responses=len(H_total[0]);
#        length_IR=len(H_total);
    
    random_ind=np.random.randint(0, high=num_impulse_responses,
                                    size=num_channels,
                                    dtype='l')
    
    H=H_total[:,random_ind];
  
    return H, fs



    
    
    
