########## ------------------------------- IMPORTS ------------------------ ##########
import os
import numpy as np
import adi
########## ---------------------------------------------------------------- ##########

class AdiGet:
    """
    Class to get data from labchart. Assumes one comment per block
     
    """

    
    def __init__(self, file_path, com_str):
        """
        
        Parameters
        ----------
        file_path : str,
        com_str : str, used to find block

        Returns
        -------
        None.

        """
        
        # get properties
        self.file_path = file_path
        self.com_str = com_str
        
        # get adi read object
        self.fread = adi.read_file(self.file_path)
        

    def get_data(self, data_ch, stim_ch, stop_com='stop'):
        """
        Get data from labchart channel object

        Parameters
        ----------
 
        data_ch : int, labchart channel number for Vm data
        stim_ch : int, labchart channel number for Current input
        stop_com, str, name of comment to be used for early block termination

        Returns
        data : array, voltage signal
        stim : array, input current
        fs : float, sampling rate in seconds
        """
  
        
        # find block
        ch_obj = self.fread.channels[data_ch]
        
        # get first comments and and comment times
        coms = []
        com_t = []
        for c in ch_obj.records:
            if not c.comments:
                coms.append('')
                com_t.append('')
            else:
                coms.append(c.comments[0].text.lower())
                com_t.append(c.comments[0].tick_position)

        # find block where search comment is present and get start time
        if self.com_str not in coms:
            print('Comment was not found in:', self.file_path)
            return None, None, None
        
        self.block = coms.index(self.com_str)
        start = com_t[self.block] + 1
        
        # check if stop comment exists and get stop time
        block_coms = [c.text.lower() for c in ch_obj.records[self.block].comments]
        stop = None
        if stop_com in block_coms:
            idx = block_coms.index(stop_com)
            stop = ch_obj.records[self.block].comments[idx].tick_position

        fs = int(ch_obj.fs[0])
        # get data
        if self.com_str == 'io':
            start -= int(fs/2)
        if start < 1:
            start = 1
        data = self.fread.channels[data_ch].get_data(self.block+1, start_sample=start, stop_sample=stop)
        
        # recreate IO stim from template
        if len(self.fread.channels) == 1:
            
            # load io template
            io_temp = np.loadtxt(os.path.join('stim','io_temp.csv'))
            
            # get comments
            com_list = [(com.text, com.tick_position) for com in self.fread.records[self.block].comments]
            com_txt, com_time = list(zip(*com_list))
            
            # recreate io baseline
            from scipy import stats
            stim = np.ones(data.shape[0])*stats.mode(io_temp, keepdims=False)[0]
            
            # find true signal end around comment
            from scipy.signal import find_peaks
            io_stop = com_time[com_txt.index('pulse_end')]-start
            k = int(fs*.01)
            io_end_sig = data[io_stop-k:io_stop+k]
            idx = find_peaks(-np.diff(io_end_sig), distance=int(k/2), prominence=1)[0][0]
            io_stop += idx-k
            stim[io_stop-len(io_temp):io_stop] = io_temp
        else:
            stim = self.fread.channels[stim_ch].get_data(self.block+1, start_sample=start, stop_sample=stop)
        
        return data, stim, fs
    
    
    
    
    
    
    
