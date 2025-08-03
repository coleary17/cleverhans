"""
Psychoacoustic masking threshold generation for adversarial audio attacks.
Ported from Python 2 to Python 3 with modern library compatibility.
"""

import numpy as np
import librosa
from scipy.fftpack import fft, ifft
from scipy import signal
import torch


def compute_PSD_matrix(audio, window_size):
    """
    First, perform STFT.
    Then, compute the PSD.
    Last, normalize PSD.
    """
    win = np.sqrt(8.0/3.) * librosa.stft(audio, center=False)
    z = abs(win / window_size)
    psd_max = np.max(z*z)
    psd = 10 * np.log10(z * z + 1e-20)  # Avoid numerical issues
    PSD = 96 - np.max(psd) + psd
    return PSD, psd_max   


def Bark(f):
    """Returns the bark-scale value for input frequency f (in Hz)"""
    return 13*np.arctan(0.00076*f) + 3.5*np.arctan(pow(f/7500.0, 2))


def quiet(f):
    """Returns threshold in quiet measured in SPL at frequency f with an offset 12(in Hz)"""
    thresh = 3.64*pow(f*0.001,-0.8) - 6.5*np.exp(-0.6*pow(0.001*f-3.3,2)) + 0.001*pow(0.001*f,4) - 12
    return thresh


def two_slops(bark_psd, delta_TM, bark_maskee):
    """
    Returns the masking threshold for each masker using two slopes as the spread function 
    """
    Ts = []
    for tone_mask in range(bark_psd.shape[0]):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker
        zero_index = np.argmax(dz > 0)
        sf = np.zeros(len(dz))
        sf[:zero_index] = 27 * dz[:zero_index]
        sf[zero_index:] = (-27 + 0.37 * max(bark_psd[tone_mask, 1] - 40, 0)) * dz[zero_index:] 
        T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        Ts.append(T)
    return Ts
    

def compute_th(PSD, barks, ATH, freqs):
    """Returns the global masking threshold"""
    # Identification of tonal maskers
    # find the index of maskers that are the local maxima
    length = len(PSD)
    masker_index = signal.argrelextrema(PSD, np.greater)[0]
    
    # delete the boundary of maskers for smoothing
    if len(masker_index) > 0:
        if masker_index[0] == 0:
            masker_index = masker_index[1:]
        if len(masker_index) > 0 and masker_index[-1] == length - 1:
            masker_index = masker_index[:-1]
    
    num_local_max = len(masker_index)
    
    if num_local_max == 0:
        # No maskers found, return ATH
        return pow(10, ATH/10.)

    # treat all the maskers as tonal (conservative way)
    # smooth the PSD 
    p_k = pow(10, PSD[masker_index]/10.)    
    p_k_prev = pow(10, PSD[masker_index - 1]/10.)
    p_k_post = pow(10, PSD[masker_index + 1]/10.)
    P_TM = 10 * np.log10(p_k_prev + p_k + p_k_post)
    
    # bark_psd: the first column bark, the second column: P_TM, the third column: the index of points
    _BARK = 0
    _PSD = 1
    _INDEX = 2
    bark_psd = np.zeros([num_local_max, 3])
    bark_psd[:, _BARK] = barks[masker_index]
    bark_psd[:, _PSD] = P_TM
    bark_psd[:, _INDEX] = masker_index
    
    # delete the masker that doesn't have the highest PSD within 0.5 Bark around its frequency 
    i = 0
    while i < bark_psd.shape[0]:
        if i + 1 >= bark_psd.shape[0]:
            break
            
        next_idx = i + 1
        while (next_idx < bark_psd.shape[0] and 
               bark_psd[next_idx, _BARK] - bark_psd[i, _BARK] < 0.5):
            # masker must be higher than quiet threshold
            if quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                bark_psd = np.delete(bark_psd, i, axis=0)
                break
                
            if bark_psd[i, _PSD] < bark_psd[next_idx, _PSD]:
                bark_psd = np.delete(bark_psd, i, axis=0)
                break
            else:
                bark_psd = np.delete(bark_psd, next_idx, axis=0)
        else:
            i += 1
    
    if bark_psd.shape[0] == 0:
        # No valid maskers after filtering
        return pow(10, ATH/10.)
    
    # compute the individual masking threshold
    delta_TM = 1 * (-6.025 - 0.275 * bark_psd[:, 0])
    Ts = two_slops(bark_psd, delta_TM, barks) 
    Ts = np.array(Ts)
    
    # compute the global masking threshold
    theta_x = np.sum(pow(10, Ts/10.), axis=0) + pow(10, ATH/10.) 
 
    return theta_x


def generate_th(audio, fs, window_size=2048):
    """
    Returns the masking threshold theta_xs and the max psd of the audio
    """
    PSD, psd_max = compute_PSD_matrix(audio, window_size)  
    freqs = librosa.fft_frequencies(sr=fs, n_fft=window_size)
    barks = Bark(freqs)

    # compute the quiet threshold 
    ATH = np.zeros(len(barks)) - np.inf
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])

    # compute the global masking threshold theta_xs 
    theta_xs = []
    # compute the global masking threshold in each window
    for i in range(PSD.shape[1]):
        theta_xs.append(compute_th(PSD[:,i], barks, ATH, freqs))
    theta_xs = np.array(theta_xs)
    return theta_xs, psd_max


class Transform(object):
    """
    PyTorch-based transform for computing PSD.
    Converted from TensorFlow to PyTorch for modern compatibility.
    """    
    def __init__(self, window_size, device='cpu'):
        self.scale = 8. / 3.
        self.frame_length = int(window_size)
        self.frame_step = int(window_size // 4)
        self.window_size = window_size
        self.device = torch.device(device)
    
    def __call__(self, x, psd_max_ori):
        """
        Compute PSD using PyTorch STFT
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        if isinstance(psd_max_ori, np.ndarray):
            psd_max_ori = torch.from_numpy(psd_max_ori).float().to(self.device)
            
        # Compute STFT
        win = torch.stft(x, n_fft=self.frame_length, hop_length=self.frame_step, 
                        window=torch.hann_window(self.frame_length).to(self.device),
                        return_complex=True)
        
        z = self.scale * torch.abs(win / self.window_size)
        psd = torch.square(z)
        
        # Reshape psd_max_ori for broadcasting
        psd_max_ori = psd_max_ori.view(-1, 1, 1)
        PSD = torch.pow(torch.tensor(10., device=self.device), 9.6) / psd_max_ori * psd
        
        return PSD
