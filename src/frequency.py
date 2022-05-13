import numpy as np 
import scipy 

from scipy import signal 
from scipy.fft import fft, fftfreq 
from scipy.interpolate import interp1d



class fft():
	"""
	Computes the 1-dimensional DFT

	:params type: 	string - It must be amplitude for amplitude/ phase for / both  
	"""
	def __init__(self, type):
		self.type = type 
			
	def fft(X, fs, axis = 0):
		"""
		:param  x:  	array-like - Input Array 
		:param fs: 		int - Input Sampling Frequency 
		:param axis: 	int, optional - Axis over which to compute FFT, default = 0.  
		:return: 		Dictionary - frequency and fft amplitude values.  
		"""

		N = np.array(x).shape[0] 
		T = 1.0 / fs 

		# Compute fft 
		yf = (fft(input_array, axis = axis)) 
		# Skip C0 coefficient
		yf_trim = 2.0/N * np.abs(yf[1:(N//2), :])
		# Compute the Phase
		yf_phase = np.angle(yf[0:N//2, :])
		# Frequency range 
		xf = fftfreq(N, T)[1:N//2] 
		if self.type == 'phase':
			fft_ = {
				'frequency': xf, 
				'fft_phase': yf_phase
			}
			return fft_
		elif self.type == 'amplitude':
			fft_ = { 
			'frequency': xf, 
			'fft_amplitude': yf_trim
					}
			return fft_

		else: 
			fft_ = { 
				'frequency': xf, 
				'fft_amplitude': yf_trim, 
				'fft_phase': 	 yf_phase
						}

		return fft_



def modal_frequency(freq, x_amp, axis): 
	"""
	
	Computes the frequency and amplitude of the first natural frequency of 1D/2D input array

	:param freq: 	array-like - Frequency bandwidth 
	:param x_amp: 	array-like - Frequency amplitudes 
	:param axis:	int, optional - Axis over which to compute peak frequency.
	:return:		Dictionary - Peak Frequency , Peak Amplitude. 	  
	"""

	peak_amp = np.amax(x_amp, axis = axis)
	peak_freq = freq[np.argmax(x_amp, axis = axis)]

	return {"peak_freq": peak_freq, "peak_amp":peak_amp}

def peak_width(freq, x):
	"""
	Computes the width of the first natural frequency of 1D input array utilizing power method. 
	Delta_f = | f[p/sqrt(2)] - f[p/sqrt(2)] |
	width =  Delta_f/ (2*f[p])	
	
	:param freq: array-like - Frequency bandwidth
	:param x:	 array-like - Input Array. It must be in frequency domain. 
	:return: 	 Dictionary - Delta Frequency @p/sqrt(2), width
	"""  
    rms_val = np.max(x)/np.sqrt(2)
    peak_idx = np.argmax(x) 
    
    x_l = x[peak_idx-6 : peak_idx+1]
    y_l = freq[peak_idx-6 : peak_idx+1]
    
    x_r = x[peak_idx : peak_idx+6]
    y_r = freq[peak_idx : peak_idx+6]
    
    
    func_l = interp1d(x_l, y_l, kind = 'linear')
    func_r = interp1d(x_r, y_r, kind = 'linear')
    
    f_l = func_l(rms_val)
    f_r = func_r(rms_val)
    
    
    delta = f_r - f_l 
    
    width = (f_r - f_l) / (2 * freq[peak_idx])
    return {'delta': delta, 'width':width} 

def psd_welch(time_ds, fs):
    
    f, Pxx_den = signal.welch(time_ds, fs = fs, axis = 0, nperseg = fs*2)
    if log == True:
        return {'freq': f, 'psd': np.log(Pxx_den)}
    return {'freq': f, 'psd': Pxx_den}

