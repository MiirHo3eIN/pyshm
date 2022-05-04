import numpy as np 
import scipy 

from scipy import signal 
from scipy.fft import fft, fftfreq 


def fft_amplitude(x, fs, axis):
	"""
	
	Compute the 1-dimensional DFT amplitude. 

	:param  x:  	array-like - Input Array 
	:param fs: 		int - Input Sampling Frequency 
	:param axis: 	int, optional - Axis over which to compute FFT.  
	:return: 		Dictionary - frequency and fft amplitude values.  
	"""

	N = np.array(x).shape[0] 
	T = 1.0 / fs 

	# Compute fft 
	yf = (fft(input_array, axis = axis)) 
	# Skip C0 coefficient
	yf_trim = 2.0/N * np.abs(yf[1:(N//2), :])	
	# Frequency range 
	xf = fftfreq(N, T)[1:N//2] 

	fft_ = { 
	'frequency': xf, 
	'fft_amplitude': yf_trim
			}


	return fft_

def fft_phase(x):
	"""
		TBD
	""" 
	return pass

def modal_frequency(freq, x_amp, axis): 
	"""
	
	Computes the frequency and amplitude of the first natural frequency of input array

	:param freq: 	array-like - Frequency range
	:param x_amp: 	array-like - Frequency amplitudes 
	:param axis:	int, optional - Axis over which to compute peak frequency.
	:return:		Dictionary - Peak Frequency , Peak Amplitude. 	  
	"""

	peak_amp = np.amax(x_amp, axis = axis)
	peak_freq = freq[np.argmax(x_amp, axis = axis)]

	return {"peak_freq": peak_freq, "peak_amp":peak_amp}



