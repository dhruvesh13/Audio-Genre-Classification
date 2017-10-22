#from scikits.talkbox.features import mfcc 
import scipy.io.wavfile
import numpy as np 
import sys
import os
import glob
from utils1 import GENRE_DIR, GENRE_LIST
from python_speech_features import mfcc
#from librosa.feature import mfcc

# Given a wavfile, computes mfcc and saves mfcc data
def create_ceps(wavfile):
	sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
	print(sampling_rate)
	"""Get MFCC
	ceps  : ndarray of MFCC
	mspec : ndarray of log-spectrum in the mel-domain
	spec  : spectrum magnitude
	"""
	ceps=mfcc(song_array)
	#ceps, mspec, spec= mfcc(song_array)
	print(ceps.shape)
	#this is done in order to replace NaN and infinite value in array
	bad_indices = np.where(np.isnan(ceps))
	b=np.where(np.isinf(ceps))
	ceps[bad_indices]=0
	ceps[b]=0
	write_ceps(ceps, wavfile)

# Saves mfcc data 
def write_ceps(ceps, wavfile):
	base_wav, ext = os.path.splitext(wavfile)
	data_wav = base_wav + ".ceps"
	np.save(data_wav, ceps)


def main():
	
	for label, genre in enumerate(GENRE_LIST):
		for fn in glob.glob(os.path.join(GENRE_DIR, genre)):
			for wavfile in os.listdir(fn):
				if wavfile.endswith("wav"):
					create_ceps(os.path.join(GENRE_DIR, genre,wavfile))

	

if __name__ == "__main__":
	main()