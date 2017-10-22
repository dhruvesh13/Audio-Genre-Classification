import scipy
import scipy.io.wavfile
import os
import sys
import glob
import numpy as np
from utils1 import GENRE_DIR, GENRE_LIST

# Extracts frequencies from a wavile and stores in a file
def create_fft(wavfile): 
	sample_rate, song_array = scipy.io.wavfile.read(wavfile)
	print(sample_rate)
	fft_features = abs(scipy.fft(song_array[:30000]))
	print(song_array)
	base_fn, ext = os.path.splitext(wavfile)
	data_fn = base_fn + ".fft"
	np.save(data_fn, fft_features)


def main():
	
	for label, genre in enumerate(GENRE_LIST):
		for fn in glob.glob(os.path.join(GENRE_DIR, genre)):
			for wavfile in os.listdir(fn):
					if wavfile.endswith("wav"):
						create_fft(os.path.join(GENRE_DIR, genre,wavfile))
				
				

	


if __name__ == "__main__":
	main()