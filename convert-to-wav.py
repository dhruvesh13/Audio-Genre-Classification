import sys
import os
import sox

# Store the directory where all the audio files are saved
genre_dirs = ['/home/dhruvesh/Desktop/dsp-final/genres/blues','/home/dhruvesh/Desktop/dsp-final/genres/classical','/home/dhruvesh/Desktop/dsp-final/genres/country',
'/home/dhruvesh/Desktop/dsp-final/genres/disco','/home/dhruvesh/Desktop/dsp-final/genres/hiphop','/home/dhruvesh/Desktop/dsp-final/genres/jazz','/home/dhruvesh/Desktop/dsp-final/genres/metal',
'/home/dhruvesh/Desktop/dsp-final/genres/pop','/home/dhruvesh/Desktop/dsp-final/genres/reggae','/home/dhruvesh/Desktop/dsp-final/genres/rock'
]
for genre_dir in genre_dirs:
	# change directory to genre_dir
	os.chdir(genre_dir)

	# echo contents before altering
	print('Contents of ' + genre_dir + ' before conversion: ')
	os.system("ls")

	# loop through each file in current dir
	for file in os.listdir(genre_dir):
		# SOX
		os.system("sox " + str(file) + " " + str(file[:-3]) + ".wav")
	
	# delete .au from current dir
	os.system("rm *.au")
	# echo contents of current dir
	print('After conversion:')
	os.system("ls")
	print('\n')

print("Conversion complete. Check respective directories.")