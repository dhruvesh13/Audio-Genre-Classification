import sys
import os
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
# from matplotlib.pyplot import specgram 

os.chdir(sys.argv[1])
# Directory provided as a command line argument will be opened to visualize the files inside
wavfiles = []
for wavfile in os.listdir(sys.argv[1]):
    if wavfile.endswith("wav"):
        wavfiles.append(wavfile)

	

wavfiles.sort()

# Declare sampling rates and song arrays for each arg
sampling_rates = []
song_arrays = []

# Read wavfiles
for wavfile in wavfiles:
	sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
	sampling_rates.append(sampling_rate)
	song_arrays.append(song_array)


i = 1  # plot number
# Plot spectrogram for each wave_file
for song_id, song_array, sampling_rate in zip(wavfiles, song_arrays, sampling_rates):
    # Create subplots
    plt.subplot(10, 10, i)
    i += 1
    #plt.title(song_id)
    plt.specgram(song_array[:30000], Fs=sampling_rate)
    print("Plotting spectrogram of song_id: " + song_id)

plt.savefig('Spectrogram.png')
plt.show()