# Audio-Genre-Classification
Automatic music genre classification using Machine Learning algorithms like- Logistic Regression and K-Nearest Neighbours

**Language used :** Python 2.7

This repository consists of development code that classifies music according to the following genres: 

1. Blues

2. Classical (Western)

3. Country

4. Disco

5. Metal

6. Pop

### The Dataset

The dataset used for training the model is the GTZAN dataset. A brief of the data set: 

* This dataset was used for the well known paper in genre classification " Musical genre classification of audio signals " by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.
* The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.
* Official web-page: [marsyas.info](http://marsyas.info/download/data_sets)
* Download size: Approximately 1.2GB
* Download link: [Download the GTZAN genre collection](http://opihi.cs.uvic.ca/sound/genres.tar.gz)

## Feature of .wav files are generated using:
1. Fast fourier Transform (FFT) (Classification accuracy- **~70%**)
2. Mel Frequency Cepstral Coefficients (MFCC) (Classification accuracy- **~75%**)

## Algorithms used:
1. Logistic Regression
2. K-Nearest Neighbours

## How to use project for testing:

1. Download dataset from: http://opihi.cs.uvic.ca/sound/genres.tar.gz.

2. Extract into suitable directory: BASE_DIR

3. Run convert-to-wav.py on each subdir of BASE_DIR.

4. Run fft-features.py on each subdir of BASE_DIR.

5. Run mfcc-features.py on each subdir of BASE_DIR.

6. Run learn.py according to run instruction provided in the code file.

7. Run tester.py with an audio file to predict the genre 

**Please note-** I have not provided the audio files used by me in the repo. So please replace the directory address wherever necessary in the code with your own local address.
Also while running the tester.py please make sure that the audio file you use is of .wav format sampled at 22050Hz and mono.
