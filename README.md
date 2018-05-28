## Project: Seizure onset detection

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pickle](included in python)
- [scipy](https://www.scipy.org/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

Recommend to  install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

### Code and how to run

To run the solution, first you will need to download the data from here : https://www.kaggle.com/c/seizure-detection/data

extract all data into clips directory (such that its clip/Dog_1/Dog_1_segment_1.mat)
In feature_extraction_v2.py go to lines 86, and 90. Change data_path to the directory where everything is extracted. Change save_directory to the pickle_data location.

To run the final solution, open random_forest.py and change line 21 to point at the pickle data location.

To get some of the visuals I used in my report, you can use Data Exploration.ipynb 

### Data

Data is obtained from the kaggle competition 3 years ago "UPenn and Mayo Clinic's Seizure Detection Challenge" (https://www.kaggle.com/c/seizure-detection). Under the section "Data", select 
clips.tar.gz and hit "Download", the file should be about 10.82GB. Data consists of "clips" which are 1 Second length ECOG data in .mat MATLAB 5 format. When loaded with scipy.io.loadmat, it comes as dictionary with following keys/features:

**Features**
- `latency`: how far in the seizure, in seconds (ictal data only)
- `__header__`: Description of the .mat file
- `globals`: empty
- `channels`: name of the channels, in EEG/ECOG data; labels 
- `freq`: sampling frequency of data
- `data`: 1 second clip of data in the format [channels x data point]

**Target Variable**
- `seizure`: ictal(in seizure), early ictal(within first 15 seconds of seizure onset), interictal(non seizure)

