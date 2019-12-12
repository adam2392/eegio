# INSTALLATION GUIDE

Setup virtual environment via Conda inside your Unix-friendly terminal (aka Mac, or Linux) is recommended (see https://docs.anaconda.com/anaconda/install/):

    conda create -n eegio # creates conda env
    conda activate eegio  # activates the environment
    conda config --add channels conda-forge # add extra channels necessary
    conda install numpy pandas mne scikit-learn scipy seaborn matplotlib pyedflib xlrd
    
To install from githug, run this command inside your virtual environment:

    pip install -e git+https://github.com/adam2392/eegio#egg=eegio
