# BrainMRI-preproc
Comprehensive and consistent brain MRI images pre-processing pipeline for Deep Learning applications enablingthe creation of a congruous data-sets.

This comprehensive  pre-processing  pipeline  able  to  prepare  raw MRI brain images data-sets (with the correspondinglesions masks, if present) so that they can be directlyfed  into  Deep  Learning  architectures.  The  pipelinehas been implemented entirely in Python and particular attention has been paid in giving the possibility to directly access each functions’ parameters to allow further customizations/optimizations. 

* The pipeline guarantees customization;
* DICOM and NIfTI formats are supported;
* Pre-processing steps consists in image registration, brain extraction and bias field correction;
* Processed images are stored in PNG format.


#  Important links
* [ANTsPy](https://antspy.readthedocs.io/en/latest/) documentation. 

# Dependencies

    Python >= 3.5
    Numpy >= 1.11
    matplotlib >= 1.5.1
    antspy >= 0.2.9
    dicom2nifti >= 2.3.0
    SimpleITK >= 2.0.2
    
# Installation

Install the python packages pip, numpy, matplotlib, antspy, dicom2nifti, simpleitk. If you have a python virtual environment, just run:

    $ pip install numpy matplotlib antspy dicom2nifti simpleitk

If not, make sure to install pip (run: 'sudo apt-get install python-pip'). 

Finally, install pypreprocess itself by running the following in the pypreprocess:

    $ python setup.py install --user

or simply 'python setup.py install' in a virtual environment.

# Development
You can check the latest version of the code with the command:

    $ git clone https://github.com/aSofworkOrange/mriprocess.git
    
