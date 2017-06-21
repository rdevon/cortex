fmri tester

This is a code to test the volumetric convolutional autoencoder using fmri data. The data used is not preprocessed, I will be replacing the data with preprocessed data soon. 
We are using 3 .nii files per subject representing 3 different runs and number of subjects is 5. So there are in all 15 such files. You can use your own data to try out this code.

Steps:

1) Place all the .nii files in a folder. Name them 's0c0' for subject 1 run 1, 's0c1' for subject one run 2, and so on.
2) Change the directory path variable in the testfmri.py code to the folder where the data is residing.
3) Place the VCAE.py file in the same directory as testfmri.py
4) run testfmri.py

You will see the cost decreasing every epoch. As the number of samples for fmri are very less, an autoencoder cannot be trained effectively using fmri data. But this can be used to device other complex structures.
