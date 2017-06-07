Convolutional autoencoder for 2D data:

1) There are two pieces to this code: CAE.py and testCAE.py

2) CAE.py contains the autoencoder. We do not run this code directly. We keep this code in the same directory as testCAE.py.

3) In testCAE.py, we use mnist dataset to test the functionality of our autoencoder. We import the autoencoder function strored in CAE.py and use it in the testCAE.py.

4) Due to the modular structure we can use the autoencoder for a variety of purposes, we just have to import it in our code like we did in testCAE.py.

5) What we need to do:
	a) Keep both files in the same directory.
	b) Note the log directory present in the top of the code in testCAE.py. YOu may change this directory as per your convenience. Tensorboard summary file will be stored here (summary only 		contains the learning curve).
	c) Set the input parameters while calling the autoencoder function in the testCAE.py file. Default paramenters are already set. If you want to know more about the parameters then read the 	commented section in the autoencoder function in the CAE.py file.
	d) Run testCAE.py file. You will see the costs per epoch on the terminal and finally a set of test images will be reconstructed using the autoencoder. The summary file will be in the log 		directory which can be used to see the learning curve.
	
	
	

Convolutional autoencoder for volumetric data:

1) There are three pieces to this code: GDATA.py, VCAE.py and testVCAE.py.

2) VCAE.py contains the volumetric autoencoder. We do not run this code directly. We keep this code in the same directory as testCAE.py. 

3) GDATA.py contains a function that takes a .nii format file and creates random files using it. This is our random data generator. We do not run this code directly either, and keep it in the 	same directory as testVCAE.py.

4) In testVCAE.py, we use an instance of the random generator to create volumetric data, and use this to verify the functioning of our autoencoder. In the same way as the 2D version, we import the 3D autoencoder from the VCAE.py file.

5) What we need to do:
	a) Keep all three files in the same directory.
	b) Set the input parameters while calling the autoencoder function in the testVCAE.py file. Default paramenters are already set. If you want to know more about the parameters then read the 		commented section in the autoencoder function in the CAE.py file.
	c) Run testVCAE.py file. You will see the costs per epoch on the terminal, Note the decreasing cost with every epoch.
