## Instructions 

**Generating data. (generate_data.py)**

To generate data run the script generate_data.py with following flags:

- --amount: the size of the data. default: 100 
- --std: the std for the noise. default: 0.01
- --std_xy: the std for the translations distribtion. default: 5 
- --size: size of the images. default:256
- --name: the name of the mrcfile to be created. default: data.mrc

**Expectation Maximization. (EM.py)** 

- --data: path to mrc file. REQUIRED 
- --init: path to the initialization file. default: if none given it uses the average of the data.
- --std: the std of the noise of the data. default: 0.01
- --std_xy: the std of the translations. default: 5
- --N: number of iterations. Default: 5 
