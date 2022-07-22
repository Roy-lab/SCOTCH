# pyNMTF

### Description 
A pytorch implementation of the regularized non-negative matrix tri-factorization algorithm. Compatible with CUDA enabled GPU's. The code was developed in an enviroment with the following packages. This uses the cuda11.6 package.  

*python 3.8.1 
*pytorch 1.12.0 py3.8_cuda11.6_cudnn8.3.2_0
*torchaudio  0.12.0 
*torchvision 0.13.0
*cudatoolkit 11.6.0
*numpy 1.22.3
*pandas 1.4.3 

### Methods/Outputs
The goal of this function is to take a matrix X of dimension n x m and factor it into three matrices, U in n x k1, S in k1 x k2, and V in k2 x m, such that the difference between X and the product U x S x V is minimized. Formally this is resolved by minimizing the objective || X - U S V ||2^2. The program outputs the three product matrices. 

#### Orthoganilty regularization
The model allows for the addition of orthoganility regularization on the U and/or the V matrix. The idea of this regularization is to generate unique factors. It is particularly useful in resolving unique clusters from the lower dimensional embedding. It is recommend to put orthoganility reg. on either or both factors such that they are unique. If both are selected, then the S matrix will have more overlap. 

#### Sparsity regularization. 
This works simililarly to the ortho regularization. In this case, the number of non-zero entries in each factor in penalized. It again will result in more unique factors although. 

### Arguments 

--in_file:	required argument	The file containing the tab delimited X matrix if full form. example: test/A.txt
--k1:		required argument	The lower dimension of the U matrix. example 2 
--k2:		required argument	The lower dimension of the V matrix. example 3
--lU:		optional		strength of ortho reg on U. example 1000 (value dependent on number of features along rows) 
--lV:		optional		strength of ortho reg on V. example 1000 (value dependent on number of features along cols)
--aU:		optional		strength of sparsity reg on U. example 1000 (value dependent on number of features along rows)
--aV:		optional		strength of sparsity reg on V. example 1000 (value dependent on number of featurs along cols)
--verbose	optional		if included, time and convergence info will be printed to terminal
--seed		optional		random seed used to initialize U, S, and V. Default 1010
--max_iter	optional		number of epochs to attempt prior to output. Default 100 
--term_tol	optional		the relative change in error prior to completion. Default 1e-5 	
--out_dir	optional		output directory to print U, S, and V.
--cpu		optional		if included, defuaults to using the CPU


