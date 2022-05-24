The linear regression exercise script defines three linear models that fit input data to target data using a least squares loss loss function.  All three models are implemented using Facebook developed Pytorch package in python.

Model 1: The minimization function and model are explicitly defined and gradient decent is performed using pytorches implementation in manualy defined loop.

Model 2: The model uses pytorches minimiztion and linear functions as well as stocastic gradient decent.

Model 3: The final model includes a linear layer, a Relu non-linear activation and final linear output layer in a simple nearal network model. It also uses stochastic gradient decent.

Log file: The logfile outputs the mean square error for the three models as well as the error for the matrix inversion solution with the same outputs and inputs.

Output plot: All four solutions are plotted against the input data.


