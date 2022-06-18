The neural network one layer script implements a single layer neural network based on the neural network implemented in the linear exercise.  The original intention of creating a seperate script with just the single layer network was to explore some of the non-deterministic behavior I was seeing in the results from the neural network model.

Data used to test the models was created using numpys rand() function to define an X value as well as an random offset. Y was calculated by added the offset to the random X. Both X and Y were converted to tensors to make them compatable with pytorch

![](NN_one_layer.png)

Model: The model includes a linear layer, a Relu non-linear activation and final linear output layer in a simple nearal network model. It also uses stochastic gradient decent.

Log file: The logfile outputs loss by iteration and mean square error for the one layer neural network model.

Output plot: The neural network model solution is plotted against the true non-noisy solution as well as the noisy data points.

Analysis: Visually the single layer neural network is very close to the true solution and the mean square error solution is only .179 which is sufficiently close.  This is expected based on final results in the linear excercise.  One of the problems I found was that the result was sensitive to how the network was initialized.  I was initializing randomly to start with, and the results from run to run would vary. In some cases it would vary significantly.  After researching the problem, I found that that this a common problem with some neural networks, particularly shallow networks for regression type tasks.
