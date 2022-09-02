The simple satellite troubleshoot script read the pkl file created in the Sat_data script and recreated the dataframe.  Using only the Bx, By and Bz magnetic field I fed a modified single layer neural network model originally created in the neural network one layer script.  

Data: The field data were scaled and centered (divided by the variance) since I was reusing code from the simple satellte. The full data set was then split in to a training (80%) and a test data (20%).  For troubleshooting purposes, I only used data from one file which included data from one satellite for a one year period.

![](Simple_sat_loss.png)

Model: The model for this case is the same as simple_sat. The primary difference is that the magnetic field data is used as both input and output. I also ran a case without the non-linear function.

Log file: The logfile outputs training loss by iteration and finaly a mean square error for the final output from the training data and the test data.

Output plot: I plotted training loss versus batch number and modeled vs measured field components for each of two cases. In the first case, I used field data as both the input and output during training. In the second case, I used position data as the input.

Analysis: As expected the fit is much better in this case.  In the case where I removed the non-linear tanh function, the difference is was essentially zero.  This makes sense because in that case I'm essentially multiplying by 1.

Next Steps
  1. Nothing planned unless additional issues appear.
