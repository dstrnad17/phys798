The simple satellite script read the pkl file created in the Sat_data script recreated the dataframe.  Using both the x, y, and z position data and Bx, By and Bz magnetic field I fed a modified single layer neural network model originally created in the neural network one layer script.  

Data: The extracted position and field data were scaled and centered (divided by the variance) due to the large difference in magnitude between the position and field data. The full data set was then split in to a training (80%) and a test data (20%).  For troubleshooting purposes, I only used data from one file which included data from one satellite for a one year period.

![](Simple_sat_loss.png)

Model: Starting from the simple neural network model I worked with previously, I had to make modifications to make it work with the new data set. Specificly I had to included that ability to take in three position data points and output the predicted magnetic field components which required me add an additional input and output to the model. Additionally while I maintained the model depth to be a single layer, I did increase the number of degrees of freedom in the layer. I also changed the stocastic gradient decent to ADAM for the additional accuracy. A final change I had to make since I wasn't getting any result initially was changing my activation layer from RELU to Tanh. Note all these changes were made while working with the larger data set. 

Log file: The logfile outputs training loss by iteration and finaly a mean square error for the final output from the training data and the test data.

Output plot: I plotted training loss versus batch number and modeled vs measured field components for each of two cases. In the first case, I used field data as both the input and output during training. In the second case, I used position data as the input.

Analysis: In the plot of training loss vs batch number, there is a visible downward trend that I had not seen in previous runs which was encoraging.  Looking at the equivilent plot when using position as the input to the model, the trend is less visible, but after looking at the data other plot, the initial downward trend is more aparrent. Looking at modeled versus measured field in both input cases, the general trend was that when using the field as both input and target the results are much better as expected.  
