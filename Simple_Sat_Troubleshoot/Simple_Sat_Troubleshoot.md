The simple satellite script read the pkl file created in the Sat_data script recreated the dataframe.  Using the x, y, and z position data I fed a modified single layer neural network model originally created in the neural network one layer script.  

Data: The extracted position and field data were scaled and centered (divided by the variance) due to the large difference in magnitude between the position and field data. The full data set was then split in to a training (95%) and a test data (5%).

![](Simple_sat_loss.png)

Model: Starting from the simple neural network model I worked with previously, I had to make modifications to make it work with the new data set. Specificly I had to included that ability to take in three position data points and output the predicted magnetic field components which required me add an additional input and output to the model. Additionally while I maintained the model depth to be a single layer, I did increase the number of degrees of freedom in the layer. I also changed the stocastic gradient decent to ADAM for the additional accuracy. A final change I had to make since I wasn't getting any result initially was changing my activation layer from RELU to Tanh. 

Log file: The logfile outputs training loss by iteration and finaly a mean square error for the final output from the training data and the test data.

Output plot: In this case I plotted training loss versus batch number.

Analysis: A plot of error vs batch number did not show a decreasing trend that I would have expected. It was unclear if this was real or an artifact or issue with my model. In addition to this core issue, I found that the results varied from run to run. As was discovered in my simpler models, I ended up having to set a random seed so that it was more consistant. I also manipulated the hyper-parameters a bit so to see if it had an effect. Specifically I modified the learning rate and the number of degrees of freedom in the model, but nothing stood out as being a particular problem. I was also curious if by using all the data, I was averging over any dynamics in the magnetic field over the years that the data measurements were collected which was causing errors. I did take a smaller snipit of data with the hope that there would be less dynmic effects, but there was not a clear cut result. As a next step I worked to better evaluate model results through plotting predicted vs actual magnetic field by component. In a seperate script I have started to evaluate the data itself, plotting error histograms, and identifying correlations.
