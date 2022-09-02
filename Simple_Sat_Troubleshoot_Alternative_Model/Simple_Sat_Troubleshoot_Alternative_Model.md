The simple satellite script read the pkl file created in the Sat_data script recreated the dataframe.  Using the x, y, and z position data I fed a modified single layer neural network model originally created in the neural network one layer script.  

Data: The extracted position and field data were scaled and centered (divided by the variance) due to the large difference in magnitude between the position and field data. The full data set was then split in to a training (95%) and a test data (5%).

![](Simple_sat_loss.png)

Model: Starting from the simple neural network model I worked with previously, I added a second layer.  All other variable remained the same. 

Log file: The logfile outputs training loss by iteration and finaly a mean square error for the final output from the training data and the test data.

Output plot: In this case I plotted training loss versus batch number.

Analysis: There were no obvious difference between the simple_sat single layer model and this two layer model.  This is most likely an artifact from only using limited data.

Next Steps:
 1. Run addition files to look for consistancy.
 2. Use the complete data set to look for improved performance.
