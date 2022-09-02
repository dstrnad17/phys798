The simple satellite Alternative model 2 script read the pkl file created in the Sat_data script and recreated the dataframe.  Using the x, y, and z position data I fed the simple sat model.

Data: The extracted position and field data were scaled and centered (divided by the variance) due to the large difference in magnitude between the position and field data. The data set was then split in to a training (95%) and a test data (5%).

![](Simple_sat_loss.png)

Model: Starting from the simple neural network model I worked with previously, I changed the output from 3 outputs to a single output.  All other variable remained the same. 

Log file: The logfile outputs training loss by iteration and finaly a mean square error for the final output from the training data and the test data.

Output plot: In this case I plotted training loss versus batch number.

Analysis: The results from this run were equivilent to one single output from the simple sat model. Based on that there was little value (except for reduced run time) in a seperate model.

Next Steps:
 1. Run addition files to look for consistancy to verify that there is not significant difference between a single simple sat single layer model output and this single output model.
 

