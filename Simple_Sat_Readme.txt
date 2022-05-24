The simple satellite script read the pkl file created in the Sat_data script recreated the dataframe.  Using the x, y, and z position data I fed a modified single layer neural network model originally created in the neural network one layer script.  Specific modifications I had to make included that ability to take in three position data points and output the predicted magnetic field components.

The full data set was split in to a training and a test data.

A plot of error vs batch number did not show a decreasing trend that I would have expected. This led to efforts to better evaluate model results through plotting predicted vs actual magnetic field by component, error histograms, and identifying correlations.



