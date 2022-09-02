


The Data Study script reads the pkl file created in the Sat_data script and recreates the dataframe. Using the x, y, and z position from the data frame, the cartesion coordinates are converted to spherical coordinates and the result is appended to the data frame. The script then uses the newly transformed coordinates to create a series of 2d polar projection histograms. 

Data: The data from the pkl file is pulled from the larger satellite data set used to to create the other scripts in this project and consist of satellite ephemeris and other magnetosphere relevant measurement data collected from multiple satellite systems over a 15 year period starting in 2001. For testing purposes a single file with one year of data from a single satellite was used.

Log file: There are no logfile outputs from this scipt at this time.

Output plots: At this time there are three 2d polar histgram plots.
  1. Radius vs. Azimuth
  2. 

Analysis: A plot of error vs batch number did not show a decreasing trend that I would have expected. It was unclear if this was real or an artifact or issue with my model. In addition to this core issue, I found that the results varied from run to run. As was discovered in my simpler models, I ended up having to set a random seed so that it was more consistant. I also manipulated the hyper-parameters a bit so to see if it had an effect. Specifically I modified the learning rate and the number of degrees of freedom in the model, but nothing stood out as being a particular problem. I was also curious if by using all the data, I was averging over any dynamics in the magnetic field over the years that the data measurements were collected which was causing errors. I did take a smaller snipit of data with the hope that there would be less dynmic effects, but there was not a clear cut result. As a next step I worked to better evaluate model results through plotting predicted vs actual magnetic field by component. In a seperate script I have started to evaluate the data itself, plotting error histograms, and identifying correlations.
