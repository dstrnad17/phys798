The satellite data script read files that included satellite data I had previously downloaded and created a Pandas dataframe. The resulting dataframe was then saved as a pkl file to be accessed later using the Simple_sat script.

Log file: The logfile outputs the mean square error for the three models as well as the error for the matrix inversion solution with the same outputs and inputs.

Output plot: All four solutions are plotted against the input data.

Analysis: One of the major issues I had to work through was the data formatting in the file.  The data was not uniformly delinated so I ended up having to read in the lables which were comma seperated, then I used a loop to run through the rest of the file which was seperated by spaces.
