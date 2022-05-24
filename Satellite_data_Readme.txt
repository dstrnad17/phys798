The satellite data script read files that included satellite data I had previously downloaded and created a Pandas dataframe. The resulting dataframe was then saved as a pkl file to be accessed later using the Simple_sat script.

One of the major issues I had to work through was the data formatting in the file.  The data was not uniformly delinated so I ended up having to read in the lables which were comma seperated, then I used a loop to run through the rest of the file which was seperated by spaces.

