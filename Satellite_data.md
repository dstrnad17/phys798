The satellite data script read files that included satellite data I had previously downloaded and created a Pandas dataframe. The resulting dataframe was then saved as a pkl file to be accessed later using the Simple_sat script.

Log file: The logfile outputs results from using the pandas head() function which is to say prints a truncated versions of the first five rows of the dataframe. 

Output plot: N/A

Analysis: One of the major issues I had to work through was the data formatting in the file.  The data was not uniformly delinated so I ended up having to read in the lables which were comma seperated, then I used a loop to run through the rest of the file which was seperated by spaces. Ultimately, I was able to fix the issue, reading in all files I downloaded and properly labeling all 31 variables correctly in a single dataframe. I was also able to create the pikle file correctly based on tests when I read it in and converted it back to a dataframe in the Simple_Sat script.
