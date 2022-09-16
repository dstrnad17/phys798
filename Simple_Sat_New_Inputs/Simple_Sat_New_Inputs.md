The Data Study script reads the pkl file created in the Sat_data script and recreates the dataframe. Using the x, y, and z position from the data frame, the cartesion coordinates are converted to spherical coordinates and the result is appended to the data frame. The script then uses the newly transformed coordinates to create a series of 2d polar projection histograms. 

Data: The data from the pkl file is pulled from the larger satellite data set used to to create the other scripts in this project and consist of satellite ephemeris and other magnetosphere relevant measurement data collected from multiple satellite systems over a 15 year period starting in 2001. For testing purposes a single file with one year of data from a single satellite was used. After a successful test all the satellite data used for analysis

Log file: There are no logfile outputs from this scipt at this time.

Output plots: At this time there are two 1d histogram plots and one 2d polar histgram plots.
  1. 1d - Radius
  2. 1d - Azimuth
  3. 2d polar - Radius vs Azimuth

Analysis: The conversion to spherical coordinates from cartesian coordinates made more sense to me based on my familiarity with satellite orbits. After coordinate conversion, the three histogram plots were used to analized the spatial distribution of satellite measurement data. As expected based on the specific satellites included in the data set, the histogram for the radius peaks between 35-42 thousand kilometers corresponding to a large number of GEO satellite measurements.  For elevation, most of the data falls between +-50 degrees and is sharply peaked at 0 which makes sense for a large number of geo satellite measurements (~0 degrees) and some LEO satellites with orbital inclinations between +-50 degrees. I expected the azimuthal angle measurements to be more uniformly distributed, but there's a small peak at zero.  This might make sense based on stationary GEO satellites except that I think the coordinates are based on a magnetic field reference system (IGRF?) which should make the normally stationary GEOs not stationary.  

Next Steps: 
 1. Investigate reference frames of the measurement data.  
 2. Create and analize histogram data for other measurement data in the data set.
 3. Do some manual correlation analyis between some of the other data measurements.
 4. TBD
