Inputs{ 
  Required Packages: Pytorch, Numpy, Matplotlibrary, Pandas, OS, SKLearn
  
  Pkl files: Two files used for testing that correspond to Cluster 1 and Goes 8 are included in the generated 
  pkl files folder. The python script shouldn't be limited to two files. It should work with any number number 
  of pkl files in a folder identified by the file path though this hasn't been tested.
  }
  
 Outputs{
  Logfile: Includes progress printout and tables with the average predicted efficiency for all 8 model cases. 
  Results for the neural network and linear model cases are in seperate tables.
  
  Plots: Plots for each of the seven inputs and  targets over time are produced.  Additionally, plots of the
  measured vs. predicted for the three target components of the magnetic field for each model case are
  produced.
  
  Tensorboard: Logfiles that include informatin on training and test lost by epoch should be available and
  viewable using tensorboard.
  }

Instructions for running{
1. If all required packages and pkl files are in place, the .py file should be able to be run as is without problem.
2. The tensorboard function has been kind of finicky so that would be the one component that may not run well.
  - If tensorboard is installed: type tensorboard -- logdir=log into the command line
  - Results should be viewable at (http://localhost:6006)
3. All plots should saved automatically to a plots folder. I created a plots folder manualy and pointed to its relative path
4. The logfile should save to the where the python script is being run. 
