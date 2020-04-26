from src import iFlow


## Init ##
iflow = iFlow.iFlow()       # Initialise iFlow
cwd = r'[ABSOLUTE PATH TO FOLDER]\iflow_DA_Scheldt'  # Set working directory
inputfile = r'input/Scheldt.txt'    # Set inputfile

## Add additional variables ##
Depthcoefficients = [-2.9013e-24,1.4030e-18,-2.4218e-13,1.7490e-8,-5.2141e-4, 15.332]    # polynomial coefficients for the depth profile
Widthcoefficients1 = [-0.02742e-3, 1.8973]    # coefficients set 1 for the width profile
Widthcoefficients2 = [4.9788e-11, -9.213e-6, 1] # coefficients set 2 for the width profile
sf00 = [0.004,0.005,0.008]     # friction coefficient; automatically splits the domain in equal pieces with these roughness coefficients

## Prepare dictionary
input = {'H0':{'type': 'functions.Polynomial', 'C': Depthcoefficients},
         'B0':{'type': 'functions.ExpRationalFunc', 'C1': Widthcoefficients1, 'C2':Widthcoefficients2},
         'sf00': sf00}        # dictionary with additional input parameters

## Run iFlow ##
output = iflow.StartFUI(inputfile, cwd, input)
