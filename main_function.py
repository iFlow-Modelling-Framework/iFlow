from src import iFlow
import os

## Init ##
iflow = iFlow.iFlow()       # Initialise iFlow
cwd = r'D:\Work\PhD\iFlow\iFlow3\Examples\Tutorial'  # Set working directory
inputfile = r'input\inputEMS.txt'    # Set inputfile

## Add additional variables ##
Depthcoefficients1 = [8.09495397e-19, -6.92278797e-14, 2.34165496e-09, -1.86980848e-04, 1.05000000e+01]    # polynomial coefficients for the depth profile
Depthcoefficients2 = 50332    # xl
Widthcoefficients = [-2.05001187e-20, 3.26623360e-15, -1.77496965e-10, 3.82812572e-06, -4.18442587e-02, 7.62139923e+02]    # coefficients set 1 for the width profile

## Prepare dictionary
input = {'H0':{'type': 'functions.PolynomialLinear', 'C': Depthcoefficients1, 'XL': Depthcoefficients2},
         'B0':{'type': 'functions.Polynomial', 'C': Widthcoefficients}}        # dictionary with additional input parameters


## Run iFlow ##
iFlowBlock = iflow.initialise(inputfile, cwd, input)
iFlowBlock.instantiateModule()
iFlowBlock.run()
out1 = iFlowBlock.getOutput()
# profiler = iFlowBlock.getProfiler()     # optional


## Run iFlow again ##
# profiler.snapshot('before copy')
iFlowBlock2,_ = iFlowBlock.deepcopy()
# profiler.snapshot('after copy')
input2 = {'L':100e3}
iFlowBlock2.addInputData(input2)
iFlowBlock2.instantiateModule()
iFlowBlock2.run()
out2 = iFlowBlock2.getOutput()

profiler = iFlowBlock2.getProfiler()
if profiler is not None:
    profiler.plot()
else:
    print('No profiler set')
# output = iFlowBlock.getOutput()