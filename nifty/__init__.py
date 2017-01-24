"""
Native iFlow Tools (NiFTy) package

This package provides tools to be used in iFlow.
You are welcome to add your functions native to iFlow to the package. Please respect the following guidelines:
- Make your tool as general as possible. Particularly allow as much generality in the number and order of dimensions.
- Include only python scripts or cross-platform compiled libraries from other langues.
- Include only one public method or class per file. This method/class should have the same name as the file.
- Include a description of how to use the tool in the docstring
- Document the required type of input and the type of output that may be expected in the docstring.
- Include comments throughout the code, so that interested users may get to understand your script in detail
"""

from toList import toList
from derivative import derivative
from complexAmplitudeProduct import complexAmplitudeProduct
from dynamicImport import dynamicImport
from eliminateNegativeFourier import eliminateNegativeFourier
from harmonicDecomposition import absoluteU, signU
from integrate import integrate
from makeRegularGrid import makeRegularGrid
from polyApproximation import polyApproximation
from primitive import primitive
from secondDerivative import secondDerivative
from splitModuleName import splitModuleName
from Timer import Timer
from dimensionalAxis import dimensionalAxis
from bandedMatrixMultiplication import bandedMatrixMultiplication
from fft import fft
from invfft import invfft
from pickleload import pickleload
from amp_phase_input import amp_phase_input
from runCallStackLoop import runCallStackLoop
from BypassDatacontainer import BypassDatacontainer
from scalemax import scalemax
from arraydot import arraydot
from toMatrix import toMatrix

# loads all file names in this package that are python files and that do not start with _
#__all__ = [os.path.basename(f)[:-3] for f in os.listdir(os.path.dirname(__file__)) if f.endswith('.py') and not f.startswith('_')]


#for mod in __all__:
#    exec('from '+mod+ ' import '+mod)
