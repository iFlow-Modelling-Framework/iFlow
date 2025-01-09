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

from .toList import toList
from .Timer import Timer
from .amp_phase_input import amp_phase_input
from .arraydot import arraydot
from .complexAmplitudeProduct import complexAmplitudeProduct
from .derivative import derivative
from .dimensionalAxis import dimensionalAxis
from .dynamicImport import dynamicImport
from .eliminateNegativeFourier import eliminateNegativeFourier
from .fft import fft
from .harmonicDecomposition import absoluteU, signU
from .integrate import integrate
from .invfft import invfft
from .makeRegularGrid import makeRegularGrid
from .pickleload import pickleload
from .polyApproximation import polyApproximation
from .primitive import primitive
from .secondDerivative import secondDerivative
from .splitModuleName import splitModuleName
from .toMatrix import toMatrix
from .invfft2 import invfft2
from .savitzky_golay import savitzky_golay
from .extcumsum import extcumsum
from .makeStaggered2DGrid import makeStaggered2DGrid