"""
CalcQ to compute the discharge corresponding to the upstream boundary and tributaries

Date: 17-Aug-20
Authors: D.M.L. Horemans
"""
import logging
import nifty as nty

class calcQ:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Module calcQ.py is running')
        d = {}
        QTrib = nty.toList(self.input.v('QTributary'))

        Q1 = self.input.v('Q') * QTrib[0] / 100.
        Qsource = [self.input.v('Q')*QTrib[1]/100., self.input.v('Q')*QTrib[2]/100.]

        d['Q1'] = Q1
        d['Qsource'] = Qsource
        return d