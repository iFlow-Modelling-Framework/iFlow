"""


Date:
Authors:
"""
import copy
import logging
import numpy as np
from src.util.diagnostics import KnownError
from nifty import toList
import numbers
import os
from itertools import product


class NetworkSetup:
    # Variables
    logger = logging.getLogger(__name__)
    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Loading network settings')
        # from testestuarysettings_5ch import setup       # make import dynamic based on input file
        # from testestuarysettings_Yangtze import setup
        # from testestuarysettings_SCNPSP import setup
        # from testestuarysettings_SCNPSP_4ch import setup
        # from testestuarysettings import setup
        # from testestuarysettings_3ch_tide import setup
        from testestuarysettings_3ch_baroc import setup
        # from testestuarysettings_2ch import setup
        numberofchannels, geometry, forcing, grid, label= setup()

        d = {}
        d['networksettings'] = {}
        d['networksettings']['numberofchannels'] = numberofchannels
        d['networksettings']['geometry'] = geometry
        d['networksettings']['grid'] = grid
        d['networksettings']['forcing'] = forcing
        d['networksettings']['label'] = label

        return d
