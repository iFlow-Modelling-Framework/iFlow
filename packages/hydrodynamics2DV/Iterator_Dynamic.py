"""
Dynamic Spring-Neap Iterator

Update: 11-10-2023
Update Author: Y.M. Dijkstra (reformulation to implicit code and reformulation to controller structure)
Original date: 24-Jan-19
Original author: D.D. Bouwman
"""

import numpy as np
from src.util import mergeDicts
import nifty as ny
import logging


class Iterator_Dynamic():
    # Variables
    logger = logging.getLogger(__name__)
    OutputList = ['zeta0', 'zeta1', 'u0', 'u1', 'c0', 'c1', 'a', 'f', 'F', 'T', 'sediment_transport', 'Av', 'Ri'] # a -> stock. f -> erodibility
    verticalsize = [1, 1, None, None, None, None, 1, 1, 1, 1, 1, 1, 1]
    frequencysize = [None, None, None, None, None, None, 1, 1, 1, 1, 1, 1, 1]

    # Methods
    def __init__(self, input, block):       
        self.input = input
        self.block = block
        return
        
    def run(self):
        ################################################################################################################
        # Initialise
        ################################################################################################################
        jmax = self.input.v('grid', 'maxIndex', 'x')
        kmax = self.input.v('grid', 'maxIndex', 'z')
        fmax = self.input.v('grid', 'maxIndex', 'f')
        
        secondsPerDay = 3600*24
        t2 = (self.input.v('MTS_grid', 'axis', 't')*(self.input.v('MTS_grid', 'high', 't') - self.input.v('MTS_grid', 'low', 't')) + self.input.v('MTS_grid', 'low', 't')).flatten()*secondsPerDay     # slow time axis in seconds (NB MTS_grid stores it in days)
        t2max = self.input.v('MTS_grid', 'maxIndex', 't')
        store = {}

        # compute seaward tidal BCs
        bc_0, bc_1 = self.calc_subtidal_bcs(t2)

        # Discharge input
        Qarray = ny.toList(self.input.v('Q1')) 

        if len(Qarray) == 1:
            Qarray = Qarray*(t2max+1)

        elif len(Qarray) != t2max+1:
            from src.util.diagnostics.KnownError import KnownError
            raise KnownError('Number of elements of Q1 incompatible; should be 1, equal to number of time steps.')

        ################################################################################################################
        # time integrate
        ################################################################################################################
        for iteration in range(0, t2max):
            self.logger.info('Time integration iteration ' + str(iteration+1) + '/' + str(t2max))
            ## 1. Prepare input
            #       Offer the input only at the next time step to allow for implicit timestepping schemes only.
            d_in = {}
            d_in['dt'] = np.real(t2[iteration+1])-np.real(t2[iteration])

            # Seaward boundary input
            d_in['A0'] = np.abs(bc_0[iteration+1, :])
            d_in['A1'] = np.abs(bc_1[iteration+1, :])
            d_in['phase0'] = np.angle(bc_0[iteration+1, :], deg=True)
            d_in['phase1'] = np.angle(bc_1[iteration+1, :], deg=True)

            # Discharge input
            d_in['Q1'] = Qarray[iteration+1]
            
            # Sediment stock input
            if iteration==0:
                IC = self.input.v('initial_condition_sediment')
                d_in['initial_condition_sediment'] = IC
                if IC == 'stock':
                    d_in['Sinit'] = self.input.v('Sinit', range(0, jmax+1))
                elif IC == 'erodibility':
                    d_in['finit'] = self.input.v('finit', range(0, jmax+1))
            else:
                dc = self.block.getInput()
                dc.merge(self.block.getOutput())
                d_in['Sinit'] = dc.v('a', range(0, jmax+1), 0, 0)
                d_in['initial_condition_sediment'] = 'stock'

            ## 2. Run the model
            self.block.addInputData(d_in)  # add the input to the controlled block of modules
            init = False
            if iteration==0:
                init = True
            self.block.run(init=init)

            ## 3. Gather input and output from the block
            dc = self.block.getInput()
            dc.merge(self.block.getOutput())

            ## 4. Process output
            list_all_keys = dc.getAllKeys() # Create current subkeys-list
            for qq, component in enumerate(self.OutputList):
                self.subkeyList = [i[1:] for i in list_all_keys if component == i[0]]
                for varkey in self.subkeyList:
                    if component == varkey:     # if no subkeys were found, set the varkey to an empty tuple
                        varkey = ()
                    totalkey = (component,) + varkey

                    # load the value as numerical data from the output
                    f = np.arange(0, fmax+1)
                    z = np.linspace(0,1, kmax+1)
                    if self.verticalsize[qq]==1:
                        z = [0]
                    if self.frequencysize[qq]==1:
                        f = [0]
                    value = dc.v(component, *varkey, x=np.linspace(0,1, jmax+1), z=z, f=f)

                    # After the first iteration, create the numpy array for computed values. Copy this data to the first and second time steps. The 0-index should be ignored
                    if iteration == 0:
                        store[totalkey] = np.expand_dims(value, axis=-1)

                    store[totalkey] = np.append(store[totalkey], np.expand_dims(value, axis=-1), axis=-1)

        ################################################################################################################
        # Put output in MTS variables
        ################################################################################################################
        d = {}
        for key in store.keys():
            d = mergeDicts(d, self.setDictHierarchy(key, store))   # Merge existing dict with calculated data

        return d
        
    def calc_subtidal_bcs(self, t2):
        """Define the seaward boundary conditions for all t.""" 
        #np.tile is used to match the dimensions of t2 (nsteps,) and inputs (3,)
        
        ## Load input
        AM0 = self.input.v('AM0')
        AS0 = self.input.v('AS0')
        AM1 = self.input.v('AM1')
        AS1 = self.input.v('AS1')
        
        phaseM0 = self.input.v('phaseM0')
        phaseS0 = self.input.v('phaseS0')
        phaseM1 = self.input.v('phaseM1')
        phaseS1 = self.input.v('phaseS1')
        
        sigmaM0 = self.input.v('OMEGA')
        sigmaS0 = self.input.v('sigmaS2')

        sigma_sn0 = (sigmaS0 - sigmaM0)
        sigma_sn1 = 2*sigma_sn0
        
        ## Process
        AM0 = np.asarray(AM0, dtype='complex')
        AM1 = np.asarray(AM1, dtype='complex')
        
        AS0 = np.tile(np.asarray(AS0, dtype='complex'), (len(t2), 1))  # Note that the amplitudes are tiled
        AS1 = np.tile(np.asarray(AS1, dtype='complex'), (len(t2), 1))  # Note that the amplitudes are tiled
        
        # Phases on input are in degrees, convert to radians
        phaseM0 = np.asarray(phaseM0, dtype='complex')/180*np.pi
        phaseS0 = np.asarray(phaseS0, dtype='complex')/180*np.pi
        phaseM1 = np.asarray(phaseM1, dtype='complex')/180*np.pi
        phaseS1 = np.asarray(phaseS1, dtype='complex')/180*np.pi

        sigma_sn0 = np.asarray(sigma_sn0, dtype='complex')
        sigma_sn1 = np.asarray(sigma_sn1, dtype='complex')
        
        T2_array = np.transpose(np.tile(t2, (len(phaseS0), 1))) # Make an array such that "t_2 + phase" can be calculated elementwise
        
        res_0 = np.tile(AM0 * np.exp(1j * phaseM0), (len(t2), 1)) #Dummy variable
        res_1 = np.tile(AM1 * np.exp(1j * phaseM1), (len(t2), 1)) #Dummy variable
        
        # Define the leading and first order seaward boundary conditions
        bc_0 = res_0 + AS0 * np.exp(1j * (sigma_sn0 * T2_array + np.tile(phaseS0, (len(t2), 1) )))
        bc_1 = res_1 + AS1 * np.exp(1j * (sigma_sn1 * T2_array + np.tile(phaseS1, (len(t2), 1) )))

        return bc_0, bc_1
    
    def setDictHierarchy(self, totalkey, store):
        """convert the data from 'self.store' to a nestled dictionaries with a hierarchy according to the specified tuple 'totalkey' """
        if len(totalkey)>1:
            d = {totalkey[-1] : store[totalkey]} #store savedata in the lowest dictionary
            for ii in reversed(totalkey[1:-1]):
                d = {ii : d} #This is were we nestle the dict in the higher dict
            d = {'MTS_'+totalkey[0]: d} #Give first key a 'MTS_' prefix to distinct it as a t2-dependent variable

        if len(totalkey) == 1:
            d = {'MTS_'+totalkey[0] : store[totalkey]}
        d['__variableOnGrid'] = {}
        d['__variableOnGrid']['MTS_'+totalkey[0]] = 'MTS_grid'
        return d
            
        