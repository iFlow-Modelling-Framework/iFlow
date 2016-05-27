from src import iFlow
import cProfile
import pstats

cProfile.run('iFlow.iFlow()', sort='cumtime')
