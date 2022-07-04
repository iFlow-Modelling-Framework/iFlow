"""
Class MemoryProfiler
Creates a profiler for memory based on the built-in tracemalloc module of Python.
"""
import tracemalloc


class MemoryProfiler():


    def __init__(self):
        self.totalsize = []
        self.occasion = []
        return

    def startProfiling(self):
        tracemalloc.start()
        return

    def snapshot(self, occasion=""):
        snapshot = tracemalloc.take_snapshot()
        stat = snapshot.statistics('traceback')
        self.totalsize.append(sum([i.size for i in stat])/1e6)
        self.occasion.append(occasion)
        return

    def plot(self):
        import matplotlib.pyplot as plt
        import step as st

        st.configure()
        print(self.totalsize)
        plt.figure(1, figsize=(2,2))

        plt.plot(self.totalsize, 'ko-')
        plt.xticks(range(0, len(self.occasion)), self.occasion, fontsize=5,rotation=45,ha='right')
        plt.ylabel('Mb')
        plt.title('iFlow memory use')
        st.show()
        return
