"""
iFlow Textual User Interface (TUI)
Asks user input and starts program selection

Date: 02-11-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
import logging
from .Program import Program
from src.util.diagnostics.NoInputFileException import NoInputFileException
import time
from nifty.Timer import Timer
import os
import importlib


class iFlowTUI:   
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self):
        return

    def start(self, version):
        """Display the menu, wait for user input and then run the program

        Parameters:
            version - version number
        """
        # display menu and ask for input file path
        filePath = None
        cwdpath = ''
        while not filePath:
            cwdpath, filePath = self.displayMenu(version)

        # merge working directory and file path
        totalPath = os.path.join(cwdpath, filePath)

        # call program selector
        program = Program(cwdpath, totalPath)
        timer = Timer()
        timer.tic()
        try:
            program.run()
        except NoInputFileException as e:
            print(e.message)
        timer.toc()
        self.logger.info(timer.string('\n'
                                      'iFlow run time'))

        # upon closing, ask input before ending. This to keep plot windows open.
        # self.askClose()

        return

    def displayMenu(self, version):
        """Display version and ask for input file path.
        If available, up to the five most recent input files are given in a quick-select menu.
        Press enter to get the last used input file, press number 1 to 5 for one of the files from the list,
        or enter the path to a different input file

        Parameters:
            version - version number

        Returns:
            path to input file
        """
        # import the self.config_menu file or create
        try:
            from . import config_menu
        except:
            with open('src/config_menu.py', 'w') as file:
                file.write('LASTINPUTFILES = []\n')
                file.write('CWD = ""')
            from . import config_menu
        self.config_menu = config_menu

        # get last files and cwd
        lastfiles = self.__getRecentFiles()
        cwdpath = self.__getCWD()

        # 1. Print menu
        #   1a. title
        print("iFlow version %s" % str(version))
        print('')

        #   1b. Workding-directory
        if cwdpath:
            print('Current working directory set to '+cwdpath+'.')
        else:
            print('No working directory set. Now working from the iFlow root directory.')
        print('Enter cwd to change the working directory.')
        print('')

        #   1c. Ask for file
        if lastfiles:
            print('Please choose an input file from the list of recent files:')
            for i, path in enumerate(lastfiles):
                print('\t' + str(i+1) + '\t' + path)
            filePath = str(input("or enter the path to a new input file: "))
            print('')
        else:
            filePath = str(input("Please enter the path to a new input file: "))
            print('')

        # 2. Handle menu output
        #   2a. change working directory
        if filePath == 'cwd' or filePath == 'CWD':
            newcwd = str(input('Please enter the path to the new working directory (entering no path will return the iFlow root directory):\n'))
            if newcwd:
                self.__updateCWD(newcwd)
            filePath = None

        #   2b. or select file
        else:
            # if no input is given, take first item from recent list
            if not filePath:
                filePath = lastfiles[0]

            # if integer, try to take the number from the list. Else find new input file with the name given
            else:
                try:
                    filePath = lastfiles[int(filePath)-1]
                except ValueError:
                    pass
                except IndexError:   # when filepath is integer, but exceeds the index of lastfiles
                    print('Index exceeds list of recent files. Please enter a different path.')
                    filePath = None

            # update list of recent files
            if filePath:
                self.__updateRecentFiles(filePath)

        return cwdpath, filePath

    def askClose(self):
        """Flush all loggers and notify user that the computation is done and wait until any key is entered.
        """
        logging.shutdown()
        time.sleep(0.05)    # short pause to make sure that log is indeed flushed before final statement
        input("Done. Press enter to close all windows and end the program.")
        return

    def __getRecentFiles(self):
        """Get list containing the paths to up to the last five used input files from the config file.

        Returns:
            list containing 0 to 5 input file paths
        """
        try:
            lastfiles = self.config_menu.LASTINPUTFILES
        except:
            lastfiles =[]
        return lastfiles

    def __getCWD(self):
        """Get last used working directory from the config file or an empty string if it does not exist.

        Returns:
            string path to cwd
        """
        # Check if the config menu file exists
        try:
            importlib.reload(self.config_menu)
            cwdpath = self.config_menu.CWD
        except:
            cwdpath = ''
        return cwdpath

    def __updateRecentFiles(self, filePath):
        """Update/create list of recently used input files in config.py.
        Removes 'filepath from the list if it was already on. Then puts 'filepath' at position one and
        shifts the other paths down by one position

        Parameter:
            filepath - (str) relavtive/absolute path to the latest used input file.
        """

        # Update the list of last input files or make one
        lastfiles = []
        try:
            lastfiles = self.config_menu.LASTINPUTFILES
            lastfiles.remove(filePath)
        except AttributeError:      #LASTINPUTFILES not found, make it
            lastfiles = []
        except:
            pass
        lastfiles.insert(0,filePath)

        # Trim the list if it is too long
        while len(lastfiles)>5:
            lastfiles.pop(-1)

        # make a new config file
        import fileinput
        for i, line in enumerate(fileinput.input("src/config_menu.py", inplace=True)):
            if i==0:
                print('LASTINPUTFILES = %s' % str(lastfiles))
            if not 'LASTINPUTFILES' in line:
                line = line.replace('\n','')
                print(line)
        return

    def __updateCWD(self, cwdPath):
        """Update/create current working directory in config.py if the directory exists.

        Parameter:
            cwdPath - (str) absolute path to the new cwd
        """
        #0. change path slashes
        cwdPath = cwdPath.replace('\\', '/')

        # 1. test existence of directory
        if not os.path.exists(cwdPath):
            print('Directory could not be located.')
            if not os.path.isabs(cwdPath):
                print('Please enter the absolute path to the working directory, not the relative path.')
            print('')
            return

        # 2. make a new config file
        import fileinput
        for i, line in enumerate(fileinput.input("src/config_menu.py", inplace=True)):
            if i==0:
                print('CWD = %s' % "'"+str(cwdPath)+"'")
            if not 'CWD' == line.strip()[:3]:
                line = line.replace('\n','')
                print(line)
        return




