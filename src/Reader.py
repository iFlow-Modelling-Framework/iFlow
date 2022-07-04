"""
Class Reader
Provides methods for opening/closing ascii files as well as methods for reading input in several types of listings.
These methods return DataContainer instances with the input data

Date: 04-05-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from .util import nestedListToDictionary
from src.util.diagnostics import KnownError
from .DataContainer import DataContainer


class Reader:
    # Variables
    
    # Methods
    def __init__(self):
        return

    def open(self, filePath):
        """Opens an ASCII file

        Parameters:
            filePath - path of input file

        Returns:
            file pointer of input file

        Raises:
            Exception - if input file was not found
        """
        try:
            self.filePointer = open(filePath, 'r')
        except IOError as e:                       # throw exception if file is not found
            raise KnownError(('No file found at ' + filePath)) from None

        return

    def close(self):
        """Closes file

        Parameters:
            filePointer - path of input file

        Raises:
            Pass when file could not be closed
        """
        try:
            self.filePointer.close()
        except:                     
            pass

        return 

    def read(self, chapter='', name = '', stop=[]):
        """General reader. Reads data in file between 'chapter' 'name' and the next 'chapter' tag or 'stop' tag (first 'chapter' is inclusive)
        This has the following structure:

            chaptername name (string)
            key value (scalar, string or vector)
            "   "

            key subkey value
                  "   "
            key   "   "
                  "   "

         Parameters:
            chapter - (str, optional) chapter name
                                      Default: ''; read full file
            name - (str, optional) name of 'chapter' to be read
            stop - (list, optional) list of tags to stop this chapter additional to chapter name itself.

        Returns:
            list containing DataContainers with read data for each chapter block read
        """
        self.filePointer.seek(0)    # restarts reading the file
        startChapter = [chapter, name]
        endChapter = [chapter] + stop
        inChapter = False
        inIndent = False

        containerList = []
        dataStructures = []

        #start reading
        for line in self.filePointer:
            line = self.__removeComment(line)
            linesplit = ((line.replace('.', ' ')).replace('\t', ' ')).split(' ')+['']
            if any([i in linesplit for i in endChapter]) and inChapter:
                # stop reading. Convert and clean result
                inChapter = not inChapter
                if inIndent:
                    # add sublist if this has not been done yet
                    dataStructures[-1].append(sublist)

                # convert raw data to a dictionary and load into a DataContainer
                dataStructures = nestedListToDictionary(dataStructures)
                try:
                    dataStructures = DataContainer(dataStructures)
                except:
                    raise KnownError('Incomplete entry in the input file. Please check if there is an unfinished entry with keyword "module".')
                containerList.append(dataStructures)

            if startChapter[0] in linesplit and startChapter[1] in linesplit and not inChapter:
                # start reading
                inChapter = not inChapter
                inIndent = False
                dataStructures = []

            if inChapter:
                #in the right chapter
                if not line == "":
                    # if line starts with a space/tab: this line belongs to a block
                    if line[:1]=='\t' or line[:1]==" ":
                        # start a new block if no block was started yet
                        if not inIndent:
                            sublist = dataStructures[-1][1:]
                            inIndent = True
                        #
                        if isinstance(sublist[0], list):
                            sublist.append(line.split())
                        #
                        else:
                            sublist = [sublist]
                            sublist.append(line.split())
                        del dataStructures[-1][1:]

                    elif (not (line[:1]=='\t' or line[:1]==" ")) and inIndent:
                        inIndent = False
                        dataStructures[-1].append(sublist)
                        dataStructures.append(line.split())
                    else:
                        dataStructures.append(line.split())

        # repeat the block controlling reading stop
        # stop reading. Convert and clean result
        if inChapter:
            if inIndent:
                # add sublist if this has not been done yet
                dataStructures[-1].append(sublist)

            # convert raw data to a dictionary and load into a DataContainer
            dataStructures = nestedListToDictionary(dataStructures)
            dataStructures = DataContainer(dataStructures)
            containerList.append(dataStructures)

        return containerList

    def readline(self, key='', name=''):
        """Scans the file for single lines containing key at the start and name somewhere in the line.
        The lines are stored in separate DataContainers in a list

            key [] name []

         Parameters:
            key - (str, optional) first tag in line
            name - (str, optional) word in the line

        Returns:
            list containing DataContainers with read data for each line found
        """
        self.filePointer.seek(0)    # restarts reading the file
        start = [key, name]
        containerList = []

        #start reading
        for line in self.filePointer:
            line = self.__removeComment(line)

            if start[0] in line[:len(start[0])] and start[1] in line:
                dataStructures = [line.split()]
                dataStructures = nestedListToDictionary(dataStructures)
                dataStructures = DataContainer(dataStructures)
                containerList.append(dataStructures)

        return containerList

    def __removeComment(self, line):
        """Remove part of the line following the comment tag # or ** and strip all trailing spaces

        Parameters:
            line - line to be stripped

        Returns:
            line stripped of comments
        """
        line = line.partition('#')[0]
        line = line.rstrip()
        return line