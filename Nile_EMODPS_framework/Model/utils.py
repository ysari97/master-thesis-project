# Utils and myFile classes. The latter is to represent a matrix
# to-be-read from an external file

import numpy as np

class myFile:
    """
    A class to represent a matrix to-be-read from an external file.

    Attributes
    ----------
    filename : str
        Path to the data file
    row : int
        Specified number of rows of the file
    col : int
        Specified number of columns of the file
    """
    def __init__(self):
        self.filename = None
        self.row = None
        self.col = None

class utils:
    """
    A class that provides the static methods which
    are meant to be used by objects from multiple classes.
    """

    @staticmethod
    def loadVector(file_name, l):
        """Loads the vector as a np.array with length=l from the specified
        file. In case the specified path does not exist, raises a print
        statement.

        Parameters
        ----------
        file_name : str
            Path to the file to-be-read
        l : int
            Length of the expected array

        Returns
        -------
        output : np.array
            The array as read from the specified external file. Elements are
            of 'float' type 
        """
        
        output = np.empty(0)
        try:
            with open(file_name, "r") as input:
                try:
                    count = 0
                    for line in input:
                        for word in line.split():
                            output = np.append(output, float(word))
                            count += 1
                            if (count == l):
                                break
                except:
                    print("Unable to iterate")
        except:
            print("Unable to open file" + "   " + file_name)

        return output

    @staticmethod
    def loadIntVector(file_name, l):
        """Loads the vector as a np.array with length=l from the specified
        file (elements casted as integer). In case the specified path does
        not exist, raises a print statement.

        Parameters
        ----------
        file_name : str
            Path to the file to-be-read
        l : int
            Length of the expected array

        Returns
        -------
        output : np.array
            The array as read from the specified external file. Elements are
            of 'int' type 
        """
        
        output = np.empty(0)
        try:
            with open(file_name, "r") as input:
                try:
                    for i, line in enumerate(input):
                        output = np.append(output, int(float(line.strip())))
                        if (i == l-1):
                            break
                except:
                    print("Unable to iterate")
        except:
            print("Unable to open file" + "   " + file_name)

        return output


    @staticmethod
    def loadMatrix(file_name, row, col):
        """Loads the matrix as a np.array with vertical length=row,
        horizontal length=col from the specified file. In case the
        specified path does not exist, raises a print statement.

        Parameters
        ----------
        file_name : str
            Path to the file to-be-read
        row : int
            Number of rows of the matrix to read
        col : int
            Number of columns of the matrix to read

        Returns
        -------
        output : np.array
            The matrix as read from the specified external file. Elements are
            of 'float' type 
        """
        
        output = np.empty((row, col))
        try:
            with open(file_name, "r") as input:
                for i, line in enumerate(input):
                    templist = line.split()
                    for j in range(col):
                        output[i][j] = float(templist[j])
        except:
            print("Unable to open file" + "   " + file_name)

        return output

    @staticmethod
    def interp_lin(X, Y, x):
        """Takes two vectors and a number. Based on the relative position
        of the number in the first vector, returns a number that has the
        equivalent relative position in the second vector.

        Parameters
        ----------
        X : np.array
            The array/vector that defines the axis for the input
        Y : np.array
            The array/vector that defines the axis for the output
        x : float
            The input for which an extrapolation is seeked

        Returns
        -------
        y : float
            Extrapolated output 
        """
        dim = X.size - 1

        # extreme cases (x<X(0) or x>X(end): extrapolation
        if(x <= X[0]):
            y = (Y[1] - Y[0]) / (X[1] - X[0]) * (x - X[0]) + Y[0] 
            return y
        
        if(x >= X[dim]):
            y = Y[dim] + (Y[dim] - Y[dim-1]) / (X[dim] - X[dim-1]) * \
                (x - X[dim])
            return y
        
        # otherwise
        # [ x - X(A) ] / [ X(B) - x ] = [ y - Y(A) ] / [ Y(B) - y ]
        # y = [ Y(B)*x - X(A)*Y(B) + X(B)*Y(A) - x*Y(A) ] / [ X(B) - X(A) ]
        delta = 0.0
        min_d = 100000000000.0
        j = -99

        for i in range(X.size):
            if (X[i] == x):
                y = Y[i]
                return y
            
            delta = abs( X[i] - x ) 
            if(delta < min_d):
                min_d = delta 
                j = i
        
        k = int()
        if(X[j] < x):
            k = j
        else:
            k = j-1
        
        a = (Y[k+1] - Y[k]) / (X[k+1] - X[k]) 
        b = Y[k] - a*X[k]
        y = a*x + b

        return y

    @staticmethod
    def normalizeVector(X, m, M):
        """Normalize an input vector (X) between a minimum (m) and
        maximum (m) value given per element.

        Parameters
        ----------
        X : np.array
            The array/vector to be normalized
        m : np.array
            The array/vector that gives the minimum values
        M : np.array
            The array/vector that gives the maximum values

        Returns
        -------
        Y : np.array
            Normalized vector output 
        """

        Y = np.empty(0)
        for i in range(X.size):
            z = ( X[i] - m[i] ) / ( M[i] - m[i] )
            Y = np.append(Y, z)

        return Y

    @staticmethod
    def deNormalizeVector(X, m, M):
        """Retrieves a normalized vector back with respect to a minimum (m) and
        maximum (m) value given per element.

        Parameters
        ----------
        X : np.array
            The array/vector to be denormalized
        m : np.array
            The array/vector that gives the minimum values
        M : np.array
            The array/vector that gives the maximum values

        Returns
        -------
        Y : np.array
            deNormalized vector output 
        """

        Y = np.empty(0)
        for i in range(X.size):
            z = X[i]*( M[i] - m[i] ) + m[i]
            Y = np.append(Y, z)

        return Y

    @staticmethod
    def standardizeVector(X, m, s):
        """Standardize an input vector (X) with a minimum (m) and
        standard (s) value given per element.

        Parameters
        ----------
        X : np.array
            The array/vector to be standardized
        m : np.array
            The array/vector that gives the minimum values
        s : np.array
            The array/vector that gives the standard values

        Returns
        -------
        Y : np.array
            Standardized vector output 
        """

        Y = np.empty(0)
        for i in range(X.size):
            z = ( X[i] - m[i] ) / ( s[i] )
            Y = np.append(Y, z)

        return Y

    @staticmethod
    def deStandardizeVector(X, m, s):
        """Retrieve back a vector that was standardized with respect to
        a minimum (m) and standard (s) value given per element.

        Parameters
        ----------
        X : np.array
            The array/vector to be destandardized
        m : np.array
            The array/vector that gives the minimum values
        s : np.array
            The array/vector that gives the standard values

        Returns
        -------
        Y : np.array
            deStandardized vector output 
        """
        Y = np.empty(0)
        for i in range(X.size):
            z = X[i]*s[i] + m[i]
            Y = np.append(Y, z)

        return Y
