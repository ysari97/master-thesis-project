# ===========================================================================
# Name        : smash.py
# Author      : YasinS, adapted from JazminZ & ?
# Version     : 0.05
# Copyright   : Your copyright notice
# ===========================================================================

# REMAINING TASKS: i) Documentation ii) ANN code translation

import numpy as np
from alternative_policy_structures import irrigation_policy

class Policy:
    """
    Placeholder
    """

    def __init__(self):

        self.functions = dict()
        self.approximator_names = list()
        self.all_parameters = np.empty(0)

    def add_policy_function(self, name, type, n_inputs, n_outputs, **kwargs):
        """
        Placeholder
        """

        if type == "ncRBF":
            self.functions[name] = ncRBF(n_inputs, n_outputs, kwargs['n_structures'])
        
        elif type == "user_specified":
            class_name = kwargs['class_name']
            klass = globals()[class_name]
            self.functions[name] = klass(n_inputs, n_outputs, kwargs)
        
        elif self.type == "ANN":
            pass

        self.approximator_names.append(name)
        # self.all_parameters = np.append
    
    def assign_free_parameters(self, full_array):
        beginning = 0
        for name in self.approximator_names:
            end = beginning + self.functions[name].getFreeParameterNumber()
            self.functions[name].setParameters(full_array[beginning:end])
            beginning = end

class abstract_approximator: # formerly abstract_approximator

    def __init__(self):
        # function input/output normalization
        self.input_max, self.output_max, \
        self.input_min, self.output_min = tuple(4*[np.empty(0)])
       
        # function input/output standardization
        self.input_mean, self.output_mean, \
        self.input_std, self.output_std = tuple(4*[np.empty(0)])

    def get_output(input):
        pass

    def getInputNumber(self):
        return self.M
    
    def getOutputNumber(self):
        return self.K

    def getFreeParameterNumber(self):
        pass
    
    def get_StdOutput(self, pInput):

        x = self.standardizeVector( pInput, self.input_mean, self.input_std )
        z = self.get_output( x ) 
        y = self.deStandardizeVector( z, self.output_mean, self.output_std )
        return y 
    
    def get_NormOutput(self, pInput):

        x = self.normalizeVector( pInput, self.input_min, self.input_max )
        z = self.get_output( x ) 
        y = self.deNormalizeVector( z, self.output_min, self.output_max )

        return y 
    
    def setMaxInput(self, pV):
        self.input_max = pV

    def setMaxOutput(self, pV):
        self.output_max = pV

    def setMinInput(self, pV):
        self.input_min = pV

    def setMinOutput(self, pV):
        self.output_min = pV

    def setMeanInput(self, pV):
        self.input_mean = pV
        
    def setMeanOutput(self, pV):
        self.output_mean = pV    

    def setStdInput(self, pV):
        self.input_std = pV
        
    def setStdOutput(self, pV):
        self.output_std = pV

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

class RBFparams:
    def __init__(self):
        self.c = np.empty(0)
        self.b = np.empty(0)
        self.w = np.empty(0)

class ncRBF(abstract_approximator):

    def __init__(self, pM, pK, pN):
        # function input/output normalization
        abstract_approximator.__init__(self)
        self.M = pM
        self.K = pK
        self.N = pN
        self.lin_param = np.empty(0)
        self.param = list()

    def setParameters(self, pTheta):
        
        cParam = RBFparams()
 
        count = 0

        for k in range(self.K):
            self.lin_param = np.append(self.lin_param, pTheta[count])
            
            count += 1

        for i in range(self.N):
            for j in range(self.M):
                cParam.c = np.append(cParam.c, pTheta[count])
                cParam.b = np.append(cParam.b, pTheta[count+1])

                count = count + 2

            for k in range(self.K):
                cParam.w = np.append(cParam.w, pTheta[count])

                count += 1
            self.param.append(cParam)
            cParam = RBFparams()

    def clearParameters(self):

        for i in range(len(self.param)):
            self.param[i] = RBFparams()

        self.param = list()
        self.lin_param = np.empty(0)

    def get_output(self, input):

        # RBF
        phi = np.empty(0)
        bf, num, den = tuple(3 * [float()])
        for j in range(self.N):
            bf = 0
            for i in range(self.M):
                
                num = (input[i] - self.param[j].c[i])*(input[i] - self.param[j].c[i])
                den = (self.param[j].b[i]*self.param[j].b[i])
                
                if(den < pow(10,-6)):
                    den = pow(10,-6)
                
                bf = bf + num / den
            
            phi = np.append(phi,  np.exp(-bf) )
        
        # output
        y = np.empty(0)
        o = float()

        for k in range(self.K):
            o = self.lin_param[k]
            for i in range(self.N):
                
                o = o + self.param[i].w[k]*phi[i]
            
            if (o>1):
                o=1.0
            if(o<0):
                o=0.0
        
            y = np.append(y, o)
        
        return y

    def getFreeParameterNumber(self):
        return self.K + self.N * (self.M * 2 + self.K)
        
    def setMaxInput(self, pV):
        super().setMaxInput(pV)
