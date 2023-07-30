# ===========================================================================
# Name        : smash.py
# Author      : YasinS, adapted from JazminZ & ?
# Version     : 0.05
# Copyright   : Your copyright notice
# ===========================================================================

# REMAINING TASKS: i) Documentation ii) ANN code translation

import numpy as np


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
            self.functions[name] = ncRBF(n_inputs, n_outputs, kwargs)

        elif type == "user_specified":
            class_name = kwargs["class_name"]
            klass = globals()[class_name]
            self.functions[name] = klass(n_inputs, n_outputs, kwargs)

        elif self.type == "ANN":
            pass

        self.approximator_names.append(name)

    def assign_free_parameters(self, full_array):
        beginning = 0
        for name in self.approximator_names:
            end = beginning + self.functions[name].get_free_parameter_number()
            self.functions[name].set_parameters(full_array[beginning:end])
            beginning = end

    def get_total_parameter_count(self):
        return sum([x.get_free_parameter_number() for x in self.functions.values()])


class abstract_approximator:  # formerly abstract_approximator
    def __init__(self, argument_dictionary):
        # Will be removed at the end!!!
        # function input/output normalization
        self.input_max = argument_dictionary["max_input"]
        self.output_max = argument_dictionary["max_output"]
        self.input_min = argument_dictionary["min_input"]
        self.output_min = argument_dictionary["min_output"]

    def get_output(input):
        pass

    def get_free_parameter_number(self):
        pass

    def get_output_std(self, pInput):

        x = self.standardize_vector(pInput, self.input_mean, self.input_std)
        z = self.get_output(x)
        y = self.destandardize_vector(z, self.output_mean, self.output_std)
        return y

    def get_output_norm(self, pInput):

        x = self.normalize_vector(pInput, self.input_min, self.input_max)
        z = self.get_output(x)
        y = self.denormalize_vector(z, self.output_min, self.output_max)

        return y

    @staticmethod
    def normalize_vector(X, m, M):
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
            z = (X[i] - m[i]) / (M[i] - m[i])
            Y = np.append(Y, z)

        return Y

    @staticmethod
    def denormalize_vector(X, m, M):
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
            z = X[i] * (M[i] - m[i]) + m[i]
            Y = np.append(Y, z)

        return Y

    @staticmethod
    def standardize_vector(X, m, s):
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
            z = (X[i] - m[i]) / (s[i])
            Y = np.append(Y, z)

        return Y

    @staticmethod
    def destandardize_vector(X, m, s):
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
            z = X[i] * s[i] + m[i]
            Y = np.append(Y, z)

        return Y


class RBFparams:
    def __init__(self):
        self.c = np.empty(0)
        self.b = np.empty(0)
        self.w = np.empty(0)


class ncRBF(abstract_approximator):
    def __init__(self, n_inputs, n_outputs, argument_dictionary):
        # function input/output normalization
        abstract_approximator.__init__(self, argument_dictionary)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.RBF_count = argument_dictionary["n_structures"]
        self.lin_param = np.empty(0)
        self.param = list()

    def set_parameters(self, pTheta):

        self.clear_parameters()

        count = 0
        self.lin_param = pTheta[: self.n_outputs]
        count += self.n_outputs

        for i in range(self.RBF_count):
            cParam = RBFparams()
            for j in range(self.n_inputs):
                cParam.c = np.append(cParam.c, pTheta[count])
                cParam.b = np.append(cParam.b, pTheta[count + 1])

                count = count + 2

            for k in range(self.n_outputs):
                cParam.w = np.append(cParam.w, pTheta[count])

                count += 1
            self.param.append(cParam)

    def clear_parameters(self):

        self.param = list()
        self.lin_param = np.empty(0)

    def get_output(self, input):

        # RBF
        phi = np.empty(0)
        for j in range(self.RBF_count):
            bf = 0
            for i in range(self.n_inputs):

                num = (input[i] - self.param[j].c[i]) ** 2
                den = self.param[j].b[i] ** 2

                if den < pow(10, -6):
                    den = pow(10, -6)

                bf = bf + num / den

            phi = np.append(phi, np.exp(-bf))

        # output
        y = np.empty(0)

        for k in range(self.n_outputs):
            o = self.lin_param[k]
            for i in range(self.RBF_count):

                o = o + self.param[i].w[k] * phi[i]

            if o > 1:
                o = 1.0
            if o < 0:
                o = 0.0

            y = np.append(y, o)

        return y

    def get_free_parameter_number(self):
        return self.n_outputs + self.RBF_count * (self.n_inputs * 2 + self.n_outputs)
