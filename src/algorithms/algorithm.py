from abc import ABCMeta, abstractmethod

class Algorithm:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, context, sqlCtx, options, saved_model_fname=None): pass


    @abstractmethod
    def train(self, data_path): pass
    """ An abstract method that is expected to train the algorithm and return training info about the model"""

    @abstractmethod
    def test(self, data_path): pass
    """ An abstract method that is expected to test the algorithm and return training info about the model"""

    @abstractmethod
    def save(self, output_fname): pass

    @abstractmethod
    def load(self, input_fname): pass

    @abstractmethod
    def predict_probability_area(model, upper_bound, lower_bound): pass

    @abstractmethod
    def predict_probability_radius(model, radius): pass


