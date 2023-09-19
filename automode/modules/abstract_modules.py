"""
This module provides two abstract classes for modules: ABCBehavior for behaviors and ABCCondition for conditions.
"""

from abc import ABCMeta, abstractmethod


class ABCBehavior:
    """
    This is an abstract class for the behaviors of AutoMoDe
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def get_by_name(name):
        """
        Returns a Behavior object that corresponds to the given name
        :param name: The name of a behavior (as defined by the AutoMoDe version)
        :return: A Behavior object that corresponds to the behavior with the given name
        """
        pass

    @staticmethod
    @abstractmethod
    def get_by_id(b_id):
        """
        Returns a Behavior object that corresponds to the given id
        :param b_id: The id of the behavior (as defined by the parser for the AutoMoDe version)
        :return: A Behavior object that corresponds to the behavior with the given name
        """
        pass

    @staticmethod
    @abstractmethod
    def get_parameters_for_behavior(name):
        """
        Returns a list of parameters for the behavior with the given name
        :param name: The name of a behavior (as defined by the AutoMoDe version)
        :return: A list of parameters for the behavior with the given name
        """
        pass

    @staticmethod
    @abstractmethod
    def random_parameter(name):
        """
        Initializes a parameter (specified by the name) with a random value that is within it's possible parameter range
        :param name:
        :return:
        """
        pass

    @property
    @abstractmethod
    def int(self):
        """
        Returns an integer representation of this behavior. Note that this is basically the reverse of get_by_id()
        :return:
        """
        pass


class ABCCondition:
    """
    This is an abstract class for the conditions of AutoMoDe
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def get_by_name(name):
        """
        Returns a Condition object that corresponds to the given name
        :param name: The name of a condition (as defined by the AutoMoDe version)
        :return: A Condition object that corresponds to the condition with the given name
        """
        pass

    @staticmethod
    @abstractmethod
    def get_by_id(c_id):
        """
        Returns a Condition object that corresponds to the given id
        :param c_id: The id of the condition (as defined by the parser for the AutoMoDe version)
        :return: A Condition object that corresponds to the condition with the given name
        """
        pass

    @staticmethod
    @abstractmethod
    def get_parameters_for_condition(name):
        """
        Returns a list of parameters for the condition with the given name
        :param name: The name of a condition (as defined by the AutoMoDe version)
        :return: A list of parameters for the condition with the given name
        """
        pass

    @staticmethod
    @abstractmethod
    def random_parameter(name):
        """
        Initializes a parameter (specified by the name) with a random value that is within it's possible parameter range
        :param name:
        :return:
        """
        pass

    @property
    @abstractmethod
    def int(self):
        """
        Returns an integer representation of this condition. Note that this is basically the reverse of get_by_id()
        :return:
        """
        pass
