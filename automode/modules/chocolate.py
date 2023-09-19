"""
This module provides implementations for the modules of AutoMoDe Chocolate
"""

import random
import logging

from automode.modules.abstract_modules import ABCBehavior, ABCCondition


class Behavior(ABCBehavior):
    """
    This class represents the behaviors of AutoMoDe Chocolate
    """

    def __init__(self, name):
        self.name = name
        self.params = Behavior.get_parameters_for_behavior(name)

    @staticmethod
    def get_by_name(name):
        """Returns a new instance of the specified behavior"""
        return Behavior(name)

    @staticmethod
    def get_by_id(b_id):
        b_name = "Failure"
        if b_id == 0:
            b_name = "Exploration"
        elif b_id == 1:
            b_name = "Stop"
        elif b_id == 2:
            b_name = "Phototaxis"
        elif b_id == 3:
            b_name = "AntiPhototaxis"
        elif b_id == 4:
            b_name = "Attraction"
        elif b_id == 5:
            b_name = "Repulsion"
        else:
            logging.error("Unknown id {} for a behavior.".format(b_id))
        return Behavior(b_name)

    @staticmethod
    def get_parameters_for_behavior(name):
        """Returns a list of names of the parameters that can be used to alter the behavior"""
        if name == "AntiPhototaxis":
            return {}
        elif name == "Attraction":
            return {"att": Behavior.random_parameter("Attraction.att")}  # real value in [1,5]
        elif name == "Exploration":
            return {"rwm": Behavior.random_parameter("Exploration.rwm")}  # Boundaries {1,2,...,100} all integers
        elif name == "Phototaxis":
            return {}
        elif name == "Stop":
            return {}
        elif name == "Repulsion":
            return {"rep": Behavior.random_parameter("Repulsion.rep")}  # real value in [1,5]
        else:
            return {}

    @staticmethod
    def random_parameter(name):
        """Returns a random uniform value for the given parameter.
        To allow identification when different parameter spaces are used for the same parameter name it must be fully
        quantified (that is [behavior].[parameter])"""
        splits = name.split(".")
        # b_name = splits[0]
        parameter_name = splits[1]
        if parameter_name == "att":
            return random.uniform(1, 5)
        if parameter_name == "rwm":
            return random.randint(0, 100)
        if parameter_name == "rep":
            return random.uniform(1, 5)
        logging.error("Invalid combination of behavior and parameter {}".format(name))
        return 0

    @property
    def int(self):
        """Returns an integer value according to the internal representation of AutoMoDe"""
        if self.name == "Exploration":
            return 0
        elif self.name == "Stop":
            return 1
        elif self.name == "Phototaxis":
            return 2
        elif self.name == "AntiPhototaxis":
            return 3
        elif self.name == "Attraction":
            return 4
        elif self.name == "Repulsion":
            return 5
        else:
            logging.error("Unknown name {} for a behavior.".format(self.name))
            return -1

    # TODO: Describe this in the class
    # This list contains all possible behaviors that exist in AutoMoDe Chocolate
    behavior_list = ["AntiPhototaxis", "Attraction", "Exploration", "Phototaxis", "Stop", "Repulsion"]

    def get_parameter_for_caption(self):
        """
        :return: a string representing the parameters and their values
        """
        param_list = ""
        if self.params:
            param_list = param_list + "("
            # "\n(" TODO: This linebreak seems to break on the cluster. Add it again if the issue is resolved
            first = True
            for key, value in self.params.items():
                if not first:
                    param_list = param_list + ", "
                first = False
                param_list = param_list + key + ": " + str(value)
                param_list = param_list + ")"
        return param_list


class Condition(ABCCondition):
    """
    This class represents the conditions of AutoMoDe Chocolate
    """

    def __init__(self, name, **kwargs):
        self.name = name
        if not kwargs:
            self.params = Condition.get_parameters_for_condition(name)
        else:
            # TODO: Include some checks here
            # print(str(kwargs))
            self.params = kwargs

    @staticmethod
    def get_by_name(name):
        """Returns a new instance of the specified condition"""
        return Condition(name)

    @staticmethod
    def get_by_id(c_id):
        t_name = "Failure"
        if c_id == 0:
            t_name = "BlackFloor"
        elif c_id == 1:
            t_name = "GrayFloor"
        elif c_id == 2:
            t_name = "WhiteFloor"
        elif c_id == 3:
            t_name = "NeighborsCount"
        elif c_id == 4:
            t_name = "InvertedNeighborsCount"
        elif c_id == 5:
            t_name = "FixedProbability"
        else:
            logging.error("Unknown id {} for a condition.".format(c_id))
        return Condition(t_name)

    @staticmethod
    def get_parameters_for_condition(name):
        """Returns a list of names of the parameters that can be used to alter the condition"""
        if name == "BlackFloor":
            return {"p": Condition.random_parameter("BlackFloor.p")}  # probably between 0 and 1
        elif name == "FixedProbability":
            return {"p": Condition.random_parameter("FixedProbability.p")}  # probably between 0 and 1
        elif name == "GrayFloor":
            return {"p": Condition.random_parameter("GrayFloor.p")}  # probably between 0 and 1
        elif name == "InvertedNeighborsCount":
            return {"w": Condition.random_parameter("InvertedNeighborsCount.w"),  # real value in [0,20]
                    "p": Condition.random_parameter("InvertedNeighborsCount.p")}  # integer in {0, 2, ..., 10}
        elif name == "NeighborsCount":
            return {"w": Condition.random_parameter("NeighborsCount.w"),  # real value in [0,20]
                    "p": Condition.random_parameter("NeighborsCount.p")}  # integer in {0, 2, ..., 10}
        elif name == "WhiteFloor":
            return {"p": Condition.random_parameter("WhiteFloor.p")}  # probably between 0 and 1
        return {}

    @staticmethod
    def random_parameter(name):
        """Returns a random uniform value for the given parameter.
        To allow identification when different parameter spaces are used for the same parameter name it must be fully
        quantified (that is [condition].[parameter])"""
        splits = name.split(".")
        condition_name = splits[0]
        parameter_name = splits[1]
        if condition_name in ("BlackFloor", "GrayFloor", "WhiteFloor", "FixedProbability"):
            if parameter_name == "p":
                return random.uniform(0, 1)
            logging.error("Invalid parameter {} for condition {}".format(parameter_name, condition_name))
        if condition_name in ("NeighborsCount", "InvertedNeighborsCount"):
            if parameter_name == "w":
                return random.uniform(0, 20)
            if parameter_name == "p":
                return random.randint(0, 10)
            logging.error("Invalid parameter {} for condition {}".format(parameter_name, condition_name))
        logging.error("Invalid combination of condition and parameter {}".format(name))
        return 0

    @property
    def int(self):
        """Returns an integer value according to the internal representation of AutoMoDe"""
        if self.name == "BlackFloor":
            return 0
        elif self.name == "GrayFloor":
            return 1
        elif self.name == "WhiteFloor":
            return 2
        elif self.name == "NeighborsCount":
            return 3
        elif self.name == "InvertedNeighborsCount":
            return 4
        elif self.name == "FixedProbability":
            return 5
        logging.error("Unknown name {} for a condition".format(self.name))
        return -1

    # TODO: Document in class doc string
    # This list contains all possible conditions that exist in AutoMoDe Chocolate
    condition_list = ["BlackFloor", "FixedProbability", "GrayFloor", "InvertedNeighborsCount", "NeighborsCount",
                      "WhiteFloor"]

    def get_parameter_for_caption(self):
        """
        :return: a string representing the parameters and their values
        """
        param_list = ""
        if self.params:
            param_list = param_list + "("
            # + "\n(" TODO: This linebreak seems to break on the cluster. Add it again if the issue is resolved
            first = True
            for key, value in self.params.items():
                if not first:
                    param_list = param_list + ", "
                first = False
                param_list = param_list + key + ": " + str(value)
                param_list = param_list + ")"
        return param_list
