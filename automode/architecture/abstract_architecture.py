"""
This module provides abstract classes that should be implemented for any architecture
"""

from abc import ABCMeta, abstractmethod
import random
import logging


class ArchitectureABC:
    """
    This abstract class is the base for all architectures. It provides the following methods:
    - create_minimal_controller
    - draw
    - perturb
    This class is intended to provide an architectural abstraction regardless of the actual design method (AutoMoDe or
    something else)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_minimal_controller(self):
        pass

    @abstractmethod
    def draw(self, graph_name):
        pass

    def _get_perturbation_operators(self):
        """Returns all methods that start with perturb_ indicating that they are indeed perturbation operators."""
        method_names = [method_name for method_name in dir(self)
                        if callable(getattr(self, method_name)) and method_name.startswith("perturb_")]
        methods = [getattr(self, x) for x in method_names]
        return methods

    def perturb(self):
        """
        Apply a random perturbation operator to this controller.

        In order for this to work, all possible perturbation operators need to start with perturb_
        """
        perturbation_operators = self._get_perturbation_operators()
        while perturbation_operators:
            perturbation_operator = random.choice(perturbation_operators)
            # execute perturbation
            result = perturbation_operator()
            # remove operator from list so it is not chosen again if it failed
            perturbation_operators.remove(perturbation_operator)
            if result:
                self.perturb_history.append(perturbation_operator.__name__)
                return
        # We cannot apply any operator -> how can this even happen?
        logging.error("A critical error appeared. We cannot apply any perturbation at his point.")


class AutoMoDeArchitectureABC(ArchitectureABC):
    """
    This abstract class is inheriting from ArchitectureABC and is providing a number of additional methods:
    - parse_from_commandline_args: Takes cmd args like AutoMoDe and returns the representation as it is implemented here
    - convert_to_commandline_args: Turns this object into a format that is readable by AutoMoDe
    - evaluate: Evaluates the current controller in ARGoS through AutoMoDe
    """

    def __init__(self, minimal=False):
        self.scores = []  # this will be a list scores
        self.agg_score = ("type", float("inf"))  # the aggregated score with respect to the fitness function, useful for short debugging
        # parameters used to keep track of the local search
        self.perturb_history = []  # a list of all applied operators -> TODO: transform this into a list of strings
        self.evaluated_instances = {}  # a dictionary with keys of seeds and entries are the scores

        if minimal:
            self.create_minimal_controller()

        self.id = -1

    @staticmethod
    @abstractmethod
    def parse_from_commandline_args(cmd_args):
        pass

    @abstractmethod
    def convert_to_commandline_args(self):
        pass
