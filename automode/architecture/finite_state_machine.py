import random
import graphviz as gv
import logging
import re

from automode.modules.chocolate import Behavior, Condition
from automode.architecture.abstract_architecture import AutoMoDeArchitectureABC

# TODO: Write documentation for methods and classes


class State:
    count = 0

    def __init__(self, behavior):
        self.behavior = behavior
        self.id = State.count
        self.ext_id = self.id  # the ext_id is used to identify this state for any external program
        State.count += 1

    def convert_to_AutoMoDe_commandline_args(self):
        """Converts this state to a format that is readable by the AutoMoDe command line"""
        args = ["--s" + str(self.ext_id), str(self.behavior.int)]
        for param in self.behavior.params:
            # TODO: Find better handling
            if param == "att" or param == "rep":
                pval = "%.2f" % self.behavior.params[param]
            elif param == "rwm":
                pval = str(self.behavior.params[param])
            else:
                logging.error("Undefined parameter")
                pval = 0
            args.extend(["--" + param + str(self.ext_id), pval])
        return args

    @property
    def name(self):
        """Returns an identifier for this state, made up by its behavior and id"""
        return self.behavior.name + "_" + str(self.id)

    def caption(self):
        """Returns a caption for the state that can be used to represent the state in graphviz"""
        caption = self.behavior.name + "_" + str(self.id)
        caption += self.behavior.get_parameter_for_caption()
        return caption


class Transition:
    count = 0

    def __init__(self, from_state, to_state, condition):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition
        self.id = Transition.count
        self.ext_id = self.id  # the ext_id is used to identify this transition for any external program
        Transition.count += 1

    def convert_to_AutoMoDe_commandline_args(self):
        """Converts this transition to a format that is readable by the AutoMoDe command line"""
        t_id = str(self.from_state.ext_id) + "x" + str(self.ext_id)
        if self.from_state.ext_id < self.to_state.ext_id:
            # This has to do with the issue of GetPossibleDestinationBehaviors in AutoMoDe
            args = ["--n" + t_id, str(self.to_state.ext_id - 1)]
        else:
            args = ["--n" + t_id, str(self.to_state.ext_id)]
        args.extend(["--c" + t_id, str(self.condition.int)])
        for param in self.condition.params:
            c = self.condition.name
            if c == "BlackFloor" or c == "GrayFloor" or c == "WhiteFloor" or c == "FixedProbability":
                if param == "p":
                    pval = "%.2f" % self.condition.params[param]
            if c == "NeighborsCount" or c == "InvertedNeighborsCount":
                if param == "w":
                    pval = "%.2f" % self.condition.params[param]
                if param == "p":
                    pval = str(self.condition.params[param])
            args.extend(["--" + param + str(t_id), pval])
        return args

    @property
    def name(self):
        """Returns an identifier for this state, made up by its behavior and id"""
        return self.condition.name + "_" + str(self.id)

    def caption(self):
        """Returns a caption for the state that can be used to represent the state in graphviz"""
        caption = self.condition.name + "_" + str(self.id)
        caption += self.condition.get_parameter_for_caption()
        return caption


class FSM(AutoMoDeArchitectureABC):
    """A finite state machine"""

    # FSM implementation

    def __init__(self, minimal=False):
        self.initial_state = None
        self.states = []
        self.transitions = []
        super().__init__(minimal=minimal)

        # used to find articulation points, find better place then here
        self.aputils_time = 0

    def create_minimal_controller(self):
        """
        Sets up a minimal controller. That is a FSM with a single state and no transitions.
        """
        # The empty FSM
        minimal_behavior = Behavior.get_by_id(0)
        self.initial_state = State(minimal_behavior)
        self.states = [self.initial_state]
        self.transitions = []

    def draw(self, graph_name):
        """Draw the graph representation of the FSM with graphviz"""
        graph = gv.Digraph(format='svg')
        for s in self.states:
            if s == self.initial_state:
                graph.node(s.name, shape="doublecircle", label=s.caption())
            else:
                graph.node(s.name, shape="circle", label=s.caption())
        for t in self.transitions:
            graph.node(t.name, shape="diamond", label=t.caption())
            graph.edge(t.from_state.name, t.name)
            graph.edge(t.name, t.to_state.name)
        filename = graph.render(filename='img/graph_' + graph_name, view=False)

    @staticmethod
    def parse_from_commandline_args(cmd_args):
        """This is the invert of convert_to_commandline_args"""

        def parse_number_of_states():
            """Used so that transitions can immediately point to the states they need"""
            number_of_states = int(to_parse.pop(0))  # this the number of states
            # Create an according number of states
            for i in range(0, number_of_states):
                s = State(stop_behavior)
                s.ext_id = i
                finite_state_machine.states.append(s)
            return number_of_states

        def parse_state():
            state_number = int(token.split("--s")[1])  # take only the number
            state_behavior_id = int(to_parse.pop(0))
            # get the state
            state = [s for s in finite_state_machine.states if s.ext_id == state_number][0]
            # set the correct behavior
            state.behavior = Behavior.get_by_id(state_behavior_id)
            if number_of_states > 1:  # HOTFIX: if there is only one state there is no number of transitions
                # TODO: Find better solution than this hotfix
                # pop until we read --nstatenumber
                tmp = to_parse.pop(0)
                number_of_transitions_delimiter = "--n" + str(state_number)
                # TODO: Improve parsing of parameters and try to add some error handling
                while tmp != number_of_transitions_delimiter:
                    # parse current attribute
                    regex_no_number = re.compile("[^0-9]+")
                    param_name = regex_no_number.match(tmp.split("--")[1]).group()
                    if param_name == "rwm":
                        param_val = int(to_parse.pop(0))
                    else:
                        param_val = float(to_parse.pop(0))
                    state.behavior.params[param_name] = param_val
                    tmp = to_parse.pop(0)
                number_of_transitions = int(to_parse.pop(0))

        def parse_transition():
            transition_id = [int(x) for x in token.split("--n")[1].split("x")]
            from_state = [s for s in finite_state_machine.states if s.ext_id == transition_id[0]][0]
            # to_state = [s for s in finite_state_machine.states if s.ext_id == transition_ids[1]][0]
            transition_ext_id = transition_id[1]
            to_state_id = int(to_parse.pop(0))
            if to_state_id < from_state.ext_id:
                to_state = [s for s in finite_state_machine.states if s.ext_id == to_state_id][0]
            else:  # self-reference not allowed
                to_state = [s for s in finite_state_machine.states if s.ext_id == to_state_id + 1][0]
            condition_count = to_parse.pop(0)
            transition_condition_id = int(to_parse.pop(0))
            # Create transition
            t = Transition(from_state, to_state, Condition.get_by_id(transition_condition_id))
            t.ext_id = transition_ext_id
            finite_state_machine.transitions.append(t)
            re_string = "--[a-z]{}x{}".format(from_state.ext_id, t.ext_id)
            param_regex = re.compile(re_string)
            # TODO: Improve parsing of parameters and try to add some error handling
            while to_parse and param_regex.match(to_parse[0]):
                param_name = to_parse.pop(0)
                regex_no_number = re.compile("[^0-9]+")
                param_name = regex_no_number.match(param_name.split("--")[1]).group()
                if isinstance(t.condition.params[param_name], int):
                    param_val = int(to_parse.pop(0))
                else:
                    param_val = float(to_parse.pop(0))
                # logging.debug("{}: {}".format(param_name, param_val))
                t.condition.params[param_name] = param_val

        # Setting up a completely empty FSM
        finite_state_machine = FSM()
        finite_state_machine.states.clear()
        # prepare the arguments
        to_parse = list(cmd_args)
        stop_behavior = Behavior.get_by_name("stop")
        while to_parse:
            token = to_parse.pop(0)
            if token == "--nstates":
                number_of_states = parse_number_of_states()
            elif "--s" in token:
                # token contains the string for a state
                parse_state()
            elif "--n" in token and "x" in token:  # TODO: Use better check (regex?)
                # token contains the string for a transition
                parse_transition()
        finite_state_machine.initial_state = [s for s in finite_state_machine.states if s.ext_id == 0][0]
        return finite_state_machine

    def convert_to_commandline_args(self):
        """Converts this FSM to a format that is readable by the AutoMoDe command line"""
        self.initial_state.ext_id = 0
        counter = 1
        for state in [s for s in self.states if s != self.initial_state]:
            state.ext_id = counter
            counter += 1
        args = ["--fsm-config", "--nstates", str(len(self.states))]
        # first send the initial state as this has to be state 0
        args.extend(self.initial_state.convert_to_AutoMoDe_commandline_args())
        # Handle the transitions from the initial state
        outgoing_transitions = [t for t in self.transitions if t.from_state == self.initial_state]
        if len(outgoing_transitions) > 0:
            counter = 0
            args.extend(["--n" + str(self.initial_state.ext_id), str(len(outgoing_transitions))])
            for transition in outgoing_transitions:
                transition.ext_id = counter
                counter += 1
                args.extend(transition.convert_to_AutoMoDe_commandline_args())
        # convert the other states
        for state in [s for s in self.states if s != self.initial_state]:
            args.extend(state.convert_to_AutoMoDe_commandline_args())
            # handle the outgoing transitions for this state
            outgoing_transitions = [t for t in self.transitions if t.from_state == state]
            if len(outgoing_transitions) > 0:
                counter = 0
                args.extend(["--n" + str(state.ext_id), str(len(outgoing_transitions))])
                for transition in outgoing_transitions:
                    transition.ext_id = counter
                    counter += 1
                    args.extend(transition.convert_to_AutoMoDe_commandline_args())
        return args

