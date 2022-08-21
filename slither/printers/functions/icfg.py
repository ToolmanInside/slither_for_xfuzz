from collections import defaultdict
from slither.printers.abstract_printer import AbstractPrinter
from slither.core.declarations.solidity_variables import SolidityFunction
from slither.core.declarations.function import Function
from slither.core.variables.variable import Variable

def _contract_subgraph(contract):
    return f'cluster_{contract.id}_{contract.name}'

# return unique id for contract function to use as node name
def _function_node(contract, function):
    return f'{contract.id}_{function.name}'

# return unique id for solidity function to use as node name
def _solidity_function_node(solidity_function):
    return f'{solidity_function.name}'

# return dot language string to add graph edge
def _edge(from_node, to_node):
    return f'"{from_node}" -> "{to_node}"'

# return dot language string to add graph node (with optional label)
def _node(node, label=None):
    return ' '.join((
        f'"{node}"',
        f'[label="{label}"]' if label is not None else '',
    ))

class CallLinkList(object):
    CALL_TYPE = ["INTERNAL", "EXTERNAL", "SOLIDITY", "NONE"]
    class LinkNode(object):
        def __init__(self, call_type):
            self.call_type = call_type
            self.prev_node = None
            self.next_node = None

    class LinkList(object):
        def __init__(self):
            self.head_node = LinkNode("NONE")
        
        def add_node(self, node):
            self.head_node.next_node = node


class CallPrinter(AbstractPrinter):
    ARGUMENT = 'calls'
    HELP = 'Export the calls of the contracts to a dot file'

    WIKI = 'https://github.com/trailofbits/slither/wiki/Printer-documentation#call-graph'

    def _process_functions(self, functions):

        contract_functions = defaultdict(set) # contract -> contract functions nodes
        contract_calls = defaultdict(set) # contract -> contract calls edges

        solidity_functions = set() # solidity function nodes
        solidity_calls = set() # solidity calls edges
        external_calls = set() # external calls edges

        all_contracts = set()

        for function in functions:
            all_contracts.add(function.contract)
        for function in functions:
            self._process_function(function.contract,
                                   function,
                                   contract_functions,
                                   contract_calls,
                                   solidity_functions,
                                   solidity_calls,
                                   external_calls,
                                   all_contracts)
        return contract_functions, contract_calls, solidity_functions, solidity_calls, external_calls, all_contracts

    def _process_function(self, contract, function, contract_functions, contract_calls, solidity_functions, solidity_calls, external_calls, all_contracts):
        contract_functions[contract].add(
            _node(_function_node(contract, function), function.name),
        )

        for internal_call in function.internal_calls:
            self._process_internal_call(contract, function, internal_call, contract_calls, solidity_functions, solidity_calls)
        for external_call in function.high_level_calls:
            self._process_external_call(contract, function, external_call, contract_functions, external_calls, all_contracts)

    def _process_internal_call(self, contract, function, internal_call, contract_calls, solidity_functions, solidity_calls):
        if isinstance(internal_call, (Function)):
            contract_calls[contract].add(_edge(
                _function_node(contract, function),
                _function_node(contract, internal_call),
            ))
        elif isinstance(internal_call, (SolidityFunction)):
            solidity_functions.add(
                _node(_solidity_function_node(internal_call)),
            )
            solidity_calls.add(_edge(
                _function_node(contract, function),
                _solidity_function_node(internal_call),
            ))

    def _process_external_call(self, contract, function, external_call, contract_functions, external_calls, all_contracts):
        external_contract, external_function = external_call

        if not external_contract in all_contracts:
            return

        # add variable as node to respective contract
        if isinstance(external_function, (Variable)):
            contract_functions[external_contract].add(_node(
                _function_node(external_contract, external_function),
                external_function.name
            ))

        external_calls.add(_edge(
            _function_node(contract, function),
            _function_node(external_contract, external_function),
        ))

    def output(self, _filename):
        info = ''
        contract_functions, contract_calls, solidity_functions, solidity_calls, external_calls, all_contracts = \
            self._process_functions(self.slither.functions)
        print(contract_functions)
        print(contract_calls)
        print(solidity_functions)
        print(solidity_calls)
        print(external_calls)
        print(all_contracts)

        self.info(info)
        res = self.generate_output(info)
        return res