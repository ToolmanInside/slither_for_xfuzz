"""
    Module printing summary of the contract
"""

from slither.printers.abstract_printer import AbstractPrinter
from slither.core.expressions.call_expression import CallExpression
from slither.core.expressions.binary_operation import BinaryOperation
from slither.core.expressions.assignment_operation import AssignmentOperation

class AbstractPresentation(object):
    def __init__(self, node): # node: core/cfg/node.py
        self.node = node
        self.expression = node.expression

        self.irs = node.irs
        self.state_variables_written = node.state_variables_written
        self.node_id = node.node_id
        self.internal_calls = node.internal_calls
        self.high_level_calls = node.high_level_calls
        self.low_level_calls = node.low_level_calls
        self.library_calls = node.library_calls
        self.calls_as_expression = node.calls_as_expression
        self.external_calls_as_expressions = node.external_calls_as_expressions
        self.internal_calls_as_expressions = node.internal_calls_as_expressions
        self.fathers = node.fathers
        self.sons = node.sons
        self.txt = ""

    def is_condition_statement(self):
        return True if len(self.node.fathers) > 1 and isinstance(self.node.expression, BinaryOperation) else False

    def condition_begin_end(self, node, current_id): # 处理是否是while或者for的循环语句
        assert(len(node.fathers) == 2)
        a = node.fathers[0].node_id
        b = node.fathers[1].node_id
        begin = a if a < b else b
        end = b if begin == a else b  # begin是循环起点的node编号，end是循环结束的node编号
        begin = current_id + 1 # 这里是不想让控制语句也被拷贝，只拷贝循环体内部的语句
        
        return begin, end

    def has_expression(self):
        if self.node.expression:
            return True
        return False

    # def has_eth_env(self):
    #     if self.node.expression:
    #         if 'blocktime' in self.node.expression or 'blocknumber' in self.node.expression:
    #             return True
    #     return False

    def has_transfer(self):
        if self.node.expression:
            # if 'transfer' in self.node.expression or 'send' in self.node.expression:
            if self.node.can_send_eth():
                return True
        return False

    def contain_require(self):
        return self.node.contains_require_or_assert()

    def is_binary(self):
        if isinstance(self.expression, BinaryOperation):
            return True
        return False

    def is_emit_func(self):
        if len(self.internal_calls) == 0 and len(self.internal_calls_as_expressions) > 0:
            # 去掉emit函数调用
            return True
        return False

    def is_external_call(self):
        if isinstance(self.expression, CallExpression):
            if len(self.external_calls_as_expressions) > 0:
                return True
        return False

    def is_internal_call(self):
        if isinstance(self.expression, CallExpression):
            if len(self.external_calls_as_expressions) <= 0:
                return True
        return False

    def is_low_level_call(self):
        if isinstance(self.expression, CallExpression):
            if len(self.external_calls_as_expressions) > 0:
                if len(self.low_level_calls) > 0:
                    return True
        return False

    def output(self):
        # txt += '\t\t{}\n'.format(node.internal_calls)
        # txt += '\t\t{}\n'.format(node.high_level_calls)
        # txt += '\t\t{}\n'.format(node.library_calls)
        # txt += '\t\t{}\n'.format(node.low_level_calls)
        # txt += '\t\t{}\n'.format(node.calls_as_expression)
        
        if self.contain_require():
            self.txt = 'RequireOrAssert'
        elif isinstance(self.expression, AssignmentOperation):
            if len(self.state_variables_written) > 0:
                self.txt = "StateAssignment"
            else:
                self.txt = "LocalAssignment"
        elif self.is_emit_func():
            return ""
        elif self.is_binary():
            if len(self.state_variables_written) > 0:
                self.txt = "StateBinary"
            else:
                self.txt = "LocalBinary"
        elif self.is_external_call():
            if self.is_low_level_call():
                self.txt = 'ExternalLowLevelCall'
            else:
                if len(self.high_level_calls) == 0:
                    self.txt = ""
                    return
                external_contract, external_function = self.high_level_calls[0]
                # self.txt = 'ExternalHighLevelCall: {0}->{1}'.format(self.node, external_function)
                self.txt = 'ExternalHighLevelCall->{}'.format((external_contract.id, external_function.name))
        elif self.is_internal_call():
            # self.txt = 'InternalCall: {0}->{1}'.format(self.node.type, self.internal_calls[0].name)
            self.txt = "InternalCall->{}".format(self.internal_calls[0])
        else:
            if self.expression == None:
                return ""
            else:
                self.txt = self.expression.__name__()
        # if len(self.node.external_calls_as_expressions) > 0:
        #     for call in self.node.external_calls_as_expressions:
        #         self.txt += '\n* {}, {}'.format(call.called, call.type_call)
        #     self.txt += '\n'
        # if len(self.node.internal_calls_as_expressions) > 0:
        #     for call in self.node.internal_calls_as_expressions:
        #         self.txt += '\n# {}, {}'.format(call.called, call.type_call)
        #     self.txt += '\n'
        return self.txt
        

class PrinterOperations(AbstractPrinter):
    ARGUMENT = "++operations++"
    HELP = "Print the operation sequence of code"
    WIKI = 'https://github.com/trailofbits/slither/wiki/Printer-documentation#slithir'

    def recur_call(self, entry_function, contract_func_dict, current_contract_id):
        if isinstance(entry_function, tuple):  # 表明这是外部调用
            op_list = contract_func_dict[(entry_function[0], entry_function[1])]
        else:
            op_list = contract_func_dict[(current_contract_id, entry_function)]
        for op in op_list:
            if not op:
                continue
            if "->" in op:
                called_function = eval(op.split('->')[1])
                self.recur_call(called_function, contract_func_dict, current_contract_id)
            print(op, end = ' ')
        return

    def output(self, _filename):
        # txt = ""
        contract_func_dict = dict() # 键值为合约名，函数名，值为sequence
        for contract in self.contracts:
            # txt += 'Contract {}\n'.format(contract.name)
            for function in contract.functions:
                txt = ""
                op_list = list()
                raw_list = list()
                node_stack = list()
                # txt += f'\nFunction {function.canonical_name} '
                # if len(function.modifiers) > 0:
                #     for modifier in function.modifiers:
                #         # txt += '\t\tModifier {}\n'.format(modifier.canonical_name)
                #         for node in modifier.nodes:
                #             abs_node = AbstractPresentation(node)
                #             if abs_node.has_expression:
                #                 contain_require = abs_node.contain_require()
                #                 abs_output = abs_node.output()
                #                 if abs_output == "":
                #                     continue
                #                 txt += '\t\tModifier{}\n'.format(abs_node.output())

                for node in function.nodes: # 先把所有node加到stack里面
                    abs_node = AbstractPresentation(node)
                    node_stack.append(abs_node)

                for abs_node in node_stack: # 对stack里面有可能的循环语句进行处理
                    if abs_node.is_condition_statement():
                        current_id = abs_node.node_id
                        begin, end = abs_node.condition_begin_end(abs_node, current_id)
                        dummy_sequence = node_stack[begin:end+1]  # 先把原数组切片，再和原数组组合起来
                        node_stack = node_stack[:begin] + dummy_sequence + node_stack[begin:]
                        
                for abs_node in node_stack:
                    if len(abs_node.fathers) == 0:
                        continue
                    # txt += '\t{}\n'.format(abs_node.expression)
                    # txt += '\t\t\tIFNode\n'
                    # txt += '\t\t\tFather\n'
                    # for father in abs_node.fathers:
                    #     txt += '\t\t\t{} '.format(father.node_id)
                    # txt += '\n'
                    # txt += '\t\t\tSon\n'
                    # for son in abs_node.sons:
                    #     txt += '\t\t\t{} '.format(son.node_id)
                    # txt += '\n'
                    # txt += '\t\t\tSelf\n'
                    # txt += '\t\t\t{}\n'.format(abs_node.node_id)

                    if abs_node.has_expression:
                        abs_output = abs_node.output()
                        if abs_output == "":
                            continue
                        txt += ' {} '.format(abs_output)
                        op_list.append(abs_output)
                        raw_list.append(abs_node)
                    # elif abs_node.irs:
                    #     txt += '\t\tIRs:\n'
                    #     for ir in abs_node.irs:
                    #         txt += '\t\t\t{}\n'.format(ir)
                # for modifier_statement in function.modifiers_statements:
                #     txt += f'\t\tModifier Call {modifier_statement.entry_point.expression}\n'
                # for modifier_statement in function.explicit_base_constructor_calls_statements:
                #     txt += f'\t\tConstructor Call {modifier_statement.entry_point.expression}\n'
                # print(txt)
                if function.name == 'slitherConstructorVariables':
                    function_name = 'constructor'
                function_name = function.name
                contract_func_dict[(contract.id, function_name)] = op_list
                # contract_func_dict[(contract.id, function_name)] = raw_list
        for contract_function in contract_func_dict.keys():
            print("{},{}: {}".format(contract_function[0], contract_function[1], contract_func_dict[(contract_function[0], contract_function[1])]))
            # print("{},{}: ".format(contract_function[0], contract_function[1]), end = '')
            # self.recur_call(contract_function, contract_func_dict, contract_function[0])
            # print("\n")
        print("\n")
        # print("=" * 15)
        self.info(txt)
        # print(contract_func_dict)
        res = self.generate_output("")
        # print(txt)
        return res

class FunctionAction(AbstractPrinter):
    ARGUMENT = "++actions++"
    HELP = "Print action sequences of code"
    WIKI = 'https://github.com/trailofbits/slither/wiki/Printer-documentation#slithir'

    def guardCheck(self, node_list):
        # action detection module
        for i, node in enumerate(node_list):
            if node.output() == "RequireOrAssert" and i == 0:
                return True
        return False

    def etherTransfer(self, node):
        # action detection module
        if node.has_transfer():
            return True
        return False

    def ethereumEnv(self, node):
        # action detection module
        if node.expression:
            exp = str(node.expression)
            if 'block.timestamp' in exp or 'block.number' in exp:
                return True
        return False

    def stateWrite(self, node):
        # action detection module
        if len(node.state_variables_written) > 0:
            return True
        return False

    def maliciousCall(self, node):
        # action detection module
        if node.output():
            if "LowLevelCall" in node.output():
                return True
        return False

    def output(self, _filename):
        txt = ""
        # txt += '\tContract {}\n'.format(self.contract.name)
        contract_func_dict = self.ops()
        for contract_function in contract_func_dict.keys():
            action_list = list()
            contract_name, function_name = contract_function[0], contract_function[1]
            if function_name == "slitherConstructorVariables":
                continue
            node_list = contract_func_dict[contract_function]
            if self.guardCheck(node_list):
                action_list.append("GuardCheck")
            for node in node_list:
                if self.etherTransfer(node):
                    action_list.append("EtherTransfer")
                    continue
                elif self.maliciousCall(node):
                    action_list.append("MaliciousCall")
                    continue
                elif self.ethereumEnv(node):
                    action_list.append("EthereumEnv")
                    continue
                # elif self.stateWrite(node): 暂时不考虑全局变量，因为真的太常见了
                #     action_list.append("StateWrite")
                #     continue
                else:
                    continue
            actions = ""
            for action in action_list:
                actions += action + ' '
            print("{},{},{}".format(contract_name, function_name, action_list))
        res = self.generate_output("")
        return res

    def ops(self):
        contract_func_dict = dict() # 键值为合约名，函数名，值为sequence
        for contract in self.contracts:
            for function in contract.functions:
                txt = ""
                op_list = list()
                raw_list = list()
                node_stack = list()
                for node in function.nodes: # 先把所有node加到stack里面
                    abs_node = AbstractPresentation(node)
                    node_stack.append(abs_node)
                for abs_node in node_stack: # 对stack里面有可能的循环语句进行处理
                    if abs_node.is_condition_statement():
                        current_id = abs_node.node_id
                        begin, end = abs_node.condition_begin_end(abs_node, current_id)
                        dummy_sequence = node_stack[begin:end+1]  # 先把原数组切片，再和原数组组合起来
                        node_stack = node_stack[:begin] + dummy_sequence + node_stack[begin:]
                for abs_node in node_stack:
                    if len(abs_node.fathers) == 0:
                        continue
                    if abs_node.has_expression:
                        abs_output = abs_node.output()
                        if abs_output == "":
                            continue
                        txt += ' {} '.format(abs_output)
                        op_list.append(abs_output)
                        raw_list.append(abs_node)
                if function.name == 'slitherConstructorVariables':
                    function_name = 'constructor'
                function_name = function.name
                contract_func_dict[(contract.name, function_name)] = raw_list
        return contract_func_dict
