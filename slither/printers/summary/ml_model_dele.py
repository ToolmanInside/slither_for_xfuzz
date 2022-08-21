from slither.detectors.abstract_detector import AbstractDetector, DetectorClassification
from slither.printers.abstract_printer import AbstractPrinter
from slither.analyses.evm import generate_source_to_evm_ins_mapping, load_evm_cfg_builder
from func_timeout import func_set_timeout
from slither.utils.colors import blue, green, magenta, red
from collections import defaultdict
from slither.printers.abstract_printer import AbstractPrinter
from slither.core.declarations.solidity_variables import SolidityFunction
from slither.core.declarations.function import Function
from slither.core.variables.variable import Variable
from slither.core.cfg.node import NodeType
from slither.core.declarations.solidity_variables import SolidityVariableComposed
from slither.solc_parsing.variables.local_variable import LocalVariableSolc
from slither.solc_parsing.variables.state_variable import StateVariableSolc
from slither.core.declarations.solidity_variables import SolidityVariable

import joblib
import os
from gensim.models.word2vec import Word2Vec
import numpy as np
import gdown
import tempfile
import time
import math
import json

class CallGraph(AbstractPrinter):
    ARGUMENT = 'call-graph'
    HELP = 'Export the call-graph of the contracts to a dot file'

    WIKI = 'https://github.com/trailofbits/slither/wiki/Printer-documentation#call-graph'

    def __init__(self, slither, logger):
        super(CallGraph, self).__init__(slither, logger)
        self.external_edges = []
        self.internal_edges = []

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

        return self.external_edges, self.internal_edges

    def _process_function(self, contract, function, contract_functions, contract_calls, solidity_functions, solidity_calls, external_calls, all_contracts):

        for internal_call in function.internal_calls:
            self._process_internal_call(contract, function, internal_call, contract_calls, solidity_functions, solidity_calls)
        for external_call in function.high_level_calls:
            self._process_external_call(contract, function, external_call, contract_functions, external_calls, all_contracts)

    def _condition_complexity(self, vairable_list):
        amount_of_functioncalls = 0
        for child_list in vairable_list:
            if 'SolidityFunctionCall' in child_list:
                amount_of_functioncalls += 1
        vacab = set()
        correspondence = 0
        for child_list in vairable_list:
            for c in child_list:
                vacab.add(c)
        vacab = list(vacab)
        amount_of_variables = len(vacab)
        for v in vacab:
            variable_in_count = 0
            total_length = len(vairable_list)
            for child_list in vairable_list:
                if v in child_list:
                    variable_in_count += 1
            correspondence += variable_in_count / total_length
        complexity = math.sqrt(amount_of_variables) * (amount_of_functioncalls + 1) * correspondence
        complexity = round(complexity, 2)
        return complexity

    def _variables_in_conditions(self, node):
        name_list = list()
        if len(node.internal_calls) > 0 or len(node.solidity_calls) > 0 or len(node.high_level_calls) > 0:
            name_list.append('SolidityFunctionCall')
        for var in node.variables_read:
            if type(var) == SolidityVariableComposed:
                name = var.name
            elif type(var) == StateVariableSolc:
                name = var.var_name
            elif type(var) == LocalVariableSolc:
                name = var.var_name
            elif type(var) == SolidityVariable:
                name = var.name
            else:
                continue
            name_list.append(name)
        return name_list

    def _process_internal_call(self, contract, function, internal_call, contract_calls, solidity_functions, solidity_calls):
        entry_distance = 0
        condition_distance = 0
        first_condition = True
        distance_stop = False
        variables_in_path = list()

        if isinstance(internal_call, (Function)):
            # print(function.name + ' ' + internal_call.name + " <===")
            for node in function.nodes:
                if node.expression != None and not distance_stop:
                    entry_distance += 1
                    if node.type == NodeType.EXPRESSION and \
                        ("assert" in str(node.expression) or "require" in str(node.expression)):
                        variables_in_path.append(self._variables_in_conditions(node))
                        if first_condition:
                            condition_distance = entry_distance
                            first_condition = False
                    if node.type == NodeType.IF:
                        variables_in_path.append(self._variables_in_conditions(node))
                        if first_condition:
                            condition_distance = entry_distance
                            first_condition = False
                    if internal_call.name in str(node.expression):
                        entry_distance = entry_distance - 1
                        distance_stop = True
            complexity = self._condition_complexity(variables_in_path)
            if entry_distance == 0 and condition_distance == 0:
                priority_score = 0.0
            else:
                priority_score = entry_distance * condition_distance / (entry_distance + condition_distance)
                priority_score = round(priority_score * complexity, 2)
            self.internal_edges.append((str(contract) + '-' + str(function), str(contract) + '-' + str(internal_call), str(priority_score)))

        elif isinstance(internal_call, (SolidityFunction)):
            pass

    def _process_external_call(self, contract, function, external_call, contract_functions, external_calls, all_contracts):
        external_contract, external_function = external_call
        if not external_contract in all_contracts:
            return
        # add variable as node to respective contract
        entry_distance = 0
        condition_distance = 0
        first_condition = True
        distance_stop = False
        variables_in_path = list()
        priority_score = 0

        # print(external_function.name + ' ' + external_function.name + " <===")
        if isinstance(external_function, StateVariableSolc):
            pass
        else:
            for node in external_function.nodes:
                if node.expression != None and not distance_stop:
                    entry_distance += 1
                    if node.type == NodeType.EXPRESSION and \
                        ("assert" in str(node.expression) or "require" in str(node.expression)):
                        variables_in_path.append(self._variables_in_conditions(node))
                        if first_condition:
                            condition_distance = entry_distance
                            first_condition = False
                    if node.type == NodeType.IF:
                        variables_in_path.append(self._variables_in_conditions(node))
                        if first_condition:
                            condition_distance = entry_distance
                            first_condition = False
                    if external_function.name in str(node.expression):
                        entry_distance = entry_distance - 1
                        distance_stop = True
            complexity = self._condition_complexity(variables_in_path)
            if entry_distance == 0 and condition_distance == 0:
                priority_score = 0.0
            else:
                priority_score = entry_distance * condition_distance / (entry_distance + condition_distance)
                priority_score = round(priority_score * complexity, 2)

        self.external_edges.append((str(contract) + '-' + str(function),
        str(external_contract) + '-' + str(external_function), str(priority_score)))

    def output(self, filename):
        res = self.generate_output("")
        return res


class ModelPredictionDele(AbstractPrinter):
    """
    Predictor Based On Machine Learning Models
    """

    ARGUMENT = 'model-prediction-dele'
    HELP = 'Model Prediction For Delegatecall Vulnerability'
    IMPACT = DetectorClassification.INFORMATIONAL
    CONFIDENCE = DetectorClassification.HIGH

    WIKI = 'https://github.com/trailofbits/slither/wiki/Printer-documentation#evm'
    WIKI_TITLE = 'Model Prediction'
    WIKI_DESCRIPTION = 'This detector is based on trained machine learning models'
    WIKI_RECOMMENDATION = 'Avoid low-level calls. Check the call success. If the call is meant for a contract, check for code existence.'

    def __init__(self, slither, logger):
        super(ModelPredictionDele, self).__init__(slither, logger)

    def output(self, _filename):
        """ Predict the function has suspicious vulnerabilities
        """
        txt = ""
        file_dict = dict()

        W2C_MODEL = WordVectorizer()
        ML_MODEL = MLModelDele()
        logger = None

        callgraph = CallGraph(self.slither, logger)
        external_edges, internal_edges = callgraph._process_functions(self.slither.functions)

        if not self.slither.crytic_compile:
            txt = 'The EVM printer requires to compile with crytic-compile'
            self.info(red(txt))
            res = self.generate_output(txt)
            return res
        evm_info = self._extract_evm_info(self.slither)

        function_path_scores = dict()

        for contract in self.slither.contracts_derived:
            # print(callgraph._process_functions(contract.functions))
            if contract.name not in file_dict.keys():
                file_dict[contract.name] = dict()

            contract_file = self.slither.source_code[contract.source_mapping['filename_absolute']].encode('utf-8')
            contract_file_lines = open(contract.source_mapping['filename_absolute'], 'r').readlines()

            contract_pcs = {}
            contract_cfg = {}

            for function in contract.functions:
                can_send_eth = False
                can_send_eth_list = []
                has_dangerous_call = False

                function_name = function.canonical_name.split('.')[1].split('(')[0]
                if function.canonical_name.split('.')[0] != contract.name:
                    continue
                if  function_name not in file_dict[contract.name].keys():
                    file_dict[contract.name][function_name] = list()
                
                try:
                    if function.is_constructor:
                        contract_cfg = evm_info['cfg_init', contract.name]
                        contract_pcs = evm_info['mapping_init', contract.name]
                    else:
                        contract_cfg = evm_info['cfg', contract.name]
                        contract_pcs = evm_info['mapping', contract.name]
                except:
                    continue

                ins_list = list()
                for node in function.nodes:
                    node_source_line = contract_file[0:node.source_mapping['start']].count("\n".encode("utf-8"))+1
                    node_pcs = contract_pcs.get(node_source_line, [])
                    
                    for pc in node_pcs:
                        ins = contract_cfg.get_instruction_at(pc).name.encode('utf-8').decode('utf-8-sig')
                        if len(ins.split(' ')) == 2:
                            ins = ins.split(' ')[0]
                        file_dict[contract.name][function_name].append(ins)  # <==== ins
                        ins_list.append(ins)
                    
                    if node.irs != []:
                        can_send_eth_list.append(node.can_send_eth())
                    
                    if len(node.external_calls_as_expressions) > 0:
                        for c in node.external_calls_as_expressions:
                            called_function = c.called
                            # arguments = [x.value for x in c.arguments]
                            if str(called_function).split('.')[-1] == 'transfer' and 'msg.sender' in str(c):
                                has_dangerous_call = True
                
                if len(file_dict[contract.name][function_name]) == 0:
                    continue
                # has_private_visibility = False
                # if function.visibility == 'private': # detect function has private
                #     has_private_visibility = True

                have_specific_modifier = False
                for modif in function._modifiers: # detect function has arbitrary modifiers
                    if modif not in ['private', 'payable', 'external', 'internal', 'public']:
                        have_specific_modifier = True

                can_send_eth = True in can_send_eth_list
                if ins_list == []:
                    continue

                if have_specific_modifier:
                    ins_list.append(1)
                else:
                    ins_list.append(0)
                if can_send_eth:
                    ins_list.append(1)
                else:
                    ins_list.append(0)
                if has_dangerous_call:
                    ins_list.append(1)
                else:
                    ins_list.append(0)

                ins_list = self.preprocess(ins_list)

                ins_vec = W2C_MODEL.vectorize(ins_list[:-14])
                other_features = ins_list[-14:].strip().split(' ')
                other_features = np.reshape(other_features, (1, -1))
                combine_vec = np.concatenate((ins_vec, other_features), axis=1)

                predict_result = ML_MODEL.predict(combine_vec)[0]

                function_priority_score = self.calculate_func_prior(function, predict_result)
                # print('function_priority_score: {}'.format(function_priority_score))

                external_caller = []
                internal_caller = []
                combine_name = str(contract.name) + '-' + str(function_name)
                if predict_result == True:
                    print(function_name)
                    for ex in external_edges:
                        if combine_name == ex[1]:
                            external_caller.append(ex[0] + ' ' + ex[2])
                            self.maintain_json(function_path_scores, function_priority_score, predict_result, combine_name, 'external', None, None, ex[0], float(ex[2]))

                    for inx in internal_edges:
                        if combine_name == inx[1]:
                            internal_caller.append(inx[0] + ' ' + inx[2])
                            self.maintain_json(function_path_scores, function_priority_score, predict_result, combine_name, 'internal', inx[0], float(inx[2]), None, None)

                    # print(green("{}\t{}\tPredict:{}".format(contract.name, function_name, predict_result)))
                    # print("\tExternal Caller: {}".format(external_caller))
                    # print("\tInternal Caller: {}".format(internal_caller))

                    if len(external_caller) == 0 and len(internal_caller) == 0 and function_name == 'fallback':
                        self.maintain_json(function_path_scores, function_priority_score, predict_result, combine_name, 'fallback', None, None, None, None)
                    elif len(external_caller) == 0 and len(internal_caller) == 0:
                        self.maintain_json(function_path_scores, function_priority_score, predict_result, combine_name, 'nocaller', None, None, None, None)

                if predict_result == False:
                    # print(red("{}\t{}\tPredict:{}".format(contract.name, function_name, predict_result)))
                    self.maintain_json(function_path_scores, function_priority_score, predict_result, combine_name, None, None, None, None, None)
                # for cc in function_path_scores.keys():
                #     for ff in function_path_scores[cc].keys():
                #         function_path_scores[cc][ff]['callers'] = sorted(function_path_scores[cc][ff]['callers'], key=lambda x: x[2], reverse=True)

            # for cc in function_path_scores.keys():
            #     function_path_scores[cc] = sorted(function_path_scores[cc], key = lambda x: x['func_priority'], reverse = True)
                
        function_path_scores = json.dumps(function_path_scores, indent = 4).encode('utf-8').decode('utf-8-sig')
        print(function_path_scores)
        res = self.generate_output("")
        return res

    def calculate_func_prior(self, function, model_predict):
        parameter_amount = len(function.parameters)
        parameter_weights = 0
        predict_factor = 0.5 if model_predict else 2
        weight_dict = {
            'address': 2,
            'uint': 1.5,
            'bytes': 1,
            'bool': 1,
            '[]': 1.8
        }
        for p in function.parameters:
            for typee, weight in weight_dict.items():
                if typee in str(p.var_type):
                    parameter_weights += weight_dict[typee]
        func_prior_score = predict_factor * (0.5 * parameter_amount + 1) * (0.2 * parameter_weights + 1)
        return round(func_prior_score, 2)

    def maintain_json(self, maintain_dict, function_priority_score, model_predict, current_contract_function, internal_or_external, internal_call, internal_call_score, external_call, external_call_score):
        current_contract = current_contract_function.split('-')[0]
        current_function = current_contract_function.split('-')[1]
        if current_contract not in maintain_dict.keys():
            maintain_dict[current_contract] = dict()
        if current_function not in maintain_dict[current_contract].keys():
            maintain_dict[current_contract][current_function] = dict()

        temp_dict = maintain_dict[current_contract][current_function]
        temp_dict['model_predict'] = bool(model_predict)
        temp_dict['func_priority'] = function_priority_score
        call_type = None
        if 'callers' not in temp_dict.keys():
            temp_dict['callers'] = list()

        if internal_or_external == 'internal':
            call_type = 'internal'
        elif internal_or_external == 'external':
            call_type = 'external'
        elif internal_or_external == 'nocaller':
            call_type = 'nocaller'
        elif internal_or_external == 'fallback':
            call_type = 'undefined'
        sub_dict = dict()
        sub_dict['type'] = call_type

        if model_predict:
            if call_type == 'internal':
                sub_dict['contract'] = internal_call.split('-')[0]
                sub_dict['function'] = internal_call.split('-')[1]
                sub_dict['priority'] = internal_call_score
            elif call_type == 'external':
                sub_dict['contract'] = external_call.split('-')[0]
                sub_dict['function'] = external_call.split('-')[1]
                sub_dict['priority'] = external_call_score
            else:
                sub_dict['contract'] = ''
                sub_dict['function'] = ''
                sub_dict['priority'] = 0.5
        else:
            pass
        
        temp_dict['callers'].append(sub_dict)
        maintain_dict[current_contract][current_function] = temp_dict

    def preprocess(self, ins_list):
        ins_list = self.replace_str(self.concatenate_list(ins_list)).strip()
        ins_list = self.add_feature(ins_list).strip()
        return ins_list
            
    def concatenate_list(self, str_list):
        result = ""
        for x in str_list:
            result += str(x)
            result += " "
        return result

    def add_feature(self, trace):
        ins = [' CALL ', ' DELEGATECALL ', ' ORIGIN ', ' BALANCE ']
        trace += ' '
        for i in ins:
            if i in trace:
                trace += '1'
            else:
                trace += '0'
            trace += ' '
        return trace

    def replace_str(self, strs):
        push = ['PUSH1', 'PUSH2', 'PUSH3', 'PUSH4', 'PUSH5', 'PUSH6', 'PUSH7', 'PUSH8', 'PUSH9', 'PUSH10', 'PUSH11', 'PUSH12', 'PUSH13', 'PUSH14', 'PUSH15', 'PUSH16', 'PUSH17', 'PUSH18', 'PUSH19', 'PUSH20', 'PUSH21', 'PUSH22', 'PUSH23', 'PUSH24', 'PUSH25', 'PUSH26', 'PUSH27', 'PUSH28', 'PUSH29', 'PUSH30', 'PUSH31', 'PUSH32']
        swap = ['SWAP1', 'SWAP2', 'SWAP3', 'SWAP4', 'SWAP5', 'SWAP6', 'SWAP7', 'SWAP8', 'SWAP9', 'SWAP10', 'SWAP11', 'SWAP12', 'SWAP13', 'SWAP14', 'SWAP15', 'SWAP16', 'SWAP17', 'SWAP18', 'SWAP19', 'SWAP20', 'SWAP21', 'SWAP22', 'SWAP23', 'SWAP24', 'SWAP25', 'SWAP26', 'SWAP27', 'SWAP28', 'SWAP29', 'SWAP30', 'SWAP31', 'SWAP32']
        dup = ['DUP1', 'DUP2', 'DUP3', 'DUP4', 'DUP5', 'DUP6', 'DUP7', 'DUP8', 'DUP9', 'DUP10', 'DUP11', 'DUP12', 'DUP13', 'DUP14', 'DUP15', 'DUP16', 'DUP17', 'DUP18', 'DUP19', 'DUP20', 'DUP21', 'DUP22', 'DUP23', 'DUP24', 'DUP25', 'DUP26', 'DUP27', 'DUP28', 'DUP29', 'DUP30', 'DUP31', 'DUP32']
        log = ['LOG0', 'LOG1', 'LOG2', 'LOG3', 'LOG4']
        char_list = strs.strip().split(' ')
        for i, char in enumerate(char_list):
            if char in push:
                char_list = char_list[:i] + ['PUSH'] + char_list[i+1:]
            if char in swap:
                char_list = char_list[:i] + ['SWAP'] + char_list[i+1:]
            if char in dup:
                char_list = char_list[:i] + ['DUP'] + char_list[i+1:]
            if char in log:
                char_list = char_list[:i] + ['LOG'] + char_list[i+1:]
        return self.concatenate_list(char_list)

    # @timeout_decorator.timeout(20, use_signals=False)
    @func_set_timeout(20)
    def _get_CFG(self, CFG, bytecode):
        return CFG(bytecode)

    def _extract_evm_info(self, slither):
        """
        Extract evm information for all derived contracts using evm_cfg_builder

        Returns: evm CFG and Solidity source to Program Counter (pc) mapping
        """

        evm_info = {}

        CFG = load_evm_cfg_builder()

        for contract in slither.contracts_derived:
            contract_bytecode_runtime = slither.crytic_compile.bytecode_runtime(contract.name)
            if len(contract_bytecode_runtime) == 0:
                continue
            contract_srcmap_runtime = slither.crytic_compile.srcmap_runtime(contract.name)
            # cfg = self._get_CFG(CFG, contract_bytecode_runtime)
            try:
                cfg = self._get_CFG(CFG, contract_bytecode_runtime)
            except:
                print("CFG Construction TIME OUT !!!")
                return
            evm_info['cfg', contract.name] = cfg
            evm_info['mapping', contract.name] = generate_source_to_evm_ins_mapping(
                cfg.instructions,
                contract_srcmap_runtime,
                slither,
                contract.source_mapping['filename_absolute'])

            contract_bytecode_init = slither.crytic_compile.bytecode_init(contract.name)
            contract_srcmap_init = slither.crytic_compile.srcmap_init(contract.name)
            cfg_init = CFG(contract_bytecode_init)

            evm_info['cfg_init', contract.name] = cfg_init
            evm_info['mapping_init', contract.name] = generate_source_to_evm_ins_mapping(
                cfg_init.instructions,
                contract_srcmap_init, slither,
                contract.source_mapping['filename_absolute'])

        return evm_info

class MLModelDele(object):
    def __init__(self):
        self._model = self._load_trained_model()

    def _load_trained_model(self):
        url = "https://drive.google.com/uc?id=1825MtDFoT_ojKJ6xRpbmyulFg71mdYpl"
        temp_path = tempfile.gettempdir()
        if not os.path.exists(os.path.join(temp_path, 'eec_dele.joblib')):
            gdown.download(url, os.path.join(temp_path, 'eec_dele.joblib'), quiet=True)
        return joblib.load(os.path.join(temp_path, 'eec_dele.joblib'))

    def predict(self, dataset):
        return self._model.predict(dataset)

class WordVectorizer(object):
    def __init__(self):
        self._wver = self._load_trained_model()

    def _load_trained_model(self):
        url = "https://drive.google.com/uc?id=1P1VrXIMx5Iglm0ek4qcnGuMicxcALaa-"
        temp_path = tempfile.gettempdir()
        if not os.path.exists(os.path.join(temp_path, 'word2vec.model')):
            gdown.download(url, os.path.join(temp_path, 'word2vec.model'), quiet=True)
        # if not os.path.exists(abs_path):
        #     # TODO: throw error
        #     raise ValueError("word2vec model not found")
        return Word2Vec.load(os.path.join(temp_path, "word2vec.model"))

    def vectorize(self, dataset):
        return self._replace_op_with_vec(dataset, self._wver)
    
    def _featurize_w2v(self, model, sentences):
        f = np.zeros((len(sentences), model.vector_size))
        for i, s in enumerate(sentences):
            for w in s:
                try:
                    vec = model[w]
                except KeyError:
                    continue
                f[i,:] = f[i,:] + vec
            f[i,:] = f[i,:] / len(s)
        return f

    def _replace_op_with_vec(self, op_seq, model):
        inter_train_data_list = []
        inter_train_data_list.append(op_seq.split(' '))
        inter_feature = self._featurize_w2v(model, inter_train_data_list)
        return inter_feature
