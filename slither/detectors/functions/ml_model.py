from slither.detectors.abstract_detector import AbstractDetector, DetectorClassification
from slither.printers.abstract_printer import AbstractPrinter
from slither.analyses.evm import generate_source_to_evm_ins_mapping, load_evm_cfg_builder
from slither.utils.colors import blue, green, magenta, red


class ModelPrediction(AbstractDetector):
    """
    Predictor Based On Machine Learning Method
    """

    ARGUMENT = 'model-prediction'
    HELP = 'Model Prediction'
    IMPACT = DetectorClassification.INFORMATIONAL
    CONFIDENCE = DetectorClassification.HIGH

    WIKI = 'https://github.com/trailofbits/slither/wiki/Printer-documentation#evm'
    WIKI_TITLE = 'Model Prediction'
    WIKI_DESCRIPTION = 'This detector is based on trained machine learning models'
    WIKI_RECOMMENDATION = 'Avoid low-level calls. Check the call success. If the call is meant for a contract, check for code existence.'

    def _detect(self):
        """ Predict the function has suspicious vulnerabilities
        """
        txt = ""
        file_dict = dict()

        if not self.slither.crytic_compile:
            txt = 'The EVM printer requires to compile with crytic-compile'
            self.info(red(txt))
            res = self.generate_output(txt)
            return res

        evm_info = self._extract_evm_info(self.slither)

        results = []
        for contract in self.slither.contracts_derived:
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

                for node in function.nodes:
                    node_source_line = contract_file[0:node.source_mapping['start']].count("\n".encode("utf-8"))+1
                    node_pcs = contract_pcs.get(node_source_line, [])
                    
                    ins_list = list()
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
                print(ins_list)

        res = self.generate_result("")
        results.append(res)

        return results

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
        ins = [' CALL ', 'SELFDESTRUCT', 'ORIGIN', 'BALANCE']
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
            cfg = CFG(contract_bytecode_runtime)
            # if cfg.fail_flag:
            #     continue
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
