"""
    Module printing evm mapping of the contract
"""
from slither.printers.abstract_printer import AbstractPrinter
from slither.analyses.evm import generate_source_to_evm_ins_mapping, load_evm_cfg_builder
from slither.utils.colors import blue, green, magenta, red

import json

class PrinterEVMFunc(AbstractPrinter):
    ARGUMENT = '++function-ins++'
    HELP = 'Print the evm instructions of nodes in functions'

    WIKI = 'https://github.com/trailofbits/slither/wiki/Printer-documentation#evm'

    def output(self, _filename):
        """
            _filename is not used
            Args:
                _filename(string)
        """
        file_dict = dict()
        txt = ""

        if not self.slither.crytic_compile:
            txt = 'The EVM printer requires to compile with crytic-compile'
            self.info(red(txt))
            res = self.generate_output(txt)
            return res
        evm_info = self._extract_evm_info(self.slither)

        for contract in self.slither.contracts_derived:
            if contract.name not in file_dict.keys():
                file_dict[contract.name] = dict()
            txt += blue('Contract {}\n'.format(contract.name))

            contract_file = self.slither.source_code[contract.source_mapping['filename_absolute']].encode('utf-8')
            contract_file_lines = open(contract.source_mapping['filename_absolute'], 'r').readlines()

            contract_pcs = {}
            contract_cfg = {}

            for function in contract.functions:
                library_call_count = 0
                can_send_eth = False
                function_name = function.canonical_name.split('.')[1].split('(')[0]
                if function.canonical_name.split('.')[0] != contract.name:
                    continue
                if  function_name not in file_dict[contract.name].keys():
                    file_dict[contract.name][function_name] = list()
                txt += blue(f'\tFunction {function.canonical_name}\n')

                # CFG and source mapping depend on function being constructor or not

                # if evm_info == {}:
                #     continue
                try:
                    if function.is_constructor:
                        contract_cfg = evm_info['cfg_init', contract.name]
                        contract_pcs = evm_info['mapping_init', contract.name]
                    else:
                        contract_cfg = evm_info['cfg', contract.name]
                        contract_pcs = evm_info['mapping', contract.name]
                except:
                    continue
                    
                has_dangerous_call = False  # detect function has external transfer function
                can_send_eth_list = []
                for node in function.nodes:
                    txt += green("\t\tNode: " + str(node) + "\n")
                    node_source_line = contract_file[0:node.source_mapping['start']].count("\n".encode("utf-8")) + 1
                    txt += green('\t\tSource line {}: {}\n'.format(node_source_line,
                                                                   contract_file_lines[node_source_line - 1].rstrip()))
                    txt += magenta('\t\tEVM Instructions:\n')
                    node_pcs = contract_pcs.get(node_source_line, [])
                    for pc in node_pcs:
                        txt += magenta('\t\t\t0x{:x}: {}\n'.format(int(pc), contract_cfg.get_instruction_at(pc)))
                        ins = contract_cfg.get_instruction_at(pc).name.encode('utf-8').decode('utf-8-sig')
                        if len(ins.split(' ')) == 2:
                            ins = ins.split(' ')[0]
                        file_dict[contract.name][function_name].append(ins)
                    
                    if node.irs != []:
                        can_send_eth_list.append(node.can_send_eth())
                    
                    library_call_count += len(node.library_calls)
                    if len(node.external_calls_as_expressions) > 0:
                        for c in node.external_calls_as_expressions:
                            called_function = c.called
                            # arguments = [x.value for x in c.arguments]
                            if str(called_function).split('.')[-1] == 'transfer' and 'msg.sender' in str(c):
                                has_dangerous_call = True

                
                if len(file_dict[contract.name][function_name]) == 0:
                    continue
                has_private_visibility = False
                if function.visibility == 'private': # detect function has private
                    has_private_visibility = True

                have_specific_modifier = False
                for modif in function._modifiers: # detect function has arbitrary modifiers
                    if modif not in ['private', 'payable', 'external', 'internal', 'public']:
                        have_specific_modifier = True

                can_send_eth = True in can_send_eth_list

                if have_specific_modifier:
                    file_dict[contract.name][function_name].append(1)
                else:
                    file_dict[contract.name][function_name].append(0)
                if can_send_eth:
                    file_dict[contract.name][function_name].append(1)
                else:
                    file_dict[contract.name][function_name].append(0)
                if has_dangerous_call:
                    file_dict[contract.name][function_name].append(1)
                else:
                    file_dict[contract.name][function_name].append(0)


            # for modifier in contract.modifiers:
            #     txt += blue(f'\tModifier {modifier.canonical_name}\n')
            #     for node in modifier.nodes:
            #         txt += green("\t\tNode: " + str(node) + "\n")
            #         node_source_line = contract_file[0:node.source_mapping['start']].count("\n".encode("utf-8")) + 1
            #         txt += green('\t\tSource line {}: {}\n'.format(node_source_line,
            #                                                        contract_file_lines[node_source_line - 1].rstrip()))
            #         txt += magenta('\t\tEVM Instructions:\n')
            #         node_pcs = contract_pcs.get(node_source_line, [])
            #         for pc in node_pcs:
            #             txt += magenta('\t\t\t0x{:x}: {}\n'.format(int(pc), contract_cfg.get_instruction_at(pc)))

        print(file_dict)
        self.info("")
        res = self.generate_output("")
        return res

    # def extract_syntax_features(self):
    #     for contract in self.slither.contracts_derived:


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
