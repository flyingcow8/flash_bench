# Dictionaries to store forward and backward kernel information
# Key: head dimension
# Value: list of dicts
forward_kernels = {"hdim32":[], "hdim64":[], "hdim96":[], "hdim128":[], "hdim160":[], "hdim192":[], "hdim224":[], "hdim256":[]}
forward_splitkv_kernels = {"hdim32":[], "hdim64":[], "hdim96":[], "hdim128":[], "hdim160":[], "hdim192":[], "hdim224":[], "hdim256":[]}
backward_kernels = {"hdim32":[], "hdim64":[], "hdim96":[], "hdim128":[], "hdim160":[], "hdim192":[], "hdim224":[], "hdim256":[]}

import yaml

def register_kernels(yaml_file):
    with open(yaml_file, 'r') as file:
        traits = yaml.safe_load(file)
    
    for op_type, kernels in traits.items():
        for kernel in kernels:
            hdim = f"hdim{kernel['hdim_qk']}"
            # Verify that the hdim is valid
            if hdim not in forward_kernels:
                raise ValueError(f"Invalid head dimension: {hdim}")
            
            if op_type == 'fwd':
                forward_kernels[hdim].append({
                    'kernel_id': kernel['kernel_id'],
                    'block_m': kernel['block_m'],
                    'block_n': kernel['block_n'],
                    'num_warps': kernel['num_warps'],
                    "hdim_v": kernel['hdim_v'],
                    'is_Q_in_regs': kernel.get('is_Q_in_regs', True),
                    'share_Q_K_smem': kernel.get('share_Q_K_smem', True)
                })
            elif op_type == 'fwd_split':
                forward_splitkv_kernels[hdim].append({
                    'kernel_id': kernel['kernel_id'],
                    'block_m': kernel['block_m'],
                    'block_n': kernel['block_n'],
                    'num_warps': kernel['num_warps'],
                    "hdim_v": kernel['hdim_v'],
                    'is_Q_in_regs': kernel.get('is_Q_in_regs', True),
                    'share_Q_K_smem': kernel.get('share_Q_K_smem', True)
                })
            elif op_type == 'bwd':
                backward_kernels[hdim].append({
                    'kernel_id': kernel['kernel_id'],
                    'block_m': kernel['block_m'],
                    'block_n': kernel['block_n'],
                    'num_warps': kernel['num_warps'],
                    "hdim_v": kernel['hdim_v'],
                    'atom_layout_mdq': kernel.get('atom_layout_mdq', 4),
                    'atom_layout_msdp': kernel.get('atom_layout_msdp', 4),
                    'atom_layout_ndkv': kernel.get('atom_layout_ndkv', 1),
                    'is_k_in_regs': kernel.get('is_k_in_regs', True),
                    'is_v_in_regs': kernel.get('is_v_in_regs', True),
                    'no_double_buffer': kernel.get('no_double_buffer', True)
                })
