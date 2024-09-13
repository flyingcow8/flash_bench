# Dictionaries to store forward and backward kernel information
# Key: head dimension
# Value: list of tuples (tile_m, tile_n, num_waves)
forward_kernels = {}
backward_kernels = {}

def register_kernel(kernels_dict, head_dim, tile_m, tile_n, num_waves):
    if head_dim not in kernels_dict:
        kernels_dict[head_dim] = []
    kernels_dict[head_dim].append((tile_m, tile_n, num_waves))


# Register forward kernels
register_kernel(forward_kernels, 96, 64, 64, 4)
register_kernel(forward_kernels, 96, 128, 64, 4)
register_kernel(forward_kernels, 128, 64, 64, 4)

# Register backward kernels
register_kernel(backward_kernels, 96, 32, 64, 4)
register_kernel(backward_kernels, 128, 32, 64, 4)
register_kernel(backward_kernels, 128, 32, 128, 8)

