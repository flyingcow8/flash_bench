import torch
from torch.profiler import profile, record_function, ProfilerActivity


def pytorch_profiler(fn, *inputs, cpu=False, verbose=True, **kwinputs):
    # Warm up
    for _ in range(30):
        fn(*inputs, **kwinputs)

    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [
        torch.profiler.ProfilerActivity.CUDA
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
    ) as prof:
        repeat_times = 1000  # You can adjust this value as needed
        for _ in range(repeat_times):
            fn(*inputs, **kwinputs)
    if verbose:
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        # print(prof.key_averages().table(row_limit=50))
    return prof


def find_flash_bwd_cuda_time(prof):
    for item in prof.key_averages():
        if 'flash_bwd_dq_dk_dv' in item.key:
            return item.cuda_time # returns time in microseconds (μs)
    return None  # Return None if 'flash_bwd' is not found

def find_flash_fwd_cuda_time(prof):
    for item in prof.key_averages():
        if 'flash_fwd_kernel' in item.key:
            return item.cuda_time # returns time in microseconds (μs)
    return None  # Return None if 'flash_fwd' is not found

def find_flash_fwd_splitkv_cuda_time(prof):
    for item in prof.key_averages():
        if 'flash_fwd_splitkv_kernel' in item.key:
            return item.cuda_time # returns time in microseconds (μs)
    return None  # Return None if 'flash_fwd_splitkv' is not found

# def pytorch_profiler(
#     fn,
#     *inputs,
#     backward=False,
#     amp=False,
#     amp_dtype=torch.float16,
#     cpu=False,
#     verbose=True,
#     **kwinputs,
# ):
#     """Wrap benchmark functions in Pytorch profiler to see CUDA information."""
#     if backward:
#         with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
#             out = fn(*inputs, **kwinputs)
#             if type(out) is tuple:
#                 out = out[0]
#             g = torch.randn_like(out)
#     for _ in range(30):  # Warm up
#         if backward:
#             for x in inputs:
#                 if isinstance(x, torch.Tensor):
#                     x.grad = None
#         with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
#             out = fn(*inputs, **kwinputs)
#             if type(out) is tuple:
#                 out = out[0]
#         # Backward should be done outside autocast
#         if backward:
#             out.backward(g, retain_graph=True)
#     activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [
#         torch.profiler.ProfilerActivity.CUDA
#     ]
#     with torch.profiler.profile(
#         activities=activities,
#         record_shapes=True,
#         # profile_memory=True,
#         with_stack=True,
#     ) as prof:
#         repeat_times = 1000  # You can adjust this value as needed
#         for _ in range(repeat_times):
#             if backward:
#                 for x in inputs:
#                     if isinstance(x, torch.Tensor):
#                         x.grad = None
#             with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
#                 out = fn(*inputs, **kwinputs)
#                 if type(out) is tuple:
#                     out = out[0]
#             if backward:
#                 out.backward(g, retain_graph=True)
#     if verbose:
#         print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
#         # print(prof.key_averages().table(row_limit=50))
#     return prof
