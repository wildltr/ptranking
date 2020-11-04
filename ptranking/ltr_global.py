import torch

#####################
# Global Attributes
#####################

""" Seed """
ltr_seed = 137

""" A Small Value """
epsilon  = 1e-8


""" GPU Setting If Expected """

global_gpu, global_device, gpu_id = False, 'cpu', None
#global_gpu, global_device, gpu_id = True, 'cuda:0', 0
#global_gpu, global_device, gpu_id = True, 'cuda:1', 1

#
if global_gpu: torch.cuda.set_device(gpu_id)

# a uniform tensor type
tensor      = torch.cuda.FloatTensor if global_gpu else torch.FloatTensor
byte_tensor = torch.cuda.ByteTensor if global_gpu else torch.ByteTensor

# uniform constants
torch_one, torch_half, torch_zero = tensor([1.0]), tensor([0.5]), tensor([0.0])
torch_two = tensor([2.0])
torch_minus_one = tensor([-1.0])

cpu_torch_one, cpu_torch_zero = torch.FloatTensor([1.0]), torch.FloatTensor([0.0])
