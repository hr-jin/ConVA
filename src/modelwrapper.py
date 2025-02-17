import torch.nn as nn
from functools import partial

class ModelWrapper(nn.Module):
    def __init__(self, hf_model):
        super(ModelWrapper, self).__init__()
        self.model  = hf_model
        self.device = hf_model.device
        self.representations = {}
        self.gradients = {}
        self.pre_hooks = {}
        self.fwd_hooks = {}
        self.bwd_hooks = {}
        self.cav_hooks = {}
        self.controller_hooks = {}
        self.layers = []
        self.cavs   = {}
    
    def forward(self, *args, **kwargs):
        self.representations = {}
        return self.model.forward(*args, **kwargs, output_hidden_states=True)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def register_forward_pre_hooks(self, layers):
        assert isinstance(layers, list) , "The parameter 'Layers' should be a list."
        def _forward_pre_hook_fn(layer_name, module, input_rep):
            self.representations[layer_name] = input_rep

        for layer_name in layers:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    pre_hook_fn = partial(_forward_pre_hook_fn, layer_name)
                    self.pre_hooks[layer_name] = module.register_forward_pre_hook(pre_hook_fn)
                    print("Load pre hook for layer '{}' successfully!".format(layer_name))
                    break

    def register_forward_hooks(self, layers):
        assert isinstance(layers, list) , "The parameter 'layers' should be a list."
        def _forward_hook_fn(layer_name, module, input, output):
            self.representations[layer_name] = output

        for layer_name in layers:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    forward_hook_fn = partial(_forward_hook_fn, layer_name)
                    self.fwd_hooks[layer_name] = module.register_forward_hook(forward_hook_fn)
                    break

    def register_backward_hooks(self, layers):
        assert isinstance(layers, list), "The parameter 'layers' should be a list."
        def _backward_hook_fn(layer_name, module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0]
        
        for layer_name in layers:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    backward_hook_fn = partial(_backward_hook_fn, layer_name)
                    self.bwd_hooks[layer_name] = module.register_full_backward_hook(backward_hook_fn)
                    print("Load backward hook for layer '{}' successfully!".format(layer_name))
                    break
            
    def remove_hooks(self):
        for layer_name in list(self.pre_hooks.keys()):
            self.pre_hooks[layer_name].remove()
            self.pre_hooks.pop(layer_name)

        for layer_name in list(self.fwd_hooks.keys()):
            self.fwd_hooks[layer_name].remove()
            self.fwd_hooks.pop(layer_name)

        for layer_name in list(self.bwd_hooks.keys()):
            self.bwd_hooks[layer_name].remove()
            self.bwd_hooks.pop(layer_name)

        self.representations = {}
        
    def set_cavs(self, cavs):
        assert isinstance(cavs, dict) , "The parameter 'cavs' should be a dict."
        assert len(self.cav_hooks) == 0, "CAVs has already been set. If you want to set other cavs, please run 'clear_cavs()' method first. "

        def _cav_hook_fn(cav, layer_name, module, input, output):
            if isinstance(output, tuple): 
                output[0].data[:, :, :] = output[0].data[:, :, :] + cav # -1
            else:
                output.data[:, :, :] = output.data[:, :, :] + cav

        for (layer_name, cav) in cavs.items():
            for name, module in self.model.named_modules():
                if name == layer_name:
                    cav_hook = partial(_cav_hook_fn, cav, layer_name)
                    self.cav_hooks[layer_name] = module.register_forward_hook(cav_hook)
                    break
    
    

    def clear_cavs(self):
        for layer_name in list(self.cav_hooks.keys()):
            self.cav_hooks[layer_name].remove()
            self.cav_hooks.pop(layer_name)

