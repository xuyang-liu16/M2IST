# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class ShareAdapter(nn.Module):
    def __init__(self,
                 d_model=128,
                 bottleneck=256,
                 adapter_scalar="1"
                ):
        super().__init__()
        self.n_embd = d_model
        self.up_size = bottleneck
        # self.scale = float(adapter_scalar)

      
        self.up_proj = nn.Linear(self.n_embd, self.up_size)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
            torch.nn.init.uniform_(self.up_proj.bias)


    def forward(self, x):
        up = self.up_proj(x)
       
       
        return up


class Adapter_text(nn.Module):
    def __init__(self,
                 d_model=768,
                 bottleneck=128,
                 dropout=0.0,
                 decoder=False,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.decoder = decoder
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
           
        self.adapter_layer_norm_cross_modal = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
  
        self.dropout = dropout
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
            torch.nn.init.uniform_(self.up_proj.bias)

            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            torch.nn.init.uniform_(self.down_proj.bias)



    def forward(self, x, add_residual=False, mode='text'):
        residual = x 
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

        
        if mode == 'text':
            up = self.up_proj(down)
       
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
    
class Adapter_vis(nn.Module):
    def __init__(self,
                 d_model=256,
                 bottleneck=128,
                 dropout=0.0,
                 decoder=False,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.decoder = decoder
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
           
        self.adapter_layer_norm_cross_modal = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()

       
        self.visual_up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.visual_up_proj.weight, a=math.sqrt(5))
            torch.nn.init.uniform_(self.visual_up_proj.bias)

            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            torch.nn.init.uniform_(self.down_proj.bias)
      
           
    def forward(self, x, add_residual=False, mode='visual'):
        residual = x 
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

      
        if mode == 'visual':
            up = self.visual_up_proj(down)
       
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class Adapter_Lora(nn.Module):
    def __init__(self,
                 d_model=768,
                 bottleneck=64,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="learnable_scalar",):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout

    def init_adapter_weights(self,):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True):
        down = self.down_proj(x)
        # down = self.non_linear_func(down)
        up = self.up_proj(down)
        output = up * self.scale
        return output
