from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda'
torch.manual_seed(42)


@dataclass
class GPTConfig:
    block_size:int = 8
    vocab_size:int = 50257
    n_embd:int = 768
    n_layer:int = 12
    n_head = 1
        

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size , config.n_embd, device=device),
            wpe = nn.Embedding(config.block_size , config.n_embd, device=device),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd , config.vocab_size , bias=False, device= device)

    def forward(self, idx, target=None):
        B,T = idx.shape # 32 x 1024
        # idx = idx.to(dtype=torch.long, device=self.lm_head.weight.device)  

        # idx_long = idx.clone().long()
        assert T <= self.config.block_size , f"Cannot forward , because the sequence length is too long !! T : {T} and block_size is {self.config.block_size} "

        pos = torch.arange(0,T,dtype=torch.long,device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T,C)
        tok_emb = self.transformer.wte(idx) # (B,T,C)
        pos_emb = pos_emb.unsqueeze(0).expand(B,-1,-1) # (1,T,C)
        # print(f">> Positional Embedding shape is : {pos_emb.shape} and dtype is {pos_emb.dtype} and the Token Embedding shape is : {tok_emb.shape} and dtype is {tok_emb.dtype}")
        x = tok_emb + pos_emb # (B,T,C)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        # target (B,T)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1)) # (BT,vocab_size) * (BT)
            return logits , loss
        # output = F.softmax(logits,dim=-1) 
        # loss = F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
        return logits , loss

        
class Block(nn.Module):
    '''
    The residual connections should not have normalisations in them , as there role is to provide a direct path for gradients to flow through the network . 
    Adding normalisation in there will affect the gradient that is the flowing through it !!  
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class CausalSelfAttention(nn.Module): # attention operation ! 
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd , 3 * config.n_embd , bias = True, device= device)
        self.c_proj = nn.Linear(config.n_embd , config.n_embd, device=device)
        self.register_buffer('bias' , torch.tril(torch.ones(config.block_size , config.block_size)))
        # print(f"self.bias shape is  : {self.bias.shape}")
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self , x):
        B,T,C = x.size() #batch , token_count , embedding_dim
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd , dim=2)
        q = q.view(B, T, self.n_head , self.n_embd // self.n_head).transpose(1,2)
        k = k.view(B, T, self.n_head , self.n_embd // self.n_head).transpose(1,2)  
        v = v.view(B, T, self.n_head , self.n_embd // self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # print("att shape is : ",att.shape)
        # print("bias shape is : ",self.bias.shape)
        att = att.masked_fill(self.bias[:T,:T]==0 , value =float('-inf'))
        att = F.softmax(att , dim=-1)
        y = att @ v
        y = y.transpose(1,2) # (B, T, nh, hs)

        y =y.contiguous().view(B,T,C)
        y = self.c_proj(y) # aka W_O 
        return y

        
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd , 4 * config.n_embd , device= device)
        self.gelu = nn.GELU(approximate='tanh') # mathematical operation only ! ( nothing gets stored !! )
        self.c_proj = nn.Linear(4 * config.n_embd , config.n_embd, device=device)

    def forward(self , x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

# ---------- 

def train():
    # sampling loop
    model = GPT(GPTConfig())
    model.to(device)

    B , T = 4 ,GPTConfig.block_size

    # THIS CANT BE RANDOM AS THIS SHOULD COME FROM THE DATA !! 
    buf = torch.randint(low =0 , high = 50257, size = (B*T +1,), device=device)
    x = buf[:B*T].view(B,T)
    y = buf[1:].view(B,T)
    print("Input is ", x) 
    print("Target is ", y)
    optimizer = torch.optim.AdamW(model.parameters() , lr=3e-4)

    for epoch in range(1):
        for i in range(50):
            optimizer.zero_grad() # this needs to be get zero as we already made the updates and this should be cleared now !  (unless doing batch updates on the dataset) !! 
            logits , loss = model.forward(x,y) # from here it knows !! 
            loss.backward()
            optimizer.step() # already updated the weights !
            print(f"Epoch : {epoch} , Step {i+1} , loss : {loss.item()}")

    print("Saving the model weights !")
    torch.save(model.state_dict() , 'gpt2_weights.pth')


# ----------

'''
# lets build MLA ( Multi-Head Latent Attention ) 
the dimensions will increase dramatically 
'''


if __name__ == "__main__":
    train()

    