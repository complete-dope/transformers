import torch 
import torch.nn as nn 
import warnings 
warnings.filterwarnings(action="ignore")
from typing import Union, List



class Attention(nn.Module): 
    def __init__(self, n_dim:int, k_dim: int, v_dim : int, heads:int):
        super().__init__()
        self.query = nn.Linear(n_dim, k_dim)
        self.keys = nn.Linear(n_dim ,k_dim )
        self.values = nn.Linear(n_dim, v_dim)
        self.output = nn.Linear(v_dim, n_dim)
        self.heads = heads
        self.n_dim = n_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
    
    def vanillaAttention(self, logits:torch.Tensor):
        '''
        logits.shape = (no. of tokens x token_dim)

        Padding : to match the seqlength of each entry in the batch we need to pass the tokens, that tells which are the actual tokens and which are the padded tokens, so this is the use of pad_tokens 
        '''
        N,D = logits.shape
        q = self.query(logits)
        k = self.keys(logits)
        v = self.values(logits)
        print("k.shape" , k.shape) # 
        print("q.shape" , q.shape)
    
        x = torch.nn.functional.softmax((q @ k.T) * (self.k_dim)**(-0.5))
        assert x.shape == (N, N)

        x = torch.tril(x, diagonal=-1)
        assert x.shape == (N, N)

        attn_matrix = x @ v 
        assert attn_matrix.shape == (N , self.v_dim) , "Mat mul didnt went as aspected"

        attention = self.output(x @ v)
        assert attention.shape == logits.shape

        return attention


    def multiHeadAttention(self, logits: torch.Tensor):
        '''
        Researchers hypothesized that dividing the attn in heads would led to each head capture some different semantic ( one is good for calculation , 2nd takes in aspects , 3rd takes in features/subspaces ... etc) but the init method is same for the whole .. so this raises a QUESTION on there hypothesis ??

        are 16 heads better than one ? : https://arxiv.org/pdf/1905.10650

        Seems like most of the heads are useless and most of them just add's up useless information 
        '''
        
        N, D = logits.shape # D=n_dim
        print(f"N : {N} ,  D: {D}")

        q = self.query(logits) # ( N x k_dim)
        k = self.keys(logits)
        v = self.values(logits)
        print("k.shape" , k.shape) # 
        print("q.shape" , q.shape)

        assert self.n_dim % self.heads == 0 , "Oops, cant divide into equal matrix values for the heads"

        q = q.view(self.heads , N, self.k_dim//self.heads)
        k = k.view(self.heads, N, self.k_dim//self.heads)
        v = v.view(self.heads, N, self.v_dim//self.heads)
        print("q.shape" , q.shape)


        # attention matrix of tokens ( N x heads x heads  )
        k = torch.transpose(k,-1, -2)
        assert k.shape == (self.heads, self.k_dim//self.heads , N) , f"K shape is {k.shape}"
        print("k.shape" , k.shape) 

        attn_matrix = torch.matmul(q, k) * (D**(-0.5))
        assert attn_matrix.shape == (self.heads, N, N), f"attn_matrix size is {attn_matrix.shape}"

        attn_matrix = torch.tril(attn_matrix, diagonal=-1)
        assert attn_matrix.shape == (self.heads, N, N), f"attn_matrix size is {attn_matrix.shape}"

        attn = torch.nn.functional.softmax(attn_matrix, dim = -1) @ v # done scene 
        assert attn.shape == (self.heads, N , self.v_dim//self.heads)
        
        attn = attn.view(N, self.v_dim)

        attention = self.output(attn)
        assert attention.shape == logits.shape

        return attention


    def lightningAttention(self, inputs: torch.Tensor):
        '''
        inputs shape = (N , L) 
        L = B . b
        B are the no. of blocks, each of sequence size b 
        '''
        BS, SL = inputs.shape # batch size, sequence length

        B =  # no. of blocks
        b =  # block size in each sequence 
        assert B * b == SL , "The Sequence length is equal to no. of blocks * block size" # B =  




class FFN(nn.Module):
    def __init__(self, n_dim:int , hidden_dim:int = 2048):
        self.layer1= nn.Linear(n_dim, hidden_dim, bias = True)
        self.layer2= nn.Linear(hidden_dim, n_dim , bias = True)
        self.n_dim = n_dim
        pass

    def vanilla_ffn(self, x:torch.Tensor):
        assert x.shape[-1] == self.n_dim, f"The Shape of x is {x.shape}"
        y = torch.nn.functional.relu(self.layer1(x))
        y = self.layer2(y)
        return y

    def switch_transformer(self, gate_logits:torch.Tensor ,num_experts:int , attention_mask:Union[torch.Tensor, None], top_k=2):
        '''
        This is used in specifically for MOE models 
        Minimax paper, where they used switch transformer 
        
        https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/blob/main/modeling_minimax_m1.py
        
        This step comes after the self attention has been done and just for the ffn it comes , the rationale behind switch transformer is ?    
        '''

        if isinstance(gate_logits ,tuple):
            '''
            Here we are taking logits from all the model layers because of the load balancing loss and don't require this for normal routing and just the final logits would also work 
            '''
            compute_device = gate_logits[0].device
            concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim = 0)

        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim = -1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)


        if attention_mask is None:
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0) # average no. of tokens per expert 
            router_prob_per_expert = torch.mean(tokens_per_expert, dim=0) # probability of going towards a token 

        else:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size*sequence_length)

            # removing the padded tokens 
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
            )

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
                expert_attention_mask, dim=0
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
                router_per_expert_attention_mask, dim=0
            )

        overall_loss = torch.sum(routing_weights * router_prob_per_expert , dim=0) / torch.sum(router_prob_per_expert, dim =0)

        return overall_loss


class Norms(nn.Module):
    def __init__(self): 
        super().__init__()
        self.scale = torch.nn.Parameter(data = torch.randn(size = (1)))
        self.shift = torch.nn.Parameter(data = torch.randn(size = (1))) 
        pass

        
    def batchNorm(input:torch.Tensor) -> torch.Tensor:
        B = input.shape[0] # input : B, T, E
        eps = 1e-5
        scale = 1
        shift = 0

        x = torch.empty_like(input=input)
        print("shape of x is " , x.shape)

        for i in range(T):
            batched_single_token =  input[:,i,:] # B,1,E
            print("expecting shape to be b,1,e ", batched_single_token.shape)
            # index wise mean ?
            batched_mean = torch.mean(batched_single_token, dim = 0) # 1,1,E
            print("expecting shape to be 1,1,e ", batched_mean.shape)


            print((batched_single_token - batched_mean).shape) # auto-broadcast
            print(torch.pow(batched_single_token - batched_mean, 2).shape)
            print((torch.sum(batched_single_token - batched_mean,keepdim=True, dim = -1)).shape)

            #variance
            variance = torch.sum(torch.pow(batched_single_token - batched_mean, 2), dim  = -1, keepdim =True) / B
            print("expecting shape to be B,1", variance.shape)

            # normalise
            x_hat = (batched_single_token-batched_mean) / torch.sqrt(variance + eps)
            print("expecting shape to be B,1,E ", x_hat.shape)

            # shift and scale
            x[:,i,:] = scale * x_hat + shift

        return x

        

    def layerNorm(input : torch.Tensor):

        B,T,E = input.shape
        x = torch.empty_like(input)

        for i in range(B):
            layered_input = input[i , : , : ] # T,E

            meaned = torch.mean(layered_input, dim = -1, keepdim = True) # T
            print("meaned shape" , meaned.shape)

            variance = torch.sum(torch.pow(layered_input - meaned , 2 ), dim = -1, keepdim = True) # T
            print("variance shape", variance.shape)

            updated = (layered_input - meaned) / torch.sqrt(variance + 1e-5) # T,E
            x[i,:,:] = updated

        return x


    def rmsNorm(input: torch.Tensor, eps=1e-8):
        B, T, E = input.shape
        x = torch.empty_like(input)

        for i in range(B):
            for j in range(T):
                vec = input[i, j, :]  # shape [E]
                rms = torch.sqrt(torch.mean(vec ** 2) + eps)  # scalar
                x[i, j, :] = vec / rms  # normalized vector

        return x




        
    def layerNorm(self): 
        pass

    
    def rmsNorm(self):
        pass




seed=42
torch.manual_seed(seed)

# testing 
# x = torch.Tensor([1.0, 2.0, 3.0, 4.0]) # 4x1
# x = x.unsqueeze(dim = 1)
# assert x.dtype == torch.float, f"Mismatch datatype {x.dtype}"
# mma = Attention(x.shape[-1], 16,16,1)
# attn = mma.vanillaAttention(x)
# print(attn)


# Multi head
N = 10 # this should be equal to the tokens / context window length
D = 768
x = torch.randn(size = (N, D))
mma = Attention(D , 1536, 1536 , 12)
attn = mma.multiHeadAttention(x)
print(attn.shape)



# single batch dataset 
batched_dataset =  "I am a good boy" , "I am very bad" , "I am taking an action to relearn the transformer"

# tokenization  


