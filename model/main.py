# This is the main model !!

import torch 
import torch.nn as nn

class Preprocessing(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = 30000 # total no of tokens in shakespeare vocab
        self.vocab_dim = 768
        self.embedding_table = nn.Embedding(self.vocab_size , self.vocab_dim , dtype=torch.float) # created a embedding table

    def forward(self , x: torch.Tensor):
        # x -> bs , seq_len ( 32 x 8 )
        # x -> [1,2,3,4] = []

        if len(x) != 8: 
            x = torch.cat((torch.zeros(8-len(x), dtype=torch.float) ,  x), dim = 0)

        # self.embedding_table    
        # print(self.embedding_table)
        x = x.to(torch.int64)
        print(">> Updated length is " , len(x) ,x)
        x = self.embedding_table(x)
        return x # 8x768 
    

class Attention(nn.Module):
    def __init__(self, embed_size=768, heads=8):
        super(Attention, self).__init__()
        self.embed_size = embed_size # 768 
        self.heads = heads # 8 
        self.head_dim = embed_size // heads # 768 / 8 = 96
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

    def forward(self , x):
        # forward pass for BS = 1 
        # x -> 8 x 768  seq_len , dimension
        # x -> 8 x 8 x 96 seq_len , heads , head_dim

        x = x.view(x.shape[0] , self.heads , self.head_dim) 
        print(">> Updated shape of x is ", x.shape)

        keys = self.keys(x) # 8 x 8 x 96   
        queries  = self.queries(x) # 8 x 8 x 96
        values = self.values(x) # 8 x 8 x 96
        
        # keys = keys.view(keys.shape[0] , keys.shape[2] , keys.shape[1])
        keys = torch.transpose(keys , 1 , 2) # 8 x 96 x 8
        print(keys.shape)
        print(queries.shape)
        numerator = queries @ keys

        print(">> Updated shape of numerator is ", numerator.shape)
        numerator = numerator / (self.head_dim ** 0.5)
        distributed_attentions = torch.softmax(numerator , dim=-1) @ values
        print(">> Updated shape of distributed_attentions is ", distributed_attentions.shape)

        return distributed_attentions

    def backward(self):
        
        
        pass



if __name__ == "__main__":
    model = Preprocessing()
    x = torch.tensor([1,2,3,4])
    input = model(x)

    attention = Attention()
    output = attention.forward(input)
    print(f">> OUTPUT SHAPE IS : {output.shape}")
            




        