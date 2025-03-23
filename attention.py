

# lets build an attention model from scratch only attention nothing else on vim !!

class Attention():
    def __init__(self, **kwargs):
        self.vocab_size = 50257 #kwargs.get('vocab_size')
        self.n_dims = 768
        self.n_heads = 8
        self.projection = nn.Linear(self.n_dims , 3*self.n_dims) # projects for k q v
    
    def forward(self, tokens):
        '''
        E = self.n_dims
        B = batch_size
        T = no of tokens ( training input size !! )
        '''
        B,T,E = tokens.shape 

        kqv = self.projection(tokens)
        k,q,v = torch.split(kqv , chunks =3, dim = -1) 
         
        # k.shape => B,T,E 
        # divide to heads and pass each head to it !!
        k = k.view(B,T,self.n_heads, E//self.n_heads).transpose(1,2) # last 2 only goes for 2d multiplication 
        v = v.view(B,T,self.n_heads , E//self.n_heads).transpose(1,2)
        q = q.view(B,T,self.n_heads, E//self.n_heads).transpose(1,2)

        # do the block matrix multiplication for efficiency and parallelism

        raw_attention = q @ k.T # B,H,T,T
        attn_scores = F.softmax(raw_attention/E**(-0.5)) # B,H,T,T

        attention_values = attn_score @ v # B,H,T, E//H
        return attention_values.view(B,T,-1) # concat all the tensors !  




## add this 

'''

## Single head and multiple attention head 

We believe that the same positions in the linear layer when passed through a single embedding tries to cover only a single pattern and we need to learn multiple patterns so that we can make chunk , and each chunk has its seperate subspace (but same latent space) in which it can learn its own patterns !! 


Each head is trying to capture something different ( really ? good knows ? similar to in conv each filter tries to learn something new !! do we really motivate the model to come up with this behavior ?? NO !! ) and the model learns the intricacies of the language !!


## ADDING THE REGULARISATION LAYER !!

'''
