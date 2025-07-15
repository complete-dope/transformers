# Implementing Switch Transformer from : https://arxiv.org/pdf/2101.03961

import torch 
import torch.nn as nn 


def switch_transformers(alpha: int , experts: int, batches: int, gate_logits , probabilities ):
    '''
    Args:
        experts: int, number of experts
        batches: int, number of batches
        gate_logits: list[float], gate logits
        probability[i] : N ,B , H , W ( experts @  this is a 3d matrix ) 
        alpha : int
    '''


    routing_weights = torch.nn.functional.softmax(gate_logits, dim = -1)
    _ , selected_experts = torch.topk(routing_weights, k = 4, dim = -1) #values , index 
    expert_mask = torch.nn.functional.one_hot(selected_experts, experts)

    summed_dot_product = torch.matmul(gate_logits , probabilities)



    pass




# input -> x1 -> h x W -> (1 x 768) 
# self attention output ->  (1 x 768)


# Router -> 1 x 768 values 

# total routers => 12 
# engaging the routers (top-k) => 4 

# send each router independently and we will take the best of that...

