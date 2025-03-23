# This is how I load the model weights and then use it for inference !
import torch 
import torch.nn.functional as F

device = "cuda"
# load the weights
loaded_weights = torch.load("gpt2_weights.pth")

# print(dir(loaded_weights))
# print(loaded_weights.keys())

# architecture structure .. 
from train_gpt2 import GPT, GPTConfig

model = GPT(GPTConfig())
model.load_state_dict(loaded_weights)
model.to(device)
model.eval()

# NOW we need to test the model for inference !!

torch.manual_seed(42)

inference_tokens = torch.randint(low =0 , high = 50257, size = (1,3), 
device=device)
full_sequence = inference_tokens

for i in range(20):
    print(f"index is : {i}")
    logits,_ = model.forward(inference_tokens)
    print(f"logits shape is {logits.shape}")
    generated_token = torch.argmax(F.softmax(logits[:,-1,:],dim=-1))
    # print("")
    if generated_token.item() > 50256:
        break
    generated_token = torch.tensor([[generated_token]], device = device)
    print('inference token shape is : ', inference_tokens.shape)
    print('generated token shape is : ', generated_token.shape)
    inference_tokens = torch.cat((inference_tokens , generated_token), dim = 1)
    inference_tokens = inference_tokens[:, -(GPTConfig.block_size):] # sliding window attention !!
    # print('inference token shape is : ', inference_tokens.shape)
    print(f"Generated token is : {generated_token} and the Inference tokens are : {inference_tokens}")

    full_sequence = torch.cat((full_sequence , generated_token), dim = 1)
    

print("Full sequence is for 20 tokens : ", full_sequence)

