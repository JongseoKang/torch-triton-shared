from pyexpat import model
import torch
from torch._dynamo import config
import torchvision.models as models
# from fx_visualizer import FXVisualizer
import torch._inductor.config as inductor_config

from huggingface_hub import login

login(
  token="",
  # add_to_git_credential=True,
)


inductor_config.cpu_backend = "triton"
torch._logging.set_logs(fusion=False, custom=False)
torch.manual_seed(0)

# # testing simple matmul model
# model = torch.nn.Sequential(
#     torch.nn.Linear(1024, 2048),
#     torch.nn.ReLU(),
#     torch.nn.Linear(2048, 2048),
#     torch.nn.ReLU(),
#     torch.nn.Linear(2048, 4096),
#     torch.nn.ReLU(),
# ).to("cpu")
# model = torch.compile(model, backend="inductor", mode="default", fullgraph=True)
# x = torch.randn(256, 1024).to("cpu")
# output = model(x)
# print(output.shape)

model_name = "resnet50"  # Change this to any model you want to test
model = models.vit_b_16(pretrained=True).cpu()
model.eval()
x = torch.randn(256, 3, 224, 224).cpu()  # Adjust input size for the ViT model

# forward through DiT to predict noise residual
compiled_model = torch.compile(model)
output = compiled_model(x)
# with torch.profiler.profile(activities=[
#         torch.profiler.ProfilerActivity.CPU,
#     ],
#     record_shapes=True,
#     with_stack=True,
#     profile_memory=True) as prof:
#     for i in range(0, 5):
#         out = compiled_model(x)
#         prof.step()

# prof.export_chrome_trace("w-fusion-resnet50.json")


############################################# gpt 2
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # 1) Load on CPU (force dtype to float32 or bfloat16 for CPU)
# MODEL_NAME = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
# model.eval()
# model  = torch.compile(model, backend="inductor", mode="default", fullgraph=True)

# # 2) A pure-Python generation function
# def generate_text(prompt: str, max_new_tokens: int = 50, temperature: float = 1.0):
#     inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
#     # Greedy / sampling generation on CPU
#     out = model(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         temperature=temperature,
#         pad_token_id=tokenizer.eos_token_id,
#     )
#     # Decode the generated tokens to text
#     return out

# # warming up the model
# out = generate_text("Code merge sort of c++20", max_new_tokens=10, temperature=0.2)
# print(out.logits.shape)

# with torch.profiler.profile(activities=[
#         torch.profiler.ProfilerActivity.CPU,
#     ],
#     record_shapes=True,
#     with_stack=True,
#     profile_memory=True) as prof:
#   for i in range(0, 5):
#     out = generate_text("Code merge sort of c++20", max_new_tokens=10, temperature=0.2)
#     prof.step()

# prof.export_chrome_trace("w-fusion-wo-avx-gpt2.json")

'''
# llama 4, need access on Hugging Face
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
).to("cpu")
model.eval()
model = torch.compile(model, backend="inductor", mode="default", fullgraph=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "code 2D segment tree with c++20"},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model(
    **inputs,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
print(outputs[0])
'''

######################################## Stable Diffusion 3.5

# from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
# from transformers import BitsAndBytesConfig

# model_id = "stabilityai/stable-diffusion-3.5-large"

# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=False,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# model_nf4 = SD3Transformer2DModel.from_pretrained(
#     model_id,
#     subfolder="transformer",
#     quantization_config=nf4_config,
#     torch_dtype=torch.bfloat16
# ).to("cpu")

# pipeline = StableDiffusion3Pipeline.from_pretrained(
#     model_id, 
#     transformer=model_nf4,
#     torch_dtype=torch.bfloat16
# ).to("cpu")
# pipeline.transformer = torch.compile(pipeline.transformer, backend="inductor", mode="default", fullgraph=True)

# prompt = "macbook pro with samsung logo"

# image = pipeline(
#     prompt=prompt,
#     num_inference_steps=28,
#     guidance_scale=4.5,
#     max_sequence_length=512,
# ).images[0]
# image.save("whimsical.png")


################################# 
# from diffusers import DiffusionPipeline
# import torch

# path = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# run_compile = True  # Set True / False

# pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
# pipe.unet.to(memory_format=torch.channels_last)

# if run_compile:
#     print("Run torch compile")
#     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# prompt = "ghibli style, a fantasy landscape with castles"

# for _ in range(3):
#     images = pipe(prompt=prompt).images

# import torch
# from diffusers import DiffusionPipeline
# from diffusers.models.attention_processor import AttnProcessor

# pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32).to("cpu")
# pipe.unet.set_default_attn_processor()

# pipe.unet = torch.compile(pipe.unet, mode="default", fullgraph=True)
# images = pipe("trashcan", num_inference_steps=1, num_images_per_prompt=1).images

# import torch.profiler
# with torch.profiler.profile(activities=[
#         torch.profiler.ProfilerActivity.CPU,
#     ],
#     record_shapes=True,
#     with_stack=True,
#     profile_memory=True) as prof:
#     for _ in range(5):
#         images = pipe("trashcan", num_inference_steps=1, num_images_per_prompt=1).images

# prof.export_chrome_trace("wo-vertical-wo-avx-sd1.json")
