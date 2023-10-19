import torch
import comfy.cli_args
comfy.cli_args.enable_ait = True

import comfy.sd
import comfy.model_management
import comfy.model_detection
import comfy.utils
import sys
import os

sys.setrecursionlimit(10000) #Needed for SDXL


ckpt_path = "models/checkpoints/sd_xl_base_1.0_0.9vae.safetensors"
# ckpt_path = "models/checkpoints/cardosAnime_v20.safetensors"

unet_name = "SDXL"

#NOTE: to get CUDA working with cmake/ms build tools on windows copy the contents of:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\visual_studio_integration\MSBuildExtensions
# to:
# C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations

os.environ["AIT_USE_CMAKE_COMPILATION"] = "1"
os.environ["AIT_ENABLE_CUDA_LTO"] = "1"

fp16 = True
sd = comfy.utils.load_torch_file(ckpt_path)
model_config = comfy.model_detection.model_config_from_unet(sd, "model.diffusion_model.", fp16)
model_config.unet_config['dtype'] = 'float16'
model = model_config.get_model(sd, "model.diffusion_model.",)


unet = model.diffusion_model

unet.name_parameter_tensor()
x = dict(unet.named_parameters())

for k in x:
    print(k, x[k])

from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor, IntVar
from aitemplate.testing import detect_target

def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"] for d in y[i]._attrs["shape"]]
        print("AIT output_{} shape: {}".format(i, y_shape))

batch_size = IntVar(values=(1, 16), name="batch_size")
height = IntVar(values=(32, 256))
width = IntVar(values=(32, 256))
prompt_size = IntVar(values=(77, 77*4))

hidden_dim = model_config.unet_config['context_dim']

latent_model_input_ait = Tensor(
    [batch_size, height, width, 4], name="x", is_input=True, dtype='float16'
)

timesteps_ait = Tensor([batch_size], name="timesteps", is_input=True)
text_embeddings_pt_ait = Tensor(
    [batch_size, prompt_size, hidden_dim], name="context", is_input=True
)

if model.adm_channels == 0:
    y_ait = None
else:
    y_ait = Tensor([batch_size, model.adm_channels], name="y", is_input=True)

# print(unet)

Y = unet(
    latent_model_input_ait,
    timesteps_ait,
    text_embeddings_pt_ait,
    y_ait
)

mark_output(Y)

# print(Y)

target = detect_target(
    use_fp16_acc=True, convert_conv_to_gemm=True
)

# print(target)

#print(params_ait.keys())
model_name = "unet_1"
compile_model(Y, target, "./optimize/{}".format(unet_name), model_name)
