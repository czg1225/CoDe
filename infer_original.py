import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
import gc
from contextlib import contextmanager
import argparse

# memory consumption evaluation
@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')



parser = argparse.ArgumentParser()
parser.add_argument("--model_depth", type=int, default=30)
parser.add_argument("--cfg", type=int, default=4)
args = parser.parse_args()


############################# 1. Download checkpoints and build models

MODEL_DEPTH = args.model_depth    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}

# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)

vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

############################# 2. Sample with classifier-free guidance

# set args
seed = 42 #@param {type:"number"}
cfg = args.cfg #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = (992, 992, 483, 483, 970, 970, 609, 609, 
                978, 978, 985, 985, 963, 963, 949, 949,)
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# sample
with torch.inference_mode():
    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    with measure_peak_memory():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            for i in range(3):
                start_event.record()
                recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
                end_event.record()
                torch.cuda.synchronize()
                # Calculation run time (milliseconds)
                elapsed_time = start_event.elapsed_time(end_event)
                print("running time:",int(elapsed_time),"ms", "batch size:",str(len(class_labels)))

    # save the generated images
    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))
    chw.save("output_original.png")
    print("generate images are saved as ---output_original.png-- ")