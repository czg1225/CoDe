import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var_ctf
import matplotlib.pyplot as plt
import gc
from contextlib import contextmanager
import argparse


@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')



parser = argparse.ArgumentParser()
parser.add_argument("--drafter_depth", type=int, default=30)
parser.add_argument("--refiner_depth", type=int, default=16)
parser.add_argument("--draft_steps", type=int, default=8)
parser.add_argument("--cfg", type=int, default=4)
parser.add_argument("--training_free", action="store_true")  
args = parser.parse_args()


MODEL_DEPTH_draft = args.drafter_depth     
MODEL_DEPTH_refine = args.refiner_depth  
draft_steps = args.draft_steps  
assert MODEL_DEPTH_draft in {16, 20, 24, 30}
assert MODEL_DEPTH_refine in {16, 20, 24, 30}
assert draft_steps in {1, 2, 3, 4, 5, 6, 7, 8, 9}


################## 1. Download checkpoints and build models

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var_draft, var_refine = build_vae_var_ctf(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth_draft=MODEL_DEPTH_draft, depth_refine=MODEL_DEPTH_refine, shared_aln=False,
    )

# load vae checkpoints
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt = 'vae_ch160v4096z32.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

# load var checkpoints
print("start loading checkpoints.........")

if args.training_free:
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    var_draft_ckpt, var_refine_ckpt = f'var_d{MODEL_DEPTH_draft}.pth', f'var_d{MODEL_DEPTH_refine}.pth'
    if not osp.exists(var_draft_ckpt): os.system(f'wget {hf_home}/{var_draft_ckpt}')
    if not osp.exists(var_refine_ckpt): os.system(f'wget {hf_home}/{var_refine_ckpt}')
    var_draft.load_state_dict(torch.load(var_draft_ckpt, map_location='cpu'), strict=True)
    var_refine.load_state_dict(torch.load(var_refine_ckpt, map_location='cpu'), strict=True)
else:
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    var_draft_ckpt, var_refine_ckpt = f'drafter_{draft_steps}.pth', f'refiner_{draft_steps}.pth'
    if not osp.exists(var_draft_ckpt): os.system(f'wget {hf_home}/{var_draft_ckpt}')
    if not osp.exists(var_refine_ckpt): os.system(f'wget {hf_home}/{var_refine_ckpt}')
    var_draft.load_state_dict(torch.load(var_draft_ckpt, map_location='cpu'),strict=True)
    var_refine.load_state_dict(torch.load(var_refine_ckpt, map_location='cpu'),strict=True)

print("loading drafter from:",var_draft_ckpt, "loading refiner from:",var_refine_ckpt)

vae.eval(), var_draft.eval(), var_refine.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var_draft.parameters(): p.requires_grad_(False)
for p in var_refine.parameters(): p.requires_grad_(False)
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
newk = [600]*10
temp = [1.1]*7+[1.0]*3

with torch.inference_mode():
    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    with measure_peak_memory():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            for i in range(3):
                start_event.record()
                # drafting stage
                f_hat, token_hub= var_draft.autoregressive_infer_cfg_draft(B=B, label_B=label_B, cfg=cfg, top_k=newk, top_p=0.95, g_seed=seed, more_smooth=more_smooth,exit_num=draft_steps, temp=temp)
                # refining stage
                recon_B3HW = var_refine.autoregressive_infer_cfg_refine(B=B, label_B=label_B, cfg=cfg, top_k=newk, top_p=0.95, g_seed=seed, more_smooth=more_smooth, 
                                                                        draft=token_hub, f_hat=f_hat, entry_num=draft_steps, temp=temp)     
                end_event.record()
                torch.cuda.synchronize()
                # Calculation run time (milliseconds)
                elapsed_time = start_event.elapsed_time(end_event)
                print("running time:",int(elapsed_time),"ms", "batch size:",str(len(class_labels)))

    # save the generated images
    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))
    chw.save("output_code.png")
    print("generate images are saved as --output_code.png-- ")