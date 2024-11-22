################## 1. Download checkpoints and build models
from tqdm import tqdm
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var_ctf
import argparse

def create_npz_from_sample_folder(sample_folder: str):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """
    import os, glob
    import numpy as np
    from tqdm import tqdm
    from PIL import Image
    
    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    assert len(pngs) == 50_000, f'{len(pngs)} png files found in {sample_folder}, but expected 50,000'
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)'):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (50_000, samples.shape[1], samples.shape[2], 3)
    npz_path = f'{sample_folder}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path



parser = argparse.ArgumentParser()
parser.add_argument("--training_free", action="store_true")
parser.add_argument("--drafter_depth", type=int, default=30)
parser.add_argument("--refiner_depth", type=int, default=16)
parser.add_argument("--draft_steps", type=int, default=8)
parser.add_argument("--cfg", type=int, default=1.5)
parser.add_argument("--output_path", type=str, default="images")
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
    hf_home = 'https://huggingface.co/Zigeng/VAR_CoDe/resolve/main'
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
torch.manual_seed(seed)
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


print("#####################sampling begin!#####################")

# sample
newk = [600]*10
temp = [1.1]*7+[1.0]*3

for i in range(1000):
     print("class index:",i, "draft_steps:", draft_steps)
     class_labels = (i,)
     for j in tqdm(range(50)):
        with torch.inference_mode():
            B = len(class_labels)
            label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                
                f_hat, token_hub= var_draft.autoregressive_infer_cfg_draft(B=B, label_B=label_B, cfg=1.5, top_k=newk, top_p=0.96, g_seed=j, more_smooth=False,exit_num=draft_steps,temp=temp)
                recon_B3HW = var_refine.autoregressive_infer_cfg_refine(B=B, label_B=label_B, cfg=1.5, top_k=newk, top_p=0.96, g_seed=j, more_smooth=False, 
                                                                    draft=token_hub, f_hat=f_hat, entry_num=draft_steps, temp=temp)

            chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
            chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
            chw = PImage.fromarray(chw.astype(np.uint8))
            chw.save(args.output_path+"/"+str(i)+"_"+str(j)+".PNG")

create_npz_from_sample_folder("args.output_path")
print("#####################sampling completed!#####################")












