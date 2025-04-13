import torch
from torch.utils.data import DataLoader
import os


from src.bbox_utils import generate_pseudo_masks, generate_bboxs, generate_cams, evaluate, visualise_results, save_pseudo_masks
from src.dataset import load_data_wrapper, PseudoMaskDataset
from src.utils import load_device, clear_cuda_cache, parse_args
from src.bbox_utils import show_grabcut_masks


########################
# Experiments to include: 
# - Sans le grab cut, juste bounding boxes (celui qui performe moins bien)
# - Weak segmentation avec bounding boxes (le basique)
# - mix grabcut et super pixel (le meilleur)
# - data augmentation? 


# Expeirment 1: 
#data_dir = "./dummy_masks_thres_0.4/"

# Expeirment 2: 
# Look at functions in bbox_utils. mix_pseudo_masks_exp?
#######################
def main():
    return None

if __name__ == "__main__":
    main()



