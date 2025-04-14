import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.dataset import inverse_normalize
from src.cam_utils import cam_to_binary_mask, get_cam_generator


def create_overlay(img, gt_mask, pred_mask, alpha=0.5):
    img_pil = Image.fromarray((img * 255).astype(np.uint8)).convert('RGBA')
    
    gt_layer = Image.new('RGBA', img_pil.size, (0, 255, 0, int(255*alpha)))
    pred_layer = Image.new('RGBA', img_pil.size, (255, 0, 0, int(255*alpha)))
    
    gt_mask = Image.fromarray((gt_mask * 255).astype(np.uint8))
    pred_mask = Image.fromarray((pred_mask * 255).astype(np.uint8))
    
    img_pil = Image.composite(gt_layer, img_pil, gt_mask)
    img_pil = Image.composite(pred_layer, img_pil, pred_mask)
    
    return img_pil.convert('RGB')


def visualise_predictions(config, dataloader, model, n_samples=5, threshold=0.5, device="cuda"):
    model.eval()
    all_samples = []
    
    with torch.no_grad():
        for images, gt_masks, info in dataloader:
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            batch_size = images.size(0)
            for i in range(batch_size):
                all_samples.append((images[i], gt_masks[i], info["name"][i]))
    
    random.shuffle(all_samples)
    selected_samples = all_samples[:n_samples]
    
    tile_size = 256 
    composite = Image.new('RGB', (4*tile_size, n_samples*tile_size + 50*n_samples), 'white')
    draw = ImageDraw.Draw(composite)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for row, (img_tensor, gt_mask, img_name) in enumerate(selected_samples):
        img = inverse_normalize(img_tensor).cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        outputs = model(img_tensor.unsqueeze(0))
        pred_mask = (outputs > threshold).float()[0].cpu().squeeze().numpy()
        gt_mask_np = gt_mask.cpu().squeeze().numpy()
        
        orig_img = Image.fromarray((img * 255).astype(np.uint8))
        gt_img = Image.fromarray((gt_mask_np * 255).astype(np.uint8))
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        overlay_img = create_overlay(img, gt_mask_np, pred_mask)
        
        y_offset = row * (tile_size + 50)
        for col, (image, title) in enumerate(zip(
            [orig_img, gt_img, pred_img, overlay_img],
            [f"Original: {img_name}", "GT Mask", "Pred Mask", "Overlay"]
        )):
            x_offset = col * tile_size
            composite.paste(image.resize((tile_size, tile_size)), 
                           (x_offset, y_offset))
            draw.text((x_offset + 10, y_offset + tile_size + 10), 
                     title, font=font, fill='black')

    save_path = config["segmentation_visualisation_save_path"]
    composite.save(save_path)
    print(f"[Saved predictions to {save_path}]")


def visualise_cams(config, dataloader, classifier, cam_type='CAM', cam_threshold=0.5, n_samples=5, device="cuda"):
    classifier.eval()
    
    target_id = config["target_id"]
    all_samples = []
    
    with torch.no_grad():
        for images, _, info in dataloader:
            images = images.to(device)
            for i in range(images.size(0)):
                all_samples.append((images[i], info["name"][i], info[target_id][i]))
    
    random.shuffle(all_samples)
    selected_samples = all_samples[:n_samples]
    
    composite = Image.new('RGB', (3*256, n_samples*256 + 50*n_samples), 'white')
    draw = ImageDraw.Draw(composite)
    
    cam_generator, generate_cam_func, cam_type = get_cam_generator(classifier, cam_type)
    
    for row, (img_tensor, img_name, img_target) in enumerate(selected_samples):
        if cam_type == 'CAM':
            with torch.no_grad():
                cam, pred_class = generate_cam_func(cam_generator, img_tensor)
        else:
            with torch.enable_grad():
                cam, pred_class = generate_cam_func(cam_generator, img_tensor)
        
        img = inverse_normalize(img_tensor).cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam.cpu().numpy()), cv2.COLORMAP_JET)
        overlayed_img = cv2.addWeighted(np.uint8(255 * img), 0.5, heatmap, 0.5, 0)
        pseudo_mask = cam_to_binary_mask(cam, cam_threshold).squeeze().cpu().numpy()
        
        orig_img = Image.fromarray((img * 255).astype(np.uint8))
        cam_img = Image.fromarray(overlayed_img)
        mask_img = Image.fromarray((pseudo_mask * 255).astype(np.uint8))
        
        y_offset = row * (256 + 50)
        for col, (image, title) in enumerate(zip(
            [orig_img, cam_img, mask_img],
            [f"{img_name} (Class {img_target+1})", 
             f"Pred Class {pred_class+1}", 
             f"Mask @ {cam_threshold}"]
        )):
            x_offset = col * 256
            composite.paste(image.resize((256, 256)), (x_offset, y_offset))
            draw.text((x_offset + 10, y_offset + 256 + 10), 
                     title, font=font, fill='black')

    save_path = config["cam_visualisation_save_path"]
    composite.save(save_path)
    print(f"[Saved CAM visualisations to {save_path}]")
