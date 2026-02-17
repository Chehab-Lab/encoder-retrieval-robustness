import os
import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import List
from datasets import get_dataset
from encoders import get_encoder, get_features
from metric import retrieval_mean_precision, RetrievalMetrics, mean_average_precision
from transformations import get_transformation

def exists(path):
    return os.path.exists(path)

def _is_already_evaluated(checkpoint_file_path, encoder_name, dataset_name):
    if exists(checkpoint_file_path):
        checkpoints = json.load(open(checkpoint_file_path))
        for checkpoint in checkpoints:
            if checkpoint['encoder'] == encoder_name and checkpoint['dataset'] == dataset_name:
                return True
    return False

def _apply_transform(image, transformation):
    img = np.asarray(image)
    img = img / 255.0
    augmented_img = (np.array(transformation([img])) * 255).astype(np.uint8)
    return augmented_img

def _get_embeddings(encoder, images, transformation, img_processor, target_dim, batch_size, device):
    embeddings = []
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i:i+batch_size]
        if transformation:
                batch_images = [_apply_transform(image, transformation) for image in batch_images]
        batch_images = img_processor(batch_images, return_tensors="pt")["pixel_values"].to(device)
        batch_emb = get_features(encoder, batch_images, target_dim, device)
        embeddings.append(batch_emb)
    embeddings = torch.cat(embeddings)
    embeddings = embeddings.cpu().numpy()
    return embeddings

def evaluate_retrieval(encoder_name: str, 
                       dataset_name: str, 
                       target_dim: int,
                       transformation_objs: List,
                       metrics = [RetrievalMetrics.MEAN_AVERAGE_PRECISION],
                       k_list: List[int] = [9],
                       batch_size: int = 64,
                       device: str = "cuda",
                       checkpoint_folder: str = "./checkpoints", 
                       checkpoint_name: str = "results",
                       verbose: bool = True):

    if verbose: print("\nVerifying checkpoints....") 
    if not exists(checkpoint_folder): os.mkdir(checkpoint_folder)
        
    checkpoint_file = os.path.join(checkpoint_folder, checkpoint_name+".json")
    if _is_already_evaluated(checkpoint_file, encoder_name, dataset_name):
        print(f"{encoder_name} already evaluated on {dataset_name}. Skipping evaluation")
        return

    encoder, img_processor = get_encoder(encoder_name, device=device)
    dataset = get_dataset(dataset_name)

    if verbose: print(f"\nExtracting data ...")
    images = []
    labels = []
    for image, label in tqdm(dataset):
        images.append(image)
        labels.append(label)
    labels = np.array(labels)
    
    if verbose: print(f"\nGetting clean image embeddings....")

    clean_emb = _get_embeddings(encoder, images, None, img_processor, target_dim, batch_size, device)

    if verbose: print("\n Evaluating embeddings....")
    mAP_results = []
    if RetrievalMetrics.MEAN_AVERAGE_PRECISION in metrics:
        if verbose: print("\n Computing mAP@k....")
        for k in k_list:
            if transformation_objs:
                for transformation in transformation_objs:
                    transform = get_transformation(transformation)
                    name = transformation['id']
                    if verbose: print(f"\nGetting {name} transformed image embeddings....")
                    augmented_emb = _get_embeddings(encoder, images, transform, img_processor, target_dim, batch_size, device)
                    result = mean_average_precision(augmented_emb, labels, clean_emb, labels, k)
                    mAP_results.append({'transformation_name': name, f'mAP@{k}': result})
            else:
                result = mean_average_precision(clean_emb, labels, clean_emb, labels, k)
                mAP_results.append({'transformation_name': None, f'mAP@{k}': result})

    mean_precision=[]
    if RetrievalMetrics.MEAN_PRECISION in metrics:
        if verbose: print("\n computing MP@k....")
        for k in k_list:
            if transformation_objs:
                for transformation in transformation_objs:
                    transform = get_transformation(transformation)
                    name = transformation['id']
                    if verbose: print(f"\nGetting {name} transformed image embeddings....")
                    augmented_emb = _get_embeddings(encoder, images, transform, img_processor, target_dim, batch_size, device)
                    result = retrieval_mean_precision(augmented_emb, labels, k)
                    mean_precision.append({'transformation_name': name, f'MP@{k}': result})
            else:
                result = retrieval_mean_precision(clean_emb, labels, k)
                mean_precision.append({'transformation_name': name, f'MP@{k}': result})                

        
    if verbose: print("\nSaving checkpoint....")
    
    results = {
        'encoder': encoder_name,
        'dataset': dataset_name,
        'metrics': {
            'mAP': mAP_results,
            'mean_precision': mean_precision
        }
        }
    
    checkpoint = json.load(open(checkpoint_file)) if exists(checkpoint_file) else []
         
    checkpoint.append(results)
    
    json.dump(checkpoint, open(checkpoint_file, 'w'), ensure_ascii=True, indent=4)

def _test_retreival_pipeline():
    encoder_name = "microsoft/resnet-50"
    dataset_name = "gpr1200"
    transformation_obj = [{
            "id": "motionblur",
            "kernelsize": 14,
            "angle": 45,
            "direction": 1
        }]
    evaluate_retrieval(encoder_name, dataset_name, 2048, transformation_obj, [RetrievalMetrics.MEAN_AVERAGE_PRECISION],[5])