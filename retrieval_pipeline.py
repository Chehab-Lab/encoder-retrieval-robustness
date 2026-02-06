import os
import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import List
from metrics import AltMetric
from datasets import get_dataset
from encoders import get_encoder
from metrics import retrieval_mean_precision

def exists(path):
    return os.path.exists(path)

def _is_already_evaluated(checkpoint_file_path, encoder_name, dataset_name):
    if exists(checkpoint_file_path):
        checkpoints = json.load(open(checkpoint_file_path))
        for checkpoint in checkpoints:
            if checkpoint['encoder'] == encoder_name and checkpoint['dataset'] == dataset_name:
                return True
    return False

def retrieve(encoder_name: str, 
             dataset_name: str, 
             metrics: List = [AltMetric.RETRIEVAL_MEAN_PRECISION], 
             k_list: List[int] = [9],
             device: str = "cuda",
             checkpoint_folder: str = "./checkpoints", 
             checkpoint_name: str = "results",
             verbose: bool = True):

    encoder, img_processor = get_encoder(encoder_name, device=device)
    dataset = get_dataset(dataset_name, None, img_processor)
    
    if verbose: print("Loading checkpoints....") 
    if not exists(checkpoint_folder): os.mkdir(checkpoint_folder)
        
    checkpoint_file = os.path.join(checkpoint_folder, checkpoint_name+".json")
    if _is_already_evaluated(checkpoint_file, encoder_name, dataset_name):
        print(f"{encoder_name} already evaluated on {dataset_name}. Skipping evaluation")
        return
    
        
    if verbose: print(f"Getting image embeddings....")
    embeddings = []
    labels = []
    encoder.eval()
    with torch.no_grad():
        for image, label in tqdm(dataset):
            image = image.to(device)
            embeddings = encoder(image)
            embeddings.append(embeddings.cpu().numpy())
            labels.append(label)
    embeddings, labels = faiss.normalize_L2(np.vstack(embeddings)), np.vstack(labels) #l2 normalized
            
    if verbose: print("Evaluating embeddings....")
    if AltMetric.MEAN_PRECISION in metrics:
        mean_precision=[]
        for k in k_list:
            mp = retrieval_mean_precision(embeddings.astype("float32"), labels, k)
            mean_precision.append({f"{k}": mp})
    else:
        mean_precision = []
        
    if verbose: print("Saving checkpoint....")
    
    results = {
        'encoder': encoder_name,
        'dataset': dataset_name,
        'metrics': {
            'mean_precision' : mean_precision
        }
    }
    
    checkpoint = json.load(open(checkpoint_file)) if exists(checkpoint_file) else []
         
    checkpoint.append(results)
    
    json.dump(checkpoint, open(checkpoint_file, 'w'), ensure_ascii=True, indent=4)
