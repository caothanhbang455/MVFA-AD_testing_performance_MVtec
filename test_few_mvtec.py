import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from dataset.mvtec_few import MVTECDataset, load_mvtec, mvtec_classes
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot-mvtec')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)
    
    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    checkpoint = torch.load(os.path.join(f'{args.save_path}', f'{args.obj}.pth'))
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])

    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))



    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MVTECDataset(load_mvtec, args.obj, args.shot, experiment_indx=0, resize=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    # few-shot image augmentation
    # augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = augment_normal_img
    augment_fewshot_mask = augment_normal_mask
    
    augment_fewshot_label = torch.Tensor([0] * len(augment_normal_img))

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)



    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, args.obj, device)

    best_result = 0

    seg_features = []
    det_features = []
    for image in support_loader:
        image = image[0].to(device)
        with torch.no_grad():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
            det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
            seg_features.append(seg_patch_tokens)
            det_features.append(det_patch_tokens)
    seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
    det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
    

    result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)



def testing(model_name='ViT-L-14-336', pretrain = 'openai', obj = 'bottle',
            batch_size = 1, save_model = 1, save_path = './ckpt/few-shot-mvtec', 
            img_size = 240, epoch = 50, learning_rate = 0.001, features_list = [6, 12, 18, 24],
            seed = 111, shot = 5, iterate = 0):
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot-mvtec')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)
    
    # fixed feature extractor
    clip_model = create_model(model_name=model_name, img_size=img_size, device=device, pretrained=pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=features_list).to(device)
    model.eval()

    checkpoint = torch.load(os.path.join(f'{save_path}', f'{obj}.pth'))
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])

    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=learning_rate, betas=(0.5, 0.999))



    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MVTECDataset(load_mvtec, obj, shot, experiment_indx=0, resize=img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


    # few-shot image augmentation
    # augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = augment_normal_img
    augment_fewshot_mask = augment_normal_mask
    
    augment_fewshot_label = torch.Tensor([0] * len(augment_normal_img))

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)



    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, obj, device)

    best_result = 0

    seg_features = []
    det_features = []
    for image in support_loader:
        image = image[0].to(device)
        with torch.no_grad():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
            det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
            seg_features.append(seg_patch_tokens)
            det_features.append(det_patch_tokens)
    seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
    det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
    

    result = test(obj, img_size, model, test_loader, text_features, seg_mem_features, det_mem_features)
    return result


def test(obj, img_size, model, test_loader, text_features, seg_mem_features, det_mem_features):
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few= []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]


            # few-shot, seg head
            anomaly_maps_few_shot = []
            for idx, p in enumerate(seg_patch_tokens):
                cos = cos_sim(seg_mem_features[idx], p)
                height = int(np.sqrt(cos.shape[1]))
                anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                        size=img_size, mode='bilinear', align_corners=True)
                anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
            score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
            seg_score_map_few.append(score_map_few)

            # zero-shot, seg head
            anomaly_maps = []
            for layer in range(len(seg_patch_tokens)):
                seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            score_map_zero = np.sum(anomaly_maps, axis=0)
            seg_score_map_zero.append(score_map_zero)
                


            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)



    seg_score_map_zero = np.array(seg_score_map_zero)
    seg_score_map_few = np.array(seg_score_map_few)

    seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
    seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())

    segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
    seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
    print(f'{obj} pAUC : {round(seg_roc_auc,4)}')

    segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
    roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
    print(f'{obj} AUC : {round(roc_auc_im, 4)}')

    return seg_roc_auc, roc_auc_im

    



if __name__ == '__main__':
    stats = []
    for obj in mvtec_classes:
        result = testing(obj=obj)
        stats.append({
            'class_name' : obj,
            'pAUROC' : result[0],
            'AUROC' : result[1]
        })


