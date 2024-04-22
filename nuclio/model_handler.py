import os
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
import tqdm
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.ops as vision_ops
from ops.foundation_models.segment_anything.utils.amg import batched_mask_to_box
from ops.ops import _nms, plot_results, convert_to_cuda
from models import ROIHeadMLP as ROIHead
from ops.foundation_models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
import io
import json
import base64
from ops.dump_clip_features import dump_clip_image_features, dump_clip_text_features
from models import PointDecoder


#1 sam 
# sam = build_sam_vit_h().eval()


#2 
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# with torch.no_grad():
#     new_image = transform(image).unsqueeze(0)
#     features = sam.image_encoder(new_image)

# #3 point decode 
# from models import PointDecoder
# point_decoder = PointDecoder(sam).eval()
# state_dict = torch.load(f'checkpoints/point_decoder_vith.pth',map_location='cpu')
# point_decoder.load_state_dict(state_dict)
# with torch.no_grad():
#     point_decoder.max_points = 500
#     point_decoder.point_threshold = 0.7
#     point_decoder.nms_kernel_size = 3
#     outputs_heatmaps = point_decoder(features)
#     pred_heatmaps = outputs_heatmaps['pred_heatmaps'].cpu().squeeze().clamp(0, 1)


# #4 show point 
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# pred_points = outputs_heatmaps['pred_points'].squeeze().reshape(-1, 2)
# pred_points_score = outputs_heatmaps['pred_points_score'].squeeze()
# print(pred_points.size())
# plt.scatter(pred_points[:, 0].cpu(), pred_points[:, 1].cpu(), s=20, marker='.', c='lime')
# plt.axis('off')
# # # plt.show()
# # plt.savefig('/home/zzhuang/tmp.jpg', bbox_inches='tight', pad_inches = 0,dpi=400)


# def read_image(path):
#     img = Image.open(path)
#     transform = A.Compose([
#         A.LongestMaxSize(1024),
#         A.PadIfNeeded(1024, border_mode=0, position=A.PadIfNeeded.PositionType.TOP_LEFT),
#     ])
#     img = Image.fromarray(transform(image=np.array(img))['image'])
#     return img


class  ModelHandler:    
   def __init__(context):
    context.logger.info('Init context...  0%')
    model_path = r'D:\demo\cvat\serverless\pseco\nuclio\sam_vit_h_4b8939.pth'
    sam = torch.load(model_path).eval()

    cls_head = ROIHead().eval()
    cls_head.load_state_dict(torch.load(f'D:\demo\cvat\serverless\pseco\nuclio\PseCoMain\checkpoints\MLP_small_box_w1_fewshot.tar', map_location='cpu')['cls_head'])

    point_decoder = PointDecoder(sam).eval()
    state_dict = torch.load(f'D:\demo\cvat\serverless\pseco\nuclio\PseCoMain\checkpoints\point_decoder_vith.pth',map_location='cpu')
    point_decoder.load_state_dict(state_dict)
    
    context.user_data.cls_head = cls_head
    context.user_data.sam = sam
    context.user_data.point_decoder = point_decoder
    context.logger.info('Init context...100%')

#    def handle(context, event, image):
#     context.logger.info('Run custom Pseco model')
#     data = event.body
#     buf = io.BytesIO(base64.b64decode(data["image"]))
#     image = Image.open(buf).convert('RGB')
    
#     w, h = image.size
#     bbox_sample = np.asarray(data['bbox_sample'], dtype=int)
#     ratio = max(w,h)/1024
#     norm_bbox_sample = [[round(bbox[0]/ratio), round(bbox[1]/ratio), round(bbox[2]/ratio), round(bbox[3]/ratio),] for bbox in bbox_sample]
#     example_boxes = torch.tensor(norm_bbox_sample)
#     example_features = dump_clip_image_features(image, example_boxes).cuda()
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     with torch.no_grad():
#         new_image = transform(image).unsqueeze(0)
#         features = context.user_data.sam.image_encoder(new_image)
#     with torch.no_grad():
#         context.user_data.point_decoder.max_points = 1000
#         context.user_data.point_decoder.point_threshold = 0.05
#         context.user_data.point_decoder.nms_kernel_size = 3
#         outputs_heatmaps = context.user_data.point_decoder(features)
#         pred_heatmaps = outputs_heatmaps['pred_heatmaps'].cpu().squeeze().clamp(0, 1)
#     pred_points = outputs_heatmaps['pred_points'].squeeze().reshape(-1, 2)
#     pred_points_score = outputs_heatmaps['pred_points_score'].squeeze()
#     _ = context.user_data.cls_head.eval()
#     with torch.no_grad():
#         all_pred_boxes = []
#         all_pred_ious = []
#         cls_outs = []
#         for indices in torch.arange(len(pred_points)).split(128):
#             with torch.no_grad():
#                 outputs_points = context.user_data.sam.forward_sam_with_embeddings(features, points=pred_points[indices])
#                 pred_boxes = outputs_points['pred_boxes']
#                 pred_logits = outputs_points['pred_ious']

#                 for anchor_size in [8, ]:
#                     anchor = torch.Tensor([[-anchor_size, -anchor_size, anchor_size, anchor_size]])
#                     anchor_boxes = pred_points[indices].repeat(1, 2) + anchor
#                     anchor_boxes = anchor_boxes.clamp(0., 1024.)
#                     outputs_boxes = context.user_data.sam.forward_sam_with_embeddings(features, points=pred_points[indices], boxes=anchor_boxes)
#                     pred_logits = torch.cat([pred_logits, outputs_boxes['pred_ious'][:, 1].unsqueeze(1)], dim=1)
#                     pred_boxes = torch.cat([pred_boxes, outputs_boxes['pred_boxes'][:, 1].unsqueeze(1)], dim=1)

#                 all_pred_boxes.append(pred_boxes)
#                 all_pred_ious.append(pred_logits)
#                 cls_outs_ = context.user_data.cls_head(features, [pred_boxes, ], [example_features, ] * len(indices))
#                 cls_outs_ = cls_outs_.sigmoid().view(-1, len(example_features), 5).mean(1)
#                 pred_logits = cls_outs_ * pred_logits
#             cls_outs.append(pred_logits)
#         pred_boxes = torch.cat(all_pred_boxes)
#         pred_ious = torch.cat(all_pred_ious)
#         cls_outs = torch.cat(cls_outs)
#         pred_boxes = pred_boxes[torch.arange(len(pred_boxes)), torch.argmax(cls_outs, dim=1)]
#         scores = cls_outs.max(1).values
#         indices = vision_ops.nms(pred_boxes, scores, 0.5)
#         pred_boxes = pred_boxes[indices]
#         scores = scores[indices]
#     pred_boxes = pred_boxes.cpu().numpy()
#     result_pred_boxes = [pred_boxes[i] for i, score in enumerate(scores) if score >= data.get("threshold", 0.5)]
#     denorm_result_pred_boxes = [[round(bbox[0]*ratio), round(bbox[1]*ratio), round(bbox[2]*ratio), round(bbox[3]*ratio),] for bbox in result_pred_boxes]
#     scores = [score for score in scores if score >= data.get("threshold", 0.5)]

#     detections = []
#     for bbox, score in zip(denorm_result_pred_boxes, scores):
#         detections.append({
#             'confidence': str(float(score)),
#             'points': bbox.tolist(),
#             'type': 'rectangle',
#         })

#     return context.Response(body=json.dumps(detections), headers={},
#                             content_type='application/json', status_code=200)
