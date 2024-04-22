import json
import base64
from PIL import Image
import io
import torch
import cv2
import numpy as np
from model_handler import ModelHandler
from ops.dump_clip_features import dump_clip_image_features, dump_clip_text_features
import torchvision.transforms as transforms
from models import PointDecoder
import torchvision.ops as vision_ops
import matplotlib.pyplot as plt





def init_context(context):
    context.logger.info("Init context...  0%")

    model = ModelHandler()
	
    context.user_data.model1 = model.cls_head
    context.user_data.model2 = model.sam
    context.user_data.model3 = model.point_decoder

    context.logger.info("Init context...100%")

         
def handler(context, event):
    
    context.logger.info('Run custom pseco model')
    data = event.body
    image_buffer = io.BytesIO(base64.b64decode(data['image']))
    image = cv2.imdecode(np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    results = context.user_data.model_handler(image)
    result = results[0]
    
    # few-shot
    # 1
    example_boxes = torch.Tensor([[866.00, 378.00, 1024.00, 508.00],
                                   [515.00, 342.00, 611.00, 442.00],
                                   [548.00, 445.00, 704.00, 635.00],
                                   [932.0, 575.0, 1024.0, 679.0],
                                   [194., 494., 378., 681.]
                                  ])

    example_features = dump_clip_image_features(image, example_boxes)
    from models import ROIHeadMLP as ROIHead
    cls_head = ROIHead().eval()
    #can xem  lai (da xong)
    cls_head.load_state_dict(context.user_data.model1, map_location='cpu')['cls_head']

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    with torch.no_grad():
         new_image = transform(image).unsqueeze(0)
         features = context.user_data.model2.image_encoder(new_image)

    #3_can xem lai (da-xong) 
    point_decoder = PointDecoder(context.user_data.model2).eval()
    point_decoder.load_state_dict(context.user_data.model3)
    
    with torch.no_grad():
         point_decoder.max_points = 500
         point_decoder.point_threshold = 0.7
         point_decoder.nms_kernel_size = 3
         outputs_heatmaps = point_decoder(features)
         pred_heatmaps = outputs_heatmaps['pred_heatmaps'].cpu().squeeze().clamp(0, 1)
    
    #4

    pred_points = outputs_heatmaps['pred_points'].squeeze().reshape(-1, 2)
    pred_points_score = outputs_heatmaps['pred_points_score'].squeeze()

    
    #5
    _ = cls_head.eval()

    with torch.no_grad():
         all_pred_boxes = []
         all_pred_ious = []
         cls_outs = []
         for indices in torch.arange(len(pred_points)).split(128):
             with torch.no_grad():
                  outputs_points = context.user_data.model2.forward_sam_with_embeddings(features, points=pred_points[indices])
                  pred_boxes = outputs_points['pred_boxes']
                  pred_logits = outputs_points['pred_ious']

                  for anchor_size in [8, ]:
                      anchor = torch.Tensor([[-anchor_size, -anchor_size, anchor_size, anchor_size]])
                      anchor_boxes = pred_points[indices].repeat(1, 2) + anchor
                      anchor_boxes = anchor_boxes.clamp(0., 1024.)
                      outputs_boxes = context.user_data.model2.forward_sam_with_embeddings(features, points=pred_points[indices], boxes=anchor_boxes)
                      pred_logits = torch.cat([pred_logits, outputs_boxes['pred_ious'][:, 1].unsqueeze(1)], dim=1)
                      pred_boxes = torch.cat([pred_boxes, outputs_boxes['pred_boxes'][:, 1].unsqueeze(1)], dim=1)

                  all_pred_boxes.append(pred_boxes)
                  all_pred_ious.append(pred_logits)
                  cls_outs_ = cls_head(features, [pred_boxes, ], [example_features, ] * len(indices))
                  cls_outs_ = cls_outs_.sigmoid().view(-1, len(example_features), 5).mean(1)
                  pred_logits = cls_outs_ * pred_logits
         cls_outs.append(pred_logits)
    pred_boxes = torch.cat(all_pred_boxes)
    pred_ious = torch.cat(all_pred_ious)
    cls_outs = torch.cat(cls_outs)
    pred_boxes = pred_boxes[torch.arange(len(pred_boxes)), torch.argmax(cls_outs, dim=1)]
    scores = cls_outs.max(1).values
    indices = vision_ops.nms(pred_boxes, scores, 0.5)
    pred_boxes = pred_boxes[indices]
    scores = scores[indices]
    COLORS = [[0.000, 0.447, 0.741], ]
    for p, (xmin, ymin, xmax, ymax), c in zip(scores, pred_boxes.tolist(), COLORS * len(pred_boxes)):
       if p < 0.1:
         continue
       plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color=c,
                               linewidth=2.0,
                               ))
       text = f'{int(p*100)}%'
       plt.gca().text(xmin, ymin, text,
            fontsize=5,
            bbox=dict(facecolor='yellow',alpha=0.5)
            )




    boxes = result.boxes.data[:, :4]
    confs = result.boxes.conf
    clss = result.boxes.cls
    # class_name = result.names

    detections = []
    threshold = 0.1
    for box, conf, cls in zip(boxes, confs, clss):

       if conf >= threshold:
        # must be in this format
        detections.append({
            'confidence': str(float(conf)),
            'points': box.tolist(),
            'type': 'rectangle',
        })

    return context.Response(body=json.dumps(detections), headers={},
                        content_type='application/json', status_code=200)
