import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.rpn import AnchorGenerator

# pretrained on the COCO set
# fastercnn_resnet50_fpn


model = fasterrcnn_resnet50_fpn_v2(
    pretrained=True
)  # , prgress=True, num_classes=91)# , pretrained_backbone=True,)
"""
https://pytorch.org/vision/0.11/models.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

input: [CHANNEL, HEIGHT, WIDTH] for each image
input during training: input tensors, targets with boxes and labels
output during training: Dict[Tensor] containing the calssification and regression losses 

INFERENCE: only input tensors, N for the number of detections
"""
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {"boxes": boxes[i], "labels": labels[i]}
targets.append(d)
output = model(images, targets)
