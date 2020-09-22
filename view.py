import torch
from matplotlib import pyplot as plt
import numpy as np
from core.datasets import PennFudanDataset
from core.models.mask_rcnn import get_instance_segmentation_model
from core.utils.detection.train import get_transform
from PIL import Image


# pick one image from the test set
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_instance_segmentation_model(2)
model.load_state_dict(torch.load('./MaskRCNN.pth'))
model.to(device)
dataset_test = PennFudanDataset('./data/PennFudanPed', get_transform(train=False))

# %%
img, _ = dataset_test[0]

# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

# %%
image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

plt.imshow(image)
plt.savefig('./output/image.jpg')
predictions = prediction[0]['masks']
# print(len(predictions))
index = 0
for _prediction in predictions:
    pred = Image.fromarray(_prediction[0].mul(255).byte().cpu().numpy())
    plt.imshow(pred)
    print('./output/pred_%d.jpg'%index)
    plt.savefig('./output/pred_%d.jpg'%index)
    index += 1



# pred = Image.fromarray(prediction[0]['masks'][1, 0].mul(255).byte().cpu().numpy())
# plt.imshow(pred)
# plt.savefig('./output/pred_9999.jpg')
plt.show()

