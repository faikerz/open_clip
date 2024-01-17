import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-256', pretrained='datacomp_s34b_b86k', jit=True)
model.eval()
# save model as torchscript
torch.jit.save(torch.jit.script(model), 'model.pt')

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model(image)
print(image_features.shape)