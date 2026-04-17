import torch
from consts import orig_clases
import torch
from torchvision import transforms

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_lc").cuda().eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
del transform.transforms[0]

def predict_raw(image):
    # Apply inference preprocessing transforms
    batch = transform(image)

    # DINOv2 returns classification logits directly
    logits = model(batch)

    return logits.softmax(-1)


batch_size = 1
total_clases_without_orig = torch.tensor([x for x in list(range(0, 1000)) if x not in orig_clases]).cuda()


def adv_loss_calc(image):
    assert len(image.shape) == 4, "Image should be of shape (batch_size, 3, h, w)"
    adv_loss = []
    pred = predict_raw(image)
    for p in pred:
        adv_loss.append(p[orig_clases].mean())
    return torch.stack(adv_loss)



def adv_loss_calc2(image):
    adv_loss = []
    pred = predict_raw(image)
    for p in pred:
        forbiden = p[orig_clases].max()
        allowed = p[total_clases_without_orig].max()
        adv_loss.append(forbiden - allowed)
    return torch.stack(adv_loss)