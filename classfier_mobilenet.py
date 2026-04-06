from IPython.core.display import Image
from torchvision.io import read_image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
# from torchvision.models import inception_v3, Inception_V3_Weights
# from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
# from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights
# from torchvision.models import vit_l_16, ViT_L_16_Weights
import torchvision
import torch
from consts import orig_clases
# weights = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
# model = vit_l_16(weights=weights)
# weights = EfficientNet_B4_Weights.IMAGENET1K_V1
# model = efficientnet_b4(weights=weights)
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights)
model = model.eval().cuda()


preprocess = weights.transforms()
#input image w x h x c

def vit_predict(image):
    with torch.no_grad():
        prediction = predict_raw(image)
        if prediction.shape[0] == 1:
            prediction = prediction.squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = weights.meta["categories"][class_id]
            # return(f"class id - {class_id} {category_name}: {100 * score:.1f}%")
            return(f"{category_name}: {100 * score:.3f}%")
        res_lst = []
        for p in prediction:
            p = p.squeeze(0).softmax(0)
            class_id = p.argmax().item()
            score = p[class_id].item()
            category_name = weights.meta["categories"][class_id]
            res_lst.append(category_name)
        return res_lst


def predict_raw(image):
    # Apply inference preprocessing transforms
    batch = preprocess(image)

    # Get model output
    output = model(batch)
    
    # Handle Inception V3's auxiliary output during training mode
    if isinstance(output, tuple):
        output = output[0]
    
    return output.softmax(-1)


batch_size = 1
# orig_clases = torch.tensor([817, 705, 609, 586, 436, 627, 468, 621, 803, 407, 408, 751, 717,866, 661]).cuda()
total_clases_without_orig = torch.tensor([x for x in list(range(0, 1000)) if x not in orig_clases]).cuda()


# def adv_loss_calc(image):
#     adv_loss = 0
#     pred = resnet_predict_raw(image)
#     for p in pred:
#       adv_loss += torch.stack([100*p.softmax(0)[c.item()] for c in orig_clases]).mean() / pred.shape[0]
#     return adv_loss

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
