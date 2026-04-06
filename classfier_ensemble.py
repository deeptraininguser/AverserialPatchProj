from IPython.core.display import Image
from torchvision.io import read_image
import torchvision
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    inception_v3, Inception_V3_Weights,
    resnet18, ResNet18_Weights,
    vgg16, VGG16_Weights,
    vit_b_16, ViT_B_16_Weights
)
from consts import orig_clases

# Initialize all models
print("Loading ensemble models...")
model = 'ensamble_classifier'
# 1. Inception V3
inception_weights = Inception_V3_Weights.IMAGENET1K_V1
inception_model = inception_v3(weights=inception_weights)
inception_model = inception_model.eval().cuda()
inception_preprocess = inception_weights.transforms()

# 2. ResNet18
resnet_weights = ResNet18_Weights.IMAGENET1K_V1
resnet_model = resnet18(weights=resnet_weights)
resnet_model = resnet_model.eval().cuda()
resnet_preprocess = resnet_weights.transforms()

# 3. VGG16
vgg_weights = VGG16_Weights.IMAGENET1K_V1
vgg_model = vgg16(weights=vgg_weights)
vgg_model = vgg_model.eval().cuda()
vgg_preprocess = vgg_weights.transforms()

# 4. ViT-B/16
vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
vit_model = vit_b_16(weights=vit_weights)
vit_model = vit_model.eval().cuda()
vit_preprocess = vit_weights.transforms()

# 5. DINOv2
dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_lc").cuda().eval()
dino_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# compile all models for speed
# print("Compiling models for speed...")
# inception_model = torch.compile(inception_model, mode="reduce-overhead")
# print('compiled inception')
# resnet_model = torch.compile(resnet_model, mode="reduce-overhead")
# print('compiled resnet')
# vgg_model = torch.compile(vgg_model, mode="reduce-overhead")
# print('compiled vgg')
# vit_model = torch.compile(vit_model, mode="reduce-overhead")
# print('compiled vit')
# dino_model = torch.compile(dino_model, mode="reduce-overhead")
# print('compiled dino')

# Use weights from one of the models for categories (they all use ImageNet)
weights = inception_weights

print("✓ All models loaded successfully!")
print(f"  - Inception V3")
print(f"  - ResNet18")
print(f"  - VGG16")
print(f"  - ViT-B/16")
print(f"  - DINOv2")


def predict_raw_balanced(image):
    """
    Ensemble prediction using all 5 models.
    
    Args:
        image: Tensor of shape (batch_size, 3, H, W) with values in [0, 1]
    
    Returns:
        Combined softmax probabilities averaged across all models
    """
    # 1. Inception V3 prediction
    inception_batch = inception_preprocess(image)
    inception_logits = inception_model(inception_batch)
    if isinstance(inception_logits, tuple):  # Inception V3 returns tuple during training
        inception_logits = inception_logits[0]
    inception_probs = F.softmax(inception_logits, dim=1)
    
    # 2. ResNet18 prediction
    resnet_batch = resnet_preprocess(image)
    resnet_logits = resnet_model(resnet_batch)
    resnet_probs = F.softmax(resnet_logits, dim=1)
    
    # 3. VGG16 prediction
    vgg_batch = vgg_preprocess(image)
    vgg_logits = vgg_model(vgg_batch)
    vgg_probs = F.softmax(vgg_logits, dim=1)
    
    # 4. ViT-B/16 prediction
    vit_batch = vit_preprocess(image)
    vit_logits = vit_model(vit_batch)
    vit_probs = F.softmax(vit_logits, dim=1)
    
    # 5. DINOv2 prediction
    dino_batch = dino_transform(image)
    dino_logits = dino_model(dino_batch)
    dino_probs = F.softmax(dino_logits, dim=1)
    
    # Ensemble: Average probabilities from all models
    # You can also use weighted average if some models perform better
    ensemble_probs = (inception_probs + resnet_probs + vgg_probs + vit_probs + dino_probs) / 5.0
    
    return ensemble_probs


def predict_raw(image, weights_dict=None):
    """
    Ensemble prediction with custom weights for each model.
    
    Args:
        image: Tensor of shape (batch_size, 3, H, W) with values in [0, 1]
        weights_dict: Dictionary with keys ['inception', 'resnet', 'vgg', 'vit', 'dino']
                     Default: equal weights (0.2 each)
    
    Returns:
        Weighted combination of softmax probabilities
    """
    if weights_dict is None:
        weights_dict = {
            'inception': 0.1,
            'resnet': 0.1,
            'vgg': 0.1,
            'vit': 0.1,
            'dino': 0.6
        }
    
    ensemble_probs = None
    
    # 1. Inception V3 prediction
    if weights_dict.get('inception', 0) > 0:
        inception_batch = inception_preprocess(image)
        inception_logits = inception_model(inception_batch)
        if isinstance(inception_logits, tuple):
            inception_logits = inception_logits[0]
        inception_probs = F.softmax(inception_logits, dim=1)
        if ensemble_probs is None:
            ensemble_probs = weights_dict['inception'] * inception_probs
        else:
            ensemble_probs += weights_dict['inception'] * inception_probs
    
    # 2. ResNet18 prediction
    if weights_dict.get('resnet', 0) > 0:
        resnet_batch = resnet_preprocess(image)
        resnet_logits = resnet_model(resnet_batch)
        resnet_probs = F.softmax(resnet_logits, dim=1)
        if ensemble_probs is None:
            ensemble_probs = weights_dict['resnet'] * resnet_probs
        else:
            ensemble_probs += weights_dict['resnet'] * resnet_probs
    
    # 3. VGG16 prediction
    if weights_dict.get('vgg', 0) > 0:
        vgg_batch = vgg_preprocess(image)
        vgg_logits = vgg_model(vgg_batch)
        vgg_probs = F.softmax(vgg_logits, dim=1)
        if ensemble_probs is None:
            ensemble_probs = weights_dict['vgg'] * vgg_probs
        else:
            ensemble_probs += weights_dict['vgg'] * vgg_probs
    
    # 4. ViT-B/16 prediction
    if weights_dict.get('vit', 0) > 0:
        vit_batch = vit_preprocess(image)
        vit_logits = vit_model(vit_batch)
        vit_probs = F.softmax(vit_logits, dim=1)
        if ensemble_probs is None:
            ensemble_probs = weights_dict['vit'] * vit_probs
        else:
            ensemble_probs += weights_dict['vit'] * vit_probs
    
    # 5. DINOv2 prediction
    if weights_dict.get('dino', 0) > 0:
        dino_batch = dino_transform(image)
        dino_logits = dino_model(dino_batch)
        dino_probs = F.softmax(dino_logits, dim=1)
        if ensemble_probs is None:
            ensemble_probs = weights_dict['dino'] * dino_probs
        else:
            ensemble_probs += weights_dict['dino'] * dino_probs
    
    return ensemble_probs


def predict_raw_per_model(image, weights_dict=None):
    """
    Get predictions from each model individually (for analysis).
    Only runs models with non-zero weights if weights_dict is provided.
    
    Args:
        image: Tensor of shape (batch_size, 3, H, W) with values in [0, 1]
        weights_dict: Optional dictionary to filter which models to run
    
    Returns:
        Dictionary with predictions from each model
    """
    with torch.no_grad():
        results = {}
        model_list = []
        
        # Inception V3
        if weights_dict is None or weights_dict.get('inception', 0) > 0:
            inception_batch = inception_preprocess(image)
            inception_logits = inception_model(inception_batch)
            if isinstance(inception_logits, tuple):
                inception_logits = inception_logits[0]
            results['inception'] = F.softmax(inception_logits, dim=1)
            model_list.append('inception')
        
        # ResNet18
        if weights_dict is None or weights_dict.get('resnet', 0) > 0:
            resnet_batch = resnet_preprocess(image)
            resnet_logits = resnet_model(resnet_batch)
            results['resnet'] = F.softmax(resnet_logits, dim=1)
            model_list.append('resnet')
        
        # VGG16
        if weights_dict is None or weights_dict.get('vgg', 0) > 0:
            vgg_batch = vgg_preprocess(image)
            vgg_logits = vgg_model(vgg_batch)
            results['vgg'] = F.softmax(vgg_logits, dim=1)
            model_list.append('vgg')
        
        # ViT-B/16
        if weights_dict is None or weights_dict.get('vit', 0) > 0:
            vit_batch = vit_preprocess(image)
            vit_logits = vit_model(vit_batch)
            results['vit'] = F.softmax(vit_logits, dim=1)
            model_list.append('vit')
        
        # DINOv2
        if weights_dict is None or weights_dict.get('dino', 0) > 0:
            dino_batch = dino_transform(image)
            dino_logits = dino_model(dino_batch)
            results['dino'] = F.softmax(dino_logits, dim=1)
            model_list.append('dino')
        
        # Ensemble (only from active models)
        if model_list:
            ensemble_probs = sum(results[m] for m in model_list) / len(model_list)
            results['ensemble'] = ensemble_probs
        
        return results


def ensemble_predict(image):
    """
    User-friendly prediction function (like the original classifiers).
    
    Returns:
        String with predicted class and confidence
    """
    with torch.no_grad():
        prediction = predict_raw(image)
        if prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = weights.meta["categories"][class_id]
            return f"{category_name}: {100 * score:.3f}%"
        
        res_lst = []
        for p in prediction:
            class_id = p.argmax().item()
            category_name = weights.meta["categories"][class_id]
            res_lst.append(category_name)
        return res_lst


# Target classes configuration
batch_size = 1
# orig_clases = torch.tensor([899]).cuda()  # water jug (default from dino)
# orig_clases = torch.tensor([817, 705, 609, 586, 436, 627, 468, 621, 803, 407, 408, 751, 717,866, 661, 864]).cuda()

total_clases_without_orig = torch.tensor(
    [x for x in list(range(0, 1000)) if x not in orig_clases]
).cuda()


def adv_loss_calc(image):
    """
    Calculate adversarial loss on ensemble predictions.
    """
    assert len(image.shape) == 4, "Image should be of shape (batch_size, 3, h, w)"
    adv_loss = []
    pred = predict_raw(image)
    for p in pred:
        adv_loss.append(p[orig_clases].mean())
    return torch.stack(adv_loss)


def adv_loss_calc2(image):
    """
    Alternative adversarial loss: difference between forbidden and allowed classes.
    """
    adv_loss = []
    pred = predict_raw(image)
    for p in pred:
        forbiden = p[orig_clases].max()
        allowed = p[total_clases_without_orig].max()
        adv_loss.append(forbiden - allowed)
    return torch.stack(adv_loss)


def print_model_agreement(image):
    """
    Utility function to see how models agree/disagree on predictions.
    """
    results = predict_raw_per_model(image)
    
    print("Individual Model Predictions:")
    print("-" * 60)
    
    for model_name in ['inception', 'resnet', 'vgg', 'vit', 'dino']:
        pred = results[model_name]
        class_id = pred[0].argmax().item()
        score = pred[0, class_id].item()
        category = weights.meta["categories"][class_id]
        print(f"{model_name:12s}: {category:30s} ({score:.3f})")
    
    print("-" * 60)
    ensemble_pred = results['ensemble']
    class_id = ensemble_pred[0].argmax().item()
    score = ensemble_pred[0, class_id].item()
    category = weights.meta["categories"][class_id]
    print(f"{'ENSEMBLE':12s}: {category:30s} ({score:.3f})")
    print("-" * 60)


print("\n✓ Ensemble classifier ready!")
print("  Main function: predict_raw(image)")
print("  Alternatives: predict_raw_weighted(image, weights_dict)")
print("               predict_raw_per_model(image)")
print("               ensemble_predict(image)")
