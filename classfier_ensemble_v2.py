from IPython.core.display import Image
from torchvision.io import read_image
import torchvision
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    convnext_base, ConvNeXt_Base_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    swin_b, Swin_B_Weights
)
from consts import orig_clases

# Initialize all models
print("Loading ensemble v2 models...")
model = 'ensamble_classifier_v2'
weights = ConvNeXt_Base_Weights
# 1. ConvNeXt Base
convnext_weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
convnext_model = convnext_base(weights=convnext_weights)
convnext_model = convnext_model.eval().cuda()
convnext_preprocess = convnext_weights.transforms()

# 2. EfficientNet B0
efficientnet_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
efficientnet_model = efficientnet_b0(weights=efficientnet_weights)
efficientnet_model = efficientnet_model.eval().cuda()
efficientnet_preprocess = efficientnet_weights.transforms()

# 3. MobileNetV3 Large
mobilenet_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
mobilenet_model = mobilenet_v3_large(weights=mobilenet_weights)
mobilenet_model = mobilenet_model.eval().cuda()
mobilenet_preprocess = mobilenet_weights.transforms()

# 4. Swin Transformer Base
swin_weights = Swin_B_Weights.IMAGENET1K_V1
swin_model = swin_b(weights=swin_weights)
swin_model = swin_model.eval().cuda()
swin_preprocess = swin_weights.transforms()

# Use weights from one of the models for categories (they all use ImageNet)
weights = convnext_weights

print("✓ All models loaded successfully!")
print(f"  - ConvNeXt Base")
print(f"  - EfficientNet B0")
print(f"  - MobileNetV3 Large")
print(f"  - Swin Transformer Base")


def predict_raw_balanced(image):
    """
    Ensemble prediction using all 4 models.
    
    Args:
        image: Tensor of shape (batch_size, 3, H, W) with values in [0, 1]
    
    Returns:
        Combined softmax probabilities averaged across all models
    """
    # 1. ConvNeXt Base prediction
    convnext_batch = convnext_preprocess(image)
    convnext_logits = convnext_model(convnext_batch)
    convnext_probs = F.softmax(convnext_logits, dim=1)
    
    # 2. EfficientNet B0 prediction
    efficientnet_batch = efficientnet_preprocess(image)
    efficientnet_logits = efficientnet_model(efficientnet_batch)
    efficientnet_probs = F.softmax(efficientnet_logits, dim=1)
    
    # 3. MobileNetV3 Large prediction
    mobilenet_batch = mobilenet_preprocess(image)
    mobilenet_logits = mobilenet_model(mobilenet_batch)
    mobilenet_probs = F.softmax(mobilenet_logits, dim=1)
    
    # 4. Swin Transformer Base prediction
    swin_batch = swin_preprocess(image)
    swin_logits = swin_model(swin_batch)
    swin_probs = F.softmax(swin_logits, dim=1)
    
    # Ensemble: Average probabilities from all models
    ensemble_probs = (convnext_probs + efficientnet_probs + mobilenet_probs + swin_probs) / 4.0
    
    return ensemble_probs


def predict_raw(image, weights_dict=None):
    """
    Ensemble prediction with custom weights for each model.
    
    Args:
        image: Tensor of shape (batch_size, 3, H, W) with values in [0, 1]
        weights_dict: Dictionary with keys ['convnext', 'efficientnet', 'mobilenet', 'swin']
                     Default: equal weights (0.25 each)
    
    Returns:
        Weighted combination of softmax probabilities
    """
    if weights_dict is None:
        weights_dict = {
            'convnext': 0.25,
            'efficientnet': 0.25,
            'mobilenet': 0.25,
            'swin': 0.25
        }
    
    ensemble_probs = None
    
    # 1. ConvNeXt Base prediction
    if weights_dict.get('convnext', 0) > 0:
        convnext_batch = convnext_preprocess(image)
        convnext_logits = convnext_model(convnext_batch)
        convnext_probs = F.softmax(convnext_logits, dim=1)
        if ensemble_probs is None:
            ensemble_probs = weights_dict['convnext'] * convnext_probs
        else:
            ensemble_probs += weights_dict['convnext'] * convnext_probs
    
    # 2. EfficientNet B0 prediction
    if weights_dict.get('efficientnet', 0) > 0:
        efficientnet_batch = efficientnet_preprocess(image)
        efficientnet_logits = efficientnet_model(efficientnet_batch)
        efficientnet_probs = F.softmax(efficientnet_logits, dim=1)
        if ensemble_probs is None:
            ensemble_probs = weights_dict['efficientnet'] * efficientnet_probs
        else:
            ensemble_probs += weights_dict['efficientnet'] * efficientnet_probs
    
    # 3. MobileNetV3 Large prediction
    if weights_dict.get('mobilenet', 0) > 0:
        mobilenet_batch = mobilenet_preprocess(image)
        mobilenet_logits = mobilenet_model(mobilenet_batch)
        mobilenet_probs = F.softmax(mobilenet_logits, dim=1)
        if ensemble_probs is None:
            ensemble_probs = weights_dict['mobilenet'] * mobilenet_probs
        else:
            ensemble_probs += weights_dict['mobilenet'] * mobilenet_probs
    
    # 4. Swin Transformer Base prediction
    if weights_dict.get('swin', 0) > 0:
        swin_batch = swin_preprocess(image)
        swin_logits = swin_model(swin_batch)
        swin_probs = F.softmax(swin_logits, dim=1)
        if ensemble_probs is None:
            ensemble_probs = weights_dict['swin'] * swin_probs
        else:
            ensemble_probs += weights_dict['swin'] * swin_probs
    
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
        
        # ConvNeXt Base
        if weights_dict is None or weights_dict.get('convnext', 0) > 0:
            convnext_batch = convnext_preprocess(image)
            convnext_logits = convnext_model(convnext_batch)
            results['convnext'] = F.softmax(convnext_logits, dim=1)
            model_list.append('convnext')
        
        # EfficientNet B0
        if weights_dict is None or weights_dict.get('efficientnet', 0) > 0:
            efficientnet_batch = efficientnet_preprocess(image)
            efficientnet_logits = efficientnet_model(efficientnet_batch)
            results['efficientnet'] = F.softmax(efficientnet_logits, dim=1)
            model_list.append('efficientnet')
        
        # MobileNetV3 Large
        if weights_dict is None or weights_dict.get('mobilenet', 0) > 0:
            mobilenet_batch = mobilenet_preprocess(image)
            mobilenet_logits = mobilenet_model(mobilenet_batch)
            results['mobilenet'] = F.softmax(mobilenet_logits, dim=1)
            model_list.append('mobilenet')
        
        # Swin Transformer Base
        if weights_dict is None or weights_dict.get('swin', 0) > 0:
            swin_batch = swin_preprocess(image)
            swin_logits = swin_model(swin_batch)
            results['swin'] = F.softmax(swin_logits, dim=1)
            model_list.append('swin')
        
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
    
    for model_name in ['convnext', 'efficientnet', 'mobilenet', 'swin']:
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


print("\n✓ Ensemble classifier v2 ready!")
print("  Main function: predict_raw(image)")
print("  Alternatives: predict_raw_weighted(image, weights_dict)")
print("               predict_raw_per_model(image)")
print("               ensemble_predict(image)")
