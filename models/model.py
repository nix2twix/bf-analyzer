# === LIBRARIES GENERAL ===
import torch
import segmentation_models_pytorch as smp

def loadCheckpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    return model
    

def buildModel(classesCount = 4, encoderName = 'resnet34', encoderWeights = 'imagenet', activation = None):
    model = smp.UnetPlusPlus(
        encoder_name=encoderName,
        encoder_weights=encoderWeights,
        in_channels=1,  # grayscale
        classes=classesCount,
        activation=activation
    )
    return model