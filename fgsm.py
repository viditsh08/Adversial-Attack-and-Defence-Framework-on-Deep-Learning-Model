import torch

def fgsm_attack(image, epsilon, data_grad):
    # Get sign of gradient
    sign_data_grad = data_grad.sign()
    
    # Create adversarial image
    perturbed_image = image + epsilon * sign_data_grad
    
    # Clamp to valid range [0,1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image