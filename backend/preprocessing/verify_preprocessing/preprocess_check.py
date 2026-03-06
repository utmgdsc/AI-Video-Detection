# Useful code snippet here to put at the end of preprocessing of ensemble and training script
# # --- INSERT DEBUGGING CODE HERE ---
# torch.save(faces.cpu(), "ensemble_tensor.pt")
# torchvision.utils.save_image(faces.cpu(), "ensemble_grid.jpg", normalize=True)
# # ----------------------------------

# # DEBUG: Check the exact mathematical shape going into the model
# logger.info(f"DEBUG: Tensor batch shape before model: {faces.shape}")


import torch

# Load the saved mathematical tensors
t_eff = torch.load("efficientnet_tensor.pt")
t_ens = torch.load("ensemble_tensor.pt")

print("=== SHAPE COMPARISON ===")
print(f"EfficientNet expected: {t_eff.shape}")
print(f"Ensemble provided:   {t_ens.shape}")

print("\n=== PIXEL DISTRIBUTION COMPARISON ===")
print("EfficientNet Math:")
print(f"Min: {t_eff.min():.4f} | Max: {t_eff.max():.4f} | Mean: {t_eff.float().mean():.4f}")

print("\nEnsemble Math:")
print(f"Min: {t_ens.min():.4f} | Max: {t_ens.max():.4f} | Mean: {t_ens.float().mean():.4f}")