import warnings
from opacus import PrivacyEngine

warnings.filterwarnings("ignore")

def apply_opacus(model, optimizer, data_loader, noise_multiplier=0.5, max_grad_norm=1.0, client_id=None, cloud_id=None):
    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=10.0,
        target_delta=1e-5,
        epochs=1,
        max_grad_norm=max_grad_norm
    )

    if client_id is not None and cloud_id is not None:
        print(f"[Cloud {cloud_id}] Client {(client_id%5)+1} Differential Privacy completed.")

    return model, optimizer, data_loader
