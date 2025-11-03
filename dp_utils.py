import warnings
from opacus import PrivacyEngine
import os

DP_LOG_DIR = os.path.join("logs", "dp")
os.makedirs(DP_LOG_DIR, exist_ok=True)
DP_CLIENT_CSV = os.path.join(DP_LOG_DIR, "dp_client_metrics.csv")

def apply_opacus(model, optimizer, dataloader, client_id, cloud_id,
                 noise_multiplier=0.5, max_grad_norm=0.7,
                 target_delta=1e-5):
    dataset_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    sample_rate = batch_size / max(1, dataset_size)
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
    )
    model._dp_meta = {
        "engine": privacy_engine,
        "sample_rate": sample_rate,
        "delta": target_delta,
        "steps": 0,
        "client_id": client_id,
        "cloud_id": cloud_id,
        "sigma": noise_multiplier,
        "clip": max_grad_norm,
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "log_csv": DP_CLIENT_CSV,
    }
    return model, optimizer, dataloader
