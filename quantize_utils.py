import torch
import torch.nn as nn

try:
    from torch.ao.quantization import (
        quantize_dynamic, get_default_qat_qconfig, prepare_qat, convert
    )
    QTOOLS_AVAILABLE = True
except Exception:
    QTOOLS_AVAILABLE = False


def dynamic_quantize_linear(model: torch.nn.Module):
    if not QTOOLS_AVAILABLE:
        raise RuntimeError("Quantization modules not available in this PyTorch build.")
    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def run_qat(student_ctor, train_loader, epochs=2, lr=1e-3, device=None):
    if not QTOOLS_AVAILABLE:
        return None, {"note": "QAT non disponible."}
    try:
        model = student_ctor().to(device)
        model.qconfig = get_default_qat_qconfig("fbgemm")
        prepare_qat(model, inplace=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # En vrai, remplacez ceci par un vrai train (perte, etc.).
                opt.zero_grad(); loss = model(x).sum()*0; loss.backward(); opt.step()
        qat_int8 = convert(model.eval().cpu(), inplace=False)
        return qat_int8, {}
    except Exception as e:
        return None, {"error": f"QAT échoué : {e}"}
