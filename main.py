#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch

from .models import TeacherCNN, StudentSmall
from .data import get_loaders
from .train import train_epoch, evaluate
from .distill import train_kd
from .prune_utils import apply_global_unstructured_pruning
from .quantize_utils import dynamic_quantize_linear, run_qat, QTOOLS_AVAILABLE
from .utils import (
    set_seed, estimate_state_dict_size_bytes, save_state_dict, save_and_gzip, write_json
)


def quick_report(name, model, loader, device):
    loss, acc, lat = evaluate(model, loader, device)
    size_mb = estimate_state_dict_size_bytes(model)/1e6
    print(f"{name:22s} | acc={acc:.3f} | size≈{size_mb:.2f}MB | batch_latency≈{lat*1000:.1f}ms")


def main():
    Bool = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="tuto_model_shrink modulaire (PyTorch/MNIST)")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="./artifacts")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--prune_ratio", type=float, default=0.5)
    p.add_argument("--qat_epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # Toggles
    p.add_argument("--teacher", action=Bool, default=True, help="Entraîner le teacher")
    p.add_argument("--student_base", action=Bool, default=False, help="Entraîner un student baseline")
    p.add_argument("--distill", action=Bool, default=True, help="Distillation teacher→student")
    p.add_argument("--prune", action=Bool, default=True, help="Pruning du student")
    p.add_argument("--dynquant", action=Bool, default=True, help="Dynamic quant INT8 (CPU)")
    p.add_argument("--qat", action=Bool, default=True, help="QAT + conversion INT8 (CPU)")
    p.add_argument("--fp16", action=Bool, default=True, help="Export poids FP16")
    p.add_argument("--gzip", action=Bool, default=True, help="Sauver aussi un .pth.gz")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch)

    # Pointeurs de modèles disponibles
    teacher = None
    student_base = None
    student_kd = None
    student_pruned = None
    student_dyn = None
    qat_int8 = None

    # 1) Teacher (souvent nécessaire pour distillation)
    if args.teacher:
        teacher = TeacherCNN().to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        for epoch in range(args.epochs):
            tr_loss, tr_acc = train_epoch(teacher, train_loader, opt, device)
            te_loss, te_acc, _ = evaluate(teacher, test_loader, device)
            print(f"[Teacher][{epoch}] train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")
        save_state_dict(teacher, out / "teacher_fp32.pth")
        if args.gzip:
            save_and_gzip(teacher, out / "teacher_fp32.pth")

    # 2) Student baseline (référence facultative)
    if args.student_base:
        student_base = StudentSmall().to(device)
        opt = torch.optim.Adam(student_base.parameters(), lr=1e-3)
        train_epoch(student_base, train_loader, opt, device)
        save_state_dict(student_base, out / "student_base_fp32.pth")
        if args.gzip:
            save_and_gzip(student_base, out / "student_base_fp32.pth")

    # 3) Distillation → student_kd
    if args.distill:
        if teacher is None:
            # En l'absence de teacher (si --no-teacher), on en entraîne un minimal
            print("[INFO] Pas de teacher disponible, entraînement rapide d'un teacher pour la distillation…")
            teacher = TeacherCNN().to(device)
            opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
            for epoch in range(max(1, args.epochs)):
                train_epoch(teacher, train_loader, opt, device)
        student_kd = StudentSmall().to(device)
        train_kd(student_kd, teacher, train_loader, device, epochs=args.epochs, lr=1e-3, T=4.0, alpha=0.7)
        save_state_dict(student_kd, out / "student_kd_fp32.pth")
        if args.gzip:
            save_and_gzip(student_kd, out / "student_kd_fp32.pth")

    # 4) Pruning + fine-tuning
    if args.prune:
        # Source prioritaire: KD -> baseline; sinon on fabrique un baseline court
        source = student_kd or student_base
        if source is None:
            if teacher is not None:
                print("[INFO] Pas de student, on crée un student baseline pour le pruning…")
                student_base = StudentSmall().to(device)
                opt = torch.optim.Adam(student_base.parameters(), lr=1e-3)
                train_epoch(student_base, train_loader, opt, device)
                source = student_base
            else:
                raise RuntimeError("Aucun modèle à pruner. Activez --distill ou --student_base (ou --teacher).")
        student_pruned = StudentSmall().to(device)
        student_pruned.load_state_dict(source.state_dict())
        apply_global_unstructured_pruning(student_pruned, amount=args.prune_ratio)
        opt = torch.optim.Adam(student_pruned.parameters(), lr=5e-4)
        train_epoch(student_pruned, train_loader, opt, device)
        save_state_dict(student_pruned, out / "student_pruned_fp32.pth")
        if args.gzip:
            save_and_gzip(student_pruned, out / "student_pruned_fp32.pth")

    # 5) Dynamic quantization (CPU)
    if args.dynquant and QTOOLS_AVAILABLE:
        quant_src = student_kd or student_pruned or student_base or teacher
        if quant_src is None:
            print("[WARN] Aucun modèle disponible pour la quantification dynamique.")
        else:
            try:
                student_dyn = dynamic_quantize_linear(quant_src)
                # Évaluation sur CPU
                loss, acc, lat = evaluate(student_dyn, test_loader, device=torch.device("cpu"))
                print(f"[DynQuant] acc={acc:.3f} | batch_latency≈{lat*1000:.1f}ms (CPU)")
                save_state_dict(student_dyn, out / "dyn_int8.pth")
                if args.gzip:
                    save_and_gzip(student_dyn, out / "dyn_int8.pth")
            except Exception as e:
                print("[WARN] Dynamic quantization échouée:", e)
    elif args.dynquant and not QTOOLS_AVAILABLE:
        print("[INFO] Quantization AO non disponible dans cette build de PyTorch.")

    # 6) QAT (CPU)
    if args.qat:
        qat_int8, info = run_qat(StudentSmall, train_loader, epochs=args.qat_epochs, lr=1e-3, device=device)
        if qat_int8 is not None:
            loss, acc, lat = evaluate(qat_int8, test_loader, device=torch.device("cpu"))
            print(f"[QAT INT8] acc={acc:.3f} | batch_latency≈{lat*1000:.1f}ms (CPU)")
            save_state_dict(qat_int8, out / "student_qat_int8.pth")
            if args.gzip:
                save_and_gzip(qat_int8, out / "student_qat_int8.pth")
        else:
            print("[INFO] QAT info:", info)

    # 7) Export FP16 (fichier)
    if args.fp16:
        src = student_kd or student_pruned or student_base or teacher
        if src is not None:
            state16 = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in src.state_dict().items()}
            torch.save(state16, out / "model_fp16.pth")
        else:
            print("[INFO] Aucun modèle pour export FP16.")

    # Récap & rapport JSON
    print("\n--- RECAP ---")
    def maybe_report(name, model, dev):
        if model is None: return None
        quick_report(name, model, test_loader, dev)
        loss, acc, lat = evaluate(model, test_loader, dev)
        return {"acc": acc, "latency_s": lat, "size_MB": estimate_state_dict_size_bytes(model)/1e6}

    rep = {}
    for name, model, dev in [
        ("teacher_fp32", teacher, device),
        ("student_base_fp32", student_base, device),
        ("student_kd_fp32", student_kd, device),
        ("student_pruned_fp32", student_pruned, device),
    ]:
        if model is not None:
            report_entry = maybe_report(name.replace("_fp32","").replace("_"," ").title(), model, dev)
            rep[name] = report_entry

    if student_dyn is not None:
        loss, acc, lat = evaluate(student_dyn, test_loader, device=torch.device("cpu"))
        rep["dyn_int8"] = {"acc": acc, "latency_s": lat}
    if qat_int8 is not None:
        loss, acc, lat = evaluate(qat_int8, test_loader, device=torch.device("cpu"))
        rep["student_qat_int8"] = {"acc": acc, "latency_s": lat}

    write_json(rep, out / "report.json")
    print(f"\nRapport: {out / 'report.json'}")

if __name__ == "__main__":
    main()
