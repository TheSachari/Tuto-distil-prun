import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, y, T=4.0, alpha=0.7):
    hard = F.cross_entropy(student_logits, y)
    p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits  / T, dim=1)
    soft = F.kl_div(p, q, reduction="batchmean") * (T*T)
    return alpha*soft + (1-alpha)*hard

def train_kd(student, teacher, loader, device, epochs=2, lr=1e-3, T=4.0, alpha=0.7):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for _ in range(epochs):
        student.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = kd_loss(s_logits, t_logits, y, T=T, alpha=alpha)
            opt.zero_grad(); loss.backward(); opt.step()
