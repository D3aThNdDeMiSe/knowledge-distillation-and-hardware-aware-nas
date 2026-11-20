"""
Knowledge-distillation framework (Deliverable 4).

Total student loss for one mini-batch:

    L = w_hard * CE(z_s, y)                                   (hard labels)
      + w_soft * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )  (soft targets)
      + beta   * sum_l  L_feat( r_l(F_s^l), F_t^l )             (intermediate features)

  * z_s, z_t: student / teacher logits. The teacher is frozen (eval mode, no grad).
  * Temperature T > 1 softens both distributions so the student also learns the
    teacher's "dark knowledge" (relative probabilities of wrong classes).
    The T^2 factor keeps soft-target gradients on the same scale as the hard-label
    gradients when T changes (Hinton et al., 2015).
  * w_hard = 1 - alpha, w_soft = alpha  -> combined hard-label + soft-label training.
  * Feature-based distillation (FitNets, Romero et al., 2015): for each tapped layer l
    a small trainable regressor r_l (1x1 conv + BN) maps the student feature map to
    the teacher's channel count; spatial size is matched with adaptive average
    pooling. Each feature map is L2-normalised per sample before the squared error,
    so the loss is scale-invariant (teacher activations have a very different
    magnitude from student activations) and bounded in [0, 4].

Each component can be switched off independently, which is how the ablations
(soft-only, hard+soft, hard+soft+feature) are produced.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_kd_loss(student_logits, teacher_logits, T):
    """Hinton KD loss: KL divergence between temperature-softened distributions, times T^2."""
    log_p_s = F.log_softmax(student_logits.float() / T, dim=1)
    p_t = F.softmax(teacher_logits.float() / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def temperature_softmax(logits, T=1.0):
    """Temperature-scaled softmax (T=1 -> ordinary softmax, T>1 -> softer distribution)."""
    return F.softmax(logits / T, dim=1)


class FeatureRegressor(nn.Module):
    """Projects a student feature map onto the teacher's feature space (1x1 conv + BN)."""

    def __init__(self, s_ch, t_ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(s_ch, t_ch, 1, bias=False), nn.BatchNorm2d(t_ch))

    def forward(self, f_s, target_hw):
        f = self.proj(f_s)
        if f.shape[-2:] != target_hw:
            f = F.adaptive_avg_pool2d(f, target_hw)
        return f


def feature_distillation_loss(f_s, f_t):
    """Squared L2 distance between per-sample L2-normalised feature maps (mean over batch)."""
    a = F.normalize(f_s.float().flatten(1), dim=1)
    b = F.normalize(f_t.float().flatten(1), dim=1)
    return (a - b).pow(2).sum(dim=1).mean()


class DistillationLoss(nn.Module):
    """Combines hard-label CE, soft-target KD and feature-based distillation.

    Args:
        student_channels / teacher_channels: dict layer_name -> channels of the tapped maps
        feat_layers: list of layer names to distil ('mid', 'final'); empty -> no feature KD
        T: softmax temperature
        alpha: weight of the soft-target term (hard term gets 1 - alpha)
        beta: weight of each feature term
        use_hard / use_soft: switch individual terms on/off (ablations)
    """

    def __init__(self, student_channels, teacher_channels, feat_layers=("mid", "final"), T=4.0,
                 alpha=0.9, beta=2.0, use_hard=True, use_soft=True):
        super().__init__()
        self.T, self.alpha, self.beta = T, alpha, beta
        self.use_hard, self.use_soft = use_hard, use_soft
        self.feat_layers = list(feat_layers)
        self.regressors = nn.ModuleDict({
            l: FeatureRegressor(student_channels[l], teacher_channels[l]) for l in self.feat_layers})
        self.ce = nn.CrossEntropyLoss()

    def forward(self, s_logits, s_feats, t_logits, t_feats, y):
        parts = {}
        loss = 0.0
        w_soft = self.alpha if self.use_hard else 1.0
        w_hard = (1.0 - self.alpha) if self.use_soft else 1.0
        if self.use_hard:
            parts["ce"] = self.ce(s_logits.float(), y)
            loss = loss + w_hard * parts["ce"]
        if self.use_soft:
            parts["kd"] = soft_target_kd_loss(s_logits, t_logits, self.T)
            loss = loss + w_soft * parts["kd"]
        for l in self.feat_layers:
            f_t = t_feats[l]
            f_s = self.regressors[l](s_feats[l], f_t.shape[-2:])
            parts[f"feat_{l}"] = feature_distillation_loss(f_s, f_t)
            loss = loss + self.beta * parts[f"feat_{l}"]
        return loss, {k: v.detach() for k, v in parts.items()}  # tensors: no GPU sync here


def make_kd_loss_fn(teacher, criterion):
    """Adapter for common.utils.train_one_epoch: runs the frozen teacher on the SAME
    augmented batch, then the student, then the combined distillation loss."""
    teacher.eval()
    need_feats = len(criterion.feat_layers) > 0

    def fn(student, x, y):
        with torch.no_grad():
            # Teacher runs in channels_last (faster for dense convs); student stays NCHW.
            x_t = x.contiguous(memory_format=torch.channels_last)
            t_logits, t_feats = teacher(x_t, return_features=True)
            t_feats = {k: v.contiguous() for k, v in t_feats.items()}
        s_logits, s_feats = student(x, return_features=True)
        loss, parts = criterion(s_logits, s_feats if need_feats else {}, t_logits,
                                t_feats if need_feats else {}, y)
        return loss, s_logits, parts

    return fn
