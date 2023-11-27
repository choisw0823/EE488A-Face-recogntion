def CosineAnnealingScheduler(optimizer, T_max, eta_min=0, **kwargs):
    print('Initialised Cosine Annealing LR scheduler')
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
