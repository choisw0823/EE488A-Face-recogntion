def OneCycleLRScheduler(optimizer, max_lr, epochs, steps_per_epoch, **kwargs):
    print('Initialised OneCycleLR scheduler')
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
