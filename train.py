import torch


def train(device, model, dataloader, loss_func, optimizer, schedule=None):
    model.train()
    batch_num = len(dataloader)
    print_interval = batch_num // 5

    for batch, (x_audios, x_mel_images, x_frames, y) in enumerate(dataloader):
        x_audios, x_mel_images, x_frames, y = x_audios.to(device), x_mel_images.to(device), x_frames.to(device), y.to(device)
        pred = model(x_audios, x_mel_images, x_frames)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if schedule is not None:
            schedule.step()

        if batch % print_interval == 0:
            correct_num = (torch.argmax(pred, dim=-1) == y).type(torch.float).sum().item()
            acc = correct_num / len(y)
            print(f"[Batch {batch:>4d}/{batch_num:>4d}]\tLoss: {loss.item():.3f}, Acc: {acc:.3f}")


def val_or_test(mode, device, model, dataloader, loss_func):
    assert mode in ["val", "test"]

    model.eval()
    size = len(dataloader.dataset)
    correct_num = 0
    loss_sum = 0
    batch_num = len(dataloader)

    with torch.no_grad():
        for x_audios, x_mel_images, x_frames, y in dataloader:
            x_audios, x_mel_images, x_frames, y = x_audios.to(device), x_mel_images.to(device), x_frames.to(device), y.to(device)
            pred = model(x_audios, x_mel_images, x_frames)
            loss_sum += loss_func(pred, y).item()
            correct_num += (torch.argmax(pred, dim=-1) == y).type(torch.float).sum().item()

    acc = correct_num / size
    print(f"{mode.capitalize()} Loss: {loss_sum/batch_num:.4f}, Accuracy: {acc:.4f}")
    return acc
