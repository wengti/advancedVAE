
import torch
import torch.nn as nn

loss_fn = nn.MSELoss()

def train_step(model, dataloader, device, optimizer):

    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        label = None
        if model.config['conditional']:
            label = y # Dataloader makes it a torch tensor and is already on device

        X_recon, mean, log_var = model(x = X, label = label)

        recon_loss = loss_fn(X_recon, X)
        kl_loss = torch.mean(torch.sum(1 + log_var - mean**2 - torch.exp(log_var), dim=1) * (-0.5))
        loss = recon_loss + 0.00001 * kl_loss

        
        train_loss += loss
        torch.cuda.synchronize() ## Synchronize check point 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    torch.cuda.synchronize() ## Synchronize check point 1
    train_loss /= len(dataloader)

    return {'model_name': model.__class__.__name__,
            'loss': train_loss}

def test_step(model, dataloader, device):

    test_loss = 0

    model.eval()
    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            label = None
            if model.config['conditional']:
                label = y

            X_recon, mean, log_var = model(x = X, label = label)

            recon_loss = loss_fn(X_recon, X)
            kl_loss = torch.mean(torch.sum(1 + log_var - mean**2 - torch.exp(log_var), dim=1) * (-0.5))
            loss = recon_loss + 0.00001 * kl_loss
            
            test_loss += loss
            torch.cuda.synchronize() ## Synchronize check point 2

        torch.cuda.synchronize() ## Synchronize check point 2
        test_loss /= len(dataloader)

        return {'model_name': model.__class__.__name__,
                'loss': test_loss}

