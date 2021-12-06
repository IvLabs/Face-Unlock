import argparse
import random
import os
from torchvision import models
from tqdm import tqdm
from termcolor import colored

import numpy as np
import torch

from config import create_config
from utils import get_train_dataset, get_val_dataset, get_dataloader, get_val_loader, get_model, get_optimizer, get_scheduler
import triplet_loss
from eval import evaluate

import wandb

parser = argparse.ArgumentParser(description="Train LFW on usinf Triplet Loss")
parser.add_argument('--config', help="Location of config file")
parser.add_argument('--data_dir', default="/", help="Dataset Location")
parser.add_argument('--checkpoint_dir', default="/checkpoints/", help="Location of checkpoints to store")
parser.add_argument('--resume', default=None, help="path to checkpoint from where to resume")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_wandb = True

def seed_init():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

scaler = torch.cuda.amp.GradScaler()
def train(p, dataset, model, criterion, optimizer):
    loader = get_dataloader(p, dataset)
    model.train()
    epoch_loss=0
    for step, (images, labels) in enumerate(loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.cuda.amp.autocast():
            embds = model(images)
            loss = criterion(labels, embds, margin=0.8)
        epoch_loss += loss.item()

        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)

        scaler.update()

        # print(f"Loss: {loss.item():.3f}, ", end=" ") 

    return epoch_loss/len(loader)

@torch.inference_mode()
def validate(model, loader):
    model.eval()
    distances = []
    labels = []
    for pair1, pair2, label in tqdm(loader):
        pair1 = pair1.to(DEVICE)
        pair2 = pair2.to(DEVICE)
        embds1 = model(pair1).cpu()
        embds2 = model(pair2).cpu()
        distance = (embds1 - embds2).norm(p=2, dim=1)
        distances.append(distance)
        labels.append(label)
    distances = torch.cat(distances)
    labels = torch.cat(labels)
    best_threshold, tar, far, precision, accuracy, fig = evaluate(distances.detach().numpy(), labels.numpy(), False)
    if log_wandb:
        wandb.log({"ROC Curve": fig}, commit=False)
        wandb.log({
            "Recall": tar,
            "Precision": precision,
            "Accuracy": accuracy,
            "FAR":far
        })
    return tar, precision, accuracy, far, best_threshold


def main():
    args = parser.parse_args()
    p = create_config(args.config)

    seed_init()

    if log_wandb:
        config = wandb.config
        config.batch_size = p.batch_size
        config.epochs = p.epochs
        config.learning_rate = p.optimizer_kwargs.lr
        config.scheduler = p.scheduler
        config.fc_layer_size = p.fc_layer_size
        config.train_dataset = "LFW"
        config.architechture = p.backbone
    # dataset
    print(colored('Get dataset and dataloaders', 'blue'))
    train_dataset = get_train_dataset(p)
    print(train_dataset)
    val_dataset = get_val_dataset(p)
    val_loader = get_val_loader(p, val_dataset)

    # model
    print(colored('Get model', 'blue'))
    model = get_model(p)
    model.to(DEVICE)
    print(model)

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # scheduler
    print(colored('Get scheduler', 'blue'))
    scheduler = get_scheduler(p, optimizer)
    print(scheduler)

    # Loss function
    criterion = triplet_loss.batch_hard_triplet_loss

    # checkpoint
    if args.resume is not None and os.path.exists(args.resume):
        print(colored('Loading checkpoint {} ...'.format(args.resume), 'blue'))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    else:
        print(colored('No checkpoint. Training from scratch.'.format(args.resume), 'blue'))
        start_epoch = 0
    
    for epoch in range(start_epoch, p.epochs):
        
        epoch_loss = train(p, train_dataset, model, criterion, optimizer)
        scheduler.step(epoch_loss)
        if log_wandb:
            # wandb.log({"loss": loss.item()})
            wandb.log({"epoch_loss": epoch_loss,
                        "lr":optimizer.state_dict()["param_groups"][0]['lr']},
                        commit="False")
        
        if epoch % 5 == 0:
            tar, precision, accuracy, far, best_threshold = validate(model, val_loader)
            print("Best Threshold: {}\nTrue Acceptance: {:.3f}\nFalse Acceptance: {:.3f}\nPrecision: {:.3f}\nAccuracy: {:.3f}".format(best_threshold, tar, far, precision, accuracy))
        
        if epoch % 10 == 0:
            # Save model checkpoint
            state = {
                'epoch': epoch+1,
                'embedding_dimension': p.fc_layer_size,
                'batch_size_training': p.batch_size,
                'model_state_dict': model.state_dict(),
                'model_architecture': p.backbone,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_distance_threshold': best_threshold
            }
            # Save model checkpoint
            torch.save(state, 'model_training_checkpoints/model_{}_triplet_epoch_{}.pt'.format(
                    p.backbone,
                    epoch
                )
            )

if __name__ == '__main__':
    main()