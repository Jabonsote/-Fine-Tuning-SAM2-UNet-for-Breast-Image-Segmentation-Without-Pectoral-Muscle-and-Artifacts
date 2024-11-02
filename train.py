import os
import argparse
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from dataset import FullDataset
from SAM2UNet import SAM2UNet

# Configuración de argumentos
parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True, help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True, help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True, help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20, help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


# Funciones para métricas IoU y Dice
def compute_iou(pred, mask):
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum((1, 2, 3))
    union = (pred + mask).sum((1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean()


def compute_dice(pred, mask):
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum((1, 2, 3))
    dice = (2. * intersection + 1e-7) / (pred.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1e-7)
    return dice.mean()


# Función de pérdida personalizada
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# Entrenamiento de una época
def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    epoch_loss, iou_score, dice_score = 0.0, 0.0, 0.0
    
    for i, batch in enumerate(dataloader):
        x = batch['image'].to(device)
        target = batch['label'].to(device)
        optimizer.zero_grad()
        
        # Forward y cálculo de pérdida
        pred0, pred1, pred2 = model(x)
        loss0 = structure_loss(pred0, target)
        loss1 = structure_loss(pred1, target)
        loss2 = structure_loss(pred2, target)
        loss = loss0 + loss1 + loss2
        
        # Métricas de IoU y Dice para predicciones
        iou = compute_iou(torch.sigmoid(pred0), target)
        dice = compute_dice(torch.sigmoid(pred0), target)
        
        # Backward y optimización
        loss.backward()
        optimizer.step()
        
        # Acumular pérdida y métricas
        epoch_loss += loss.item()
        iou_score += iou.item()
        dice_score += dice.item()
        
        # Log en wandb para cada lote
        if i % 50 == 0:
            wandb.log({"batch_loss": loss.item(), "batch_iou": iou.item(), "batch_dice": dice.item(),
                       "batch_lr": scheduler.get_last_lr()[0]})
    
    # Calcular promedios por época
    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_iou = iou_score / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    
    # Log de métricas por época
    wandb.log({"epoch_loss": avg_epoch_loss, "epoch_iou": avg_iou, "epoch_dice": avg_dice,
               "epoch_lr": scheduler.get_last_lr()[0]})
    scheduler.step()
    
    return avg_epoch_loss, avg_iou, avg_dice


# Visualizar ejemplos de predicción en WandB
def log_predictions(model, dataloader, device, epoch):
    model.eval()
    images_to_log = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            pred = torch.sigmoid(model(x)[0])
            images_to_log.append(wandb.Image(x[0].cpu(), caption="Original"))
            images_to_log.append(wandb.Image(target[0].cpu(), caption="Mask"))
            images_to_log.append(wandb.Image(pred[0].cpu(), caption="Prediction"))
            if len(images_to_log) >= 6:  # Limita el número de imágenes
                break
    wandb.log({"examples_epoch_{}".format(epoch + 1): images_to_log})


# Función principal de entrenamiento
def main(args):
    # Inicializa wandb y registra hiperparámetros
    run = wandb.init(project="SAM2-UNet-training", config=args,
                     name=f"SAM2-UNet_lr{args.lr}_bs{args.batch_size}",
                     tags=["segmentation", "SAM2-UNet"])
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preparación de dataset y modelo
    dataset = FullDataset(config.train_image_path, config.train_mask_path, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=10)
    model = SAM2UNet(config.hiera_path).to(device)
    optimizer = opt.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, config.epoch, eta_min=1.0e-7)
    os.makedirs(config.save_path, exist_ok=True)
    
    # Inicializa mejor IoU
    best_iou = 0.0
    best_model_path = None
    
    # Entrenamiento por épocas
    for epoch in range(config.epoch):
        avg_loss, avg_iou, avg_dice = train_one_epoch(model, dataloader, optimizer, scheduler, device)
        
        # Guardar mejores predicciones y métricas en consola
        print(f"Epoch [{epoch + 1}/{config.epoch}] - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")
        
        # Guarda ejemplos de predicciones en WandB cada 5 épocas
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.epoch:
            log_predictions(model, dataloader, device, epoch)
        
        # Guardar modelo si el IoU actual es el mejor
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_model_path = os.path.join(config.save_path, f'best_model_epoch-{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)
            
            # Crear y agregar el archivo al artefacto antes de hacer log
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(best_model_path)
            wandb.run.summary["best_iou"] = best_iou
            wandb.log_artifact(artifact)  # Registrar el artefacto en wandb
            
            print(f"[Saving Best Model: IoU={best_iou:.4f} at {best_model_path}]")

    
    # Termina wandb
    wandb.finish()


if __name__ == "__main__":
    main(args)
