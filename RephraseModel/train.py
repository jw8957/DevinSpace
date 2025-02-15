import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from data_processor import ContentDataset
from model import ContentFilterModel
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: ContentFilterModel,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    output_dir: str = "model_outputs",
    early_stopping_patience: int = 3
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0.0
    no_improvement = 0
    
    try:
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                # Reshape outputs and labels for loss calculation
                active_loss = attention_mask.view(-1) == 1
                active_logits = outputs.view(-1, 2)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                
                loss = torch.nn.functional.cross_entropy(active_logits, active_labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_train_loss}")
            
            # Validation phase
            model.eval()
            val_loss = 0
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    # Reshape outputs and labels for loss calculation
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = outputs.view(-1, 2)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    
                    loss = torch.nn.functional.cross_entropy(active_logits, active_labels)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=-1)
                    # Only collect predictions for non-padded tokens
                    for pred, label, mask in zip(preds, labels, attention_mask):
                        valid_preds = pred[mask == 1].cpu().numpy()
                        valid_labels = label[mask == 1].cpu().numpy()
                        predictions.extend(valid_preds)
                        true_labels.extend(valid_labels)
            
            avg_val_loss = val_loss / len(val_loader)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary'
            )
            
            logger.info(f"Validation Loss: {avg_val_loss}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            
            # Save best model and check early stopping
            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_f1,
                }, os.path.join(output_dir, "best_model.pt"))
                logger.info(f"Saved new best model with F1: {f1:.4f}")
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def prepare_datasets():
    """Prepare and split datasets for training"""
    logger.info("Loading English dataset...")
    en_file = "/home/ubuntu/attachments/f0b4f54e-a7fc-49a6-9b0b-9b343acacf4b/cc_sample.20250215_en_2000.rephrase.jsonl"
    logger.info(f"English file exists: {os.path.exists(en_file)}")
    en_dataset = ContentDataset(en_file)
    
    logger.info("Loading Chinese dataset...")
    zh_file = "/home/ubuntu/attachments/29ad0211-497f-4350-aedd-335fd3e3d4fc/cc_sample.20250215_zh-hans_2000.rephrase.jsonl"
    logger.info(f"Chinese file exists: {os.path.exists(zh_file)}")
    zh_dataset = ContentDataset(zh_file)
    
    # Combine datasets
    total_size = len(en_dataset) + len(zh_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Create a combined dataset
    combined_dataset = torch.utils.data.ConcatDataset([en_dataset, zh_dataset])
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create output directory
        output_dir = "model_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset, val_dataset = prepare_datasets()
        logger.info(f"Created datasets - Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        logger.info("Data loaders created successfully")
        
        # Initialize and train model
        logger.info("Initializing model...")
        model = ContentFilterModel()
        model.to(device)
        
        train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            output_dir=output_dir,
            early_stopping_patience=3,
            num_epochs=10
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
