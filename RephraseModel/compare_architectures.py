import torch
from model import ContentFilterModel
from model_attention_only import ContentFilterModelAttentionOnly
from data_processor import ContentDataset
from torch.utils.data import DataLoader
import time
import psutil
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def measure_inference_latency(model, test_loader, device):
    """Measure average inference time per batch"""
    times = []
    with torch.no_grad():
        for batch in test_loader:
            start = time.time()
            _ = model(batch['input_ids'].to(device), 
                     batch['attention_mask'].to(device))
            times.append(time.time() - start)
    return sum(times) / len(times)

def measure_memory_usage(model, test_batch, device):
    """Measure peak memory usage during inference"""
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated()
    _ = model(test_batch['input_ids'].to(device), 
              test_batch['attention_mask'].to(device))
    end_mem = torch.cuda.memory_allocated()
    return end_mem - start_mem

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 2)
            
            # Only consider non-padded tokens
            active_accuracy = attention_mask.view(-1) == 1
            labels = labels.view(-1)[active_accuracy]
            predicted = predicted.view(-1)[active_accuracy]
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def compare_models(train_loader, val_loader, test_loader):
    """Compare BiLSTM+Attention vs Attention-only architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize models
        logger.info("Initializing BiLSTM+Attention model...")
        bilstm_model = ContentFilterModel()
        logger.info("Model parameters: %d", sum(p.numel() for p in bilstm_model.parameters()))
        bilstm_model = bilstm_model.to(device)
        
        logger.info("Initializing Attention-only model...")
        attn_model = ContentFilterModelAttentionOnly()
        logger.info("Model parameters: %d", sum(p.numel() for p in attn_model.parameters()))
        attn_model = attn_model.to(device)
        
        logger.info("Models initialized successfully")
    
    # Results dictionary
    results = {
        'accuracy': {'bilstm': 0, 'attention': 0},
        'latency': {'bilstm': 0, 'attention': 0},
        'memory': {'bilstm': 0, 'attention': 0},
        'training_time': {'bilstm': 0, 'attention': 0}
    }
    logger.info("Starting model comparison...")
    
    # Compare models
    for model_name, model in [('bilstm', bilstm_model), 
                            ('attention', attn_model)]:
        # Training time
        start_time = time.time()
        train_model(model, train_loader, val_loader)
        results['training_time'][model_name] = time.time() - start_time
        
        # Inference metrics
        results['latency'][model_name] = measure_inference_latency(
            model, test_loader, device)
        results['memory'][model_name] = measure_memory_usage(
            model, next(iter(test_loader)), device)
        results['accuracy'][model_name] = evaluate_model(
            model, test_loader, device)
    
    return results

def train_model(model, train_loader, val_loader, 
                num_epochs=5, learning_rate=2e-5):
    """Train model with early stopping"""
    device = next(model.parameters()).device
    logger.info(f"Training model on device: {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    patience = 3
    no_improvement = 0
    logger.info(f"Starting training for {num_epochs} epochs with learning rate {learning_rate}")
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        # Training
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            if batch_idx % 10 == 0:
                logger.info(f"Training batch {batch_idx}/{len(train_loader)}")
            
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss only on non-padded tokens
            active_loss = attention_mask.view(-1) == 1
            active_logits = outputs.view(-1, 2)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            loss = torch.nn.functional.cross_entropy(
                active_logits, active_labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                # Calculate validation loss
                active_loss = attention_mask.view(-1) == 1
                active_logits = outputs.view(-1, 2)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                
                val_loss += torch.nn.functional.cross_entropy(
                    active_logits, active_labels).item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break
