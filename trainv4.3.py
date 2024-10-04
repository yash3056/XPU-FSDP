import warnings
import os
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress all outputs during imports
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import pandas as pd
import numpy as np
from tqdm import tqdm
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch
from huggingface_hub import login
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

# Restore original stdout and stderr
sys.stdout = original_stdout
sys.stderr = original_stderr

def setup_fsdp(process_group):  # Add process_group as an argument
    """Configure FSDP settings optimized for XPU devices"""

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )
    cpu_offload = CPUOffload(offload_params=True)

    fsdp_config = {
        "mixed_precision": mixed_precision_policy,
        "cpu_offload": cpu_offload,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "forward_prefetch": True,
        "limit_all_gathers": True,
        "use_orig_params": True,
        "sync_module_states": True,
        "process_group": process_group # Use the passed process_group
    }

    return fsdp_config

def get_policies(model):
    transformer_wrap_policy = partial(  # Use partial to create a callable
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
        recurse=True,  # Still provide the arguments here
        nonwrapped_numel=1e6 
    )
    return transformer_wrap_policy   # Return the partial function

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group ONLY HERE
    dist.init_process_group("ccl", rank=rank, world_size=world_size)
    torch.xpu.set_device(rank)
    return dist.new_group(range(world_size)) #create new process group

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def plot_loss_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()

def plot_accuracy_curves(train_accuracies, val_accuracies):
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, 'b-', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curves.png')
    plt.close()

def validate_xpu_setup(rank, world_size):
    """Validate XPU device setup"""
    try:
        current_device = torch.xpu.current_device()
        assert current_device == rank, f"Process {rank} is not on correct XPU device (on device {current_device})"
        
        # Verify all XPUs are available
        available_devices = torch.xpu.device_count()
        assert available_devices >= world_size, f"Not enough XPU devices: {available_devices} available, {world_size} required"
        
        # Verify memory on current device
        memory_info = torch.xpu.get_device_properties(current_device)
        print(f"XPU {rank} memory: {memory_info.total_memory / 1024**3:.2f} GB")
        
        return True
    except Exception as e:
        print(f"XPU validation failed on rank {rank}: {str(e)}")
        return False

def train(rank, world_size):
    #setup(rank, world_size)
    # Get XPU process group
    process_group = setup(rank, world_size)

    if not validate_xpu_setup(rank, world_size):
        cleanup()
        return
        
    # Load dataset
    df = pd.read_csv("mental_health.csv")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
    )

    # Model initialization
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )

    # Move model to XPU and optimize
    model.to(f'xpu:{rank}')
    # model = ipex.llm.optimize(model, dtype=torch.bfloat16, device=f"xpu:{rank}", inplace=True)

    # Setup FSDP
    fsdp_config = setup_fsdp(process_group)
    wrap_policy = get_policies(model)

    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        device_id=f'xpu:{rank}',  # This ensures each process uses its assigned XPU
        **fsdp_config
    )

    # Create datasets and dataloaders
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset,batch_size=8,sampler=train_sampler,pin_memory=True,prefetch_factor=2,num_workers=4)
    val_dataloader = DataLoader(val_dataset,batch_size=8,sampler=val_sampler,pin_memory=True,prefetch_factor=2,num_workers=4)

    # Training parameters
    num_epochs = 10
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=100,num_training_steps=total_steps)

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        train_sampler.set_epoch(epoch)

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=rank != 0):
            input_ids = batch['input_ids'].to(f'xpu:{rank}')
            attention_mask = batch['attention_mask'].to(f'xpu:{rank}').to(torch.bfloat16)
            labels = batch['label'].to(f'xpu:{rank}')

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Calculate training accuracy
            preds = torch.argmax(outputs.logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Training accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", disable=rank != 0):
                input_ids = batch['input_ids'].to(f'xpu:{rank}')
                attention_mask = batch['attention_mask'].to(f'xpu:{rank}').to(torch.bfloat16)
                labels = batch['label'].to(f'xpu:{rank}')

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Gather predictions and labels from all processes
        all_preds = torch.tensor(all_preds).to(f'xpu:{rank}')
        all_labels = torch.tensor(all_labels).to(f'xpu:{rank}')
        gathered_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]

        dist.all_gather(gathered_preds, all_preds)
        dist.all_gather(gathered_labels, all_labels)

        if rank == 0:
            all_preds = torch.cat(gathered_preds).cpu().numpy()
            all_labels = torch.cat(gathered_labels).cpu().numpy()

            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )
            auc = roc_auc_score(all_labels, all_preds)

            avg_val_loss = val_loss / len(val_dataloader)
            val_accuracy = correct_val / total_val
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")

            # Plot metrics
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion_matrix(cm, class_names=['Negative', 'Positive'])

            fpr, tpr, _ = roc_curve(all_labels, all_preds)
            plot_roc_curve(fpr, tpr, auc)

            plot_loss_curves(train_losses, val_losses)
            plot_accuracy_curves(train_accuracies, val_accuracies)

            # Save checkpoint with FSDP
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = model.state_dict()
                if rank == 0:
                    torch.save(state_dict, f"checkpoint_epoch_{epoch}.pt")

    # Final model save
    if rank == 0:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
            torch.save(state_dict, "final_model_state.pt")
        
        # Save the tokenizer
        tokenizer.save_pretrained("./final_llama70b_model")
        print("Training completed and model saved.")
    
    cleanup()

def main():
    """
    Main function to run the training process
    """

    # Set environment variables for better performance
    os.environ["CCL_WORKER_COUNT"] = "4"
    os.environ["CCL_LOG_LEVEL"] = "info"
    os.environ["CCL_CMA_ENABLED"] = "0"  # Disable CMA for better performance
    
    # XPU specific optimizations
    os.environ["IPEX_MERGE_FUSION"] = "1"
    os.environ["IPEX_AUTO_KERNEL_SELECTION"] = "1"
    os.environ["IPEX_OPTIMIZE_MEMORY"] = "1"

    # Login to Hugging Face Hub
    #login(token=os.environ['HUGGINGFACE_TOKEN'])

    world_size = 16  # Number of GPUs
    try:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        # Attempt to clean up in case of failure
        try:
            cleanup()
        except Exception as cleanup_error:
            print(f"Cleanup failed with error: {str(cleanup_error)}")
        raise e
    finally:
        # Ensure cleanup is always attempted
        cleanup()
        
if __name__ == "__main__":
    main()