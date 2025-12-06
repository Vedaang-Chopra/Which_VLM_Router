# Enhanced Training Guide with W&B and Data Validation

This guide shows how to enhance the existing notebooks with:
1. **Data validation and cleaning**
2. **Weights & Biases (W&B) logging**
3. **Complete train/val/test evaluation**
4. **Final accuracy plots on all splits**

## Prerequisites

```bash
# Install dependencies
cd router_train
pip install -r requirements.txt

# Login to W&B (one-time setup)
wandb login
```

## Key Additions to All Notebooks

### 1. Import Additional Modules

Add these imports at the top of each notebook:

```python
import wandb
from training.data_validation import (
    validate_profiles_dataframe,
    clean_profiles_dataframe,
    validate_train_val_test_split,
)
from training.train_utils import (
    init_wandb,
    train_epoch_reward_router,
    evaluate_reward_router,
    compute_routing_accuracy,
    log_final_results,
)
```

### 2. Data Validation (After Loading from SQL)

Replace simple data loading with validation + cleaning:

```python
# Load data
df_profiles_raw = load_profiles_real_schema(
    db_config=db_config,
    limit=LIMIT,
)

print(f"\n{'='*60}")
print("DATA VALIDATION")
print(f"{'='*60}")

# Validate data quality
validation_report = validate_profiles_dataframe(
    df=df_profiles_raw,
    allow_missing_images=True,
    allow_missing_confidence=True,
    max_missing_cost_pct=10.0,
    max_missing_latency_pct=10.0,
    max_missing_glider_pct=5.0,
)

if not validation_report['validation_passed']:
    print("\\n⚠️  Data validation found critical issues:")
    for error in validation_report['errors']:
        print(f"  ERROR: {error}")
    raise ValueError("Data validation failed")

# Clean data
df_profiles, cleaning_stats = clean_profiles_dataframe(
    df=df_profiles_raw,
    fill_missing_cost=True,
    fill_missing_latency=True,
    fill_missing_confidence=True,
    drop_missing_glider=True,
    drop_duplicates=True,
)

print(f"\\n{'='*60}")
print("DATA CLEANING SUMMARY")
print(f"{'='*60}")
print(f"  Initial rows:      {cleaning_stats['initial_rows']:,}")
print(f"  Final rows:        {cleaning_stats['final_rows']:,}")
print(f"  Dropped rows:      {cleaning_stats['dropped_rows']:,}")
if 'filled_cost' in cleaning_stats:
    print(f"  Filled costs:      {cleaning_stats['filled_cost']:,}")
if 'filled_latency' in cleaning_stats:
    print(f"  Filled latencies:  {cleaning_stats['filled_latency']:,}")
print(f"{'='*60}\\n")
```

### 3. Load ALL Splits (train/val/test)

```python
# Load train/val/test splits
print("[LOADING ALL DATA SPLITS]")

df_train_profiles = load_profiles_real_schema(db_config, data_split="train")
df_val_profiles = load_profiles_real_schema(db_config, data_split="val")
df_test_profiles = load_profiles_real_schema(db_config, data_split="test")

# Clean all splits
df_train_profiles, _ = clean_profiles_dataframe(df_train_profiles)
df_val_profiles, _ = clean_profiles_dataframe(df_val_profiles)
df_test_profiles, _ = clean_profiles_dataframe(df_test_profiles)

# Validate split integrity
split_stats = validate_train_val_test_split(
    train_df=df_train_profiles,
    val_df=df_val_profiles,
    test_df=df_test_profiles,
)

print(f"\\nSplit Statistics:")
print(f"  Train: {split_stats['train_samples']:,} samples ({split_stats['train_pct']:.1f}%)")
print(f"  Val:   {split_stats['val_samples']:,} samples ({split_stats['val_pct']:.1f}%)")
print(f"  Test:  {split_stats['test_samples']:,} samples ({split_stats['test_pct']:.1f}%)")
```

### 4. Initialize W&B

Add this before training:

```python
# Initialize Weights & Biases
RUN_NAME = f"reward_router_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

wandb_config = {
    'model_type': 'reward_router',  # or 'pairwise_router', 'classical_router'
    'num_epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'text_encoder': TEXT_ENCODER,
    'max_seq_length': MAX_SEQ_LENGTH,
    'num_models': len(model_to_id),
    'num_modes': len(mode_to_id),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
}

wandb_run = init_wandb(
    project_name="vlm-router-training",
    run_name=RUN_NAME,
    config=wandb_config,
    tags=['reward_router', 'sql_data'],  # Adjust tags per notebook
)
```

### 5. Training Loop with W&B Logging

```python
# Training loop
history = {
    'train_loss': [],
    'val_loss': [],
    'val_pearson': [],
    'val_routing_accuracy': [],
}

print("[TRAINING]\\n")

for epoch in range(NUM_EPOCHS):
    # Train
    train_metrics = train_epoch_reward_router(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        gradient_clip=GRADIENT_CLIP,
        log_wandb=True,
        epoch=epoch,
    )

    # Evaluate on validation set
    val_metrics, val_preds, val_targets = evaluate_reward_router(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        split_name="val",
    )

    # Compute routing accuracy on validation
    val_df_with_preds = val_df.copy()
    val_df_with_preds['pred_reward'] = val_preds

    val_routing_metrics = compute_routing_accuracy(
        df=val_df_with_preds,
        model_to_id=model_to_id,
        id_to_model=id_to_model,
        mode_to_id=mode_to_id,
        id_to_mode=id_to_mode,
    )

    # Update history
    history['train_loss'].append(train_metrics['train_loss'])
    history['val_loss'].append(val_metrics['val_loss'])
    history['val_pearson'].append(val_metrics['val_pearson'])
    history['val_routing_accuracy'].append(val_routing_metrics['routing_accuracy'])

    # Log to W&B
    if wandb_run:
        wandb.log({
            'epoch': epoch + 1,
            **train_metrics,
            **val_metrics,
            **{f'val_routing/{k}': v for k, v in val_routing_metrics.items()},
        })

    # Print epoch summary
    print(f"\\n[EPOCH {epoch+1}/{NUM_EPOCHS}]")
    print(f"  Train Loss:       {train_metrics['train_loss']:.4f}")
    print(f"  Val Loss:         {val_metrics['val_loss']:.4f}")
    print(f"  Val Pearson:      {val_metrics['val_pearson']:.4f}")
    print(f"  Val Routing Acc:  {100*val_routing_metrics['routing_accuracy']:.2f}%")
    print()

print("✓ Training complete")
```

### 6. Final Evaluation on All Splits

```python
# Final evaluation on all splits
print(f"\\n{'='*80}")
print("FINAL EVALUATION ON ALL SPLITS")
print(f"{'='*80}\\n")

# Train set
train_metrics, train_preds, train_targets = evaluate_reward_router(
    model, train_loader, criterion, DEVICE, "train"
)
train_df_with_preds = train_df.copy()
train_df_with_preds['pred_reward'] = train_preds
train_routing_metrics = compute_routing_accuracy(
    train_df_with_preds, model_to_id, id_to_model, mode_to_id, id_to_mode
)

# Val set
val_metrics, val_preds, val_targets = evaluate_reward_router(
    model, val_loader, criterion, DEVICE, "val"
)
val_df_with_preds = val_df.copy()
val_df_with_preds['pred_reward'] = val_preds
val_routing_metrics = compute_routing_accuracy(
    val_df_with_preds, model_to_id, id_to_model, mode_to_id, id_to_mode
)

# Test set
test_metrics, test_preds, test_targets = evaluate_reward_router(
    model, test_loader, criterion, DEVICE, "test"
)
test_df_with_preds = test_df.copy()
test_df_with_preds['pred_reward'] = test_preds
test_routing_metrics = compute_routing_accuracy(
    test_df_with_preds, model_to_id, id_to_model, mode_to_id, id_to_mode
)

# Log final results
log_final_results(
    train_metrics=train_metrics,
    val_metrics=val_metrics,
    test_metrics=test_metrics,
    val_routing_metrics=val_routing_metrics,
    test_routing_metrics=test_routing_metrics,
    log_wandb=True,
)
```

### 7. Final Accuracy Plots

```python
# Plot final accuracy on all splits
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Routing Accuracy by Mode
for idx, (split_name, routing_metrics) in enumerate([
    ('Train', train_routing_metrics),
    ('Val', val_routing_metrics),
    ('Test', test_routing_metrics),
]):
    ax = axes[0, idx]

    # Extract per-mode accuracy
    mode_acc = {
        mode: routing_metrics.get(f'routing_accuracy_{mode}', 0)
        for mode in ['accuracy', 'cheap', 'fast', 'balanced']
    }

    ax.bar(mode_acc.keys(), [100*v for v in mode_acc.values()],
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.axhline(100*routing_metrics['routing_accuracy'], color='red',
               linestyle='--', label=f"Overall={100*routing_metrics['routing_accuracy']:.1f}%")
    ax.set_ylabel('Routing Accuracy (%)')
    ax.set_title(f'{split_name} Set Routing Accuracy')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

# Row 2: Reward Recovery
for idx, (split_name, routing_metrics) in enumerate([
    ('Train', train_routing_metrics),
    ('Val', val_routing_metrics),
    ('Test', test_routing_metrics),
]):
    ax = axes[1, idx]

    # Reward recovery bar
    recovery_pct = 100 * routing_metrics.get('reward_recovery', 0)
    ax.bar(['Reward\\nRecovery'], [recovery_pct],
           color='green' if recovery_pct > 90 else 'orange',
           edgecolor='black', alpha=0.7, width=0.5)
    ax.axhline(100, color='red', linestyle='--', label='Oracle (100%)')
    ax.set_ylabel('Reward Recovery (%)')
    ax.set_title(f'{split_name} Set Reward Recovery')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Log to W&B
if wandb_run:
    wandb.log({"final_accuracy_plots": wandb.Image(fig)})
```

### 8. Final Summary Table

```python
# Create summary table
summary_data = []
for split_name, metrics, routing_metrics in [
    ('Train', train_metrics, train_routing_metrics),
    ('Val', val_metrics, val_routing_metrics),
    ('Test', test_metrics, test_routing_metrics),
]:
    summary_data.append({
        'Split': split_name,
        'Loss': f"{metrics[f'{split_name.lower()}_loss']:.4f}",
        'Pearson': f"{metrics[f'{split_name.lower()}_pearson']:.4f}",
        'Routing Acc': f"{100*routing_metrics['routing_accuracy']:.2f}%",
        'Reward Recovery': f"{100*routing_metrics['reward_recovery']:.2f}%",
        'Avg Reward Gap': f"{routing_metrics['avg_reward_gap']:.4f}",
    })

summary_df = pd.DataFrame(summary_data)

print(f"\\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}")
display(summary_df)

# Log to W&B
if wandb_run:
    wandb.log({"summary_table": wandb.Table(dataframe=summary_df)})
```

### 9. Close W&B Run

```python
# Close W&B run
if wandb_run:
    wandb.finish()
    print("\\n✓ W&B run finished")
```

## What to Expect in W&B Dashboard

After running the enhanced notebooks, you'll see:

1. **Training Curves**:
   - `train/batch_loss` - Loss per batch
   - `train_loss` - Average loss per epoch
   - `val_loss`, `test_loss` - Validation/test loss
   - `val_pearson`, `test_pearson` - Prediction correlation

2. **Routing Metrics**:
   - `val_routing/routing_accuracy` - Overall routing accuracy
   - `val_routing/routing_accuracy_{mode}` - Per-mode accuracy
   - `val_routing/reward_recovery` - % of oracle reward recovered
   - Similar metrics for test set

3. **Final Summary**:
   - `final_test_accuracy` - Test set routing accuracy
   - `final_test_reward_recovery` - Test set reward recovery
   - `final_test_pearson` - Test set prediction correlation

4. **Visualizations**:
   - Accuracy plots by mode and split
   - Training curves
   - Summary tables

## Notebook-Specific Adjustments

### For Pairwise Ranking Router (03)

Change the training function call to:
```python
# Use pairwise training instead
loss = model.compute_pairwise_loss(
    sample_texts=batch['sample_texts'],
    model_i_ids=batch['model_i_ids'],
    model_j_ids=batch['model_j_ids'],
    mode_ids=batch['mode_ids'],
    margin=MARGIN,
)
```

Evaluation uses `rank_models()` instead of predicting rewards.

### For Classical Router (04)

Training uses combined CE + KL loss:
```python
loss, loss_dict = model.compute_loss(
    sample_texts=batch['sample_texts'],
    mode_ids=batch['mode_ids'],
    hard_labels=batch['hard_labels'],
    soft_labels=batch['soft_labels'],
    temperature=TEMPERATURE,
    alpha=ALPHA,
)

# Log individual loss components
if wandb_run:
    wandb.log({
        'train/ce_loss': loss_dict['ce_loss'],
        'train/kl_loss': loss_dict.get('kl_loss', 0.0),
    })
```

## Common Issues and Solutions

### Issue: "Module 'wandb' not found"
**Solution**: Install wandb: `pip install wandb`

### Issue: "W&B login required"
**Solution**: Run `wandb login` and enter your API key

### Issue: "Data validation failed"
**Solution**: Check the validation report errors and adjust `max_missing_*_pct` parameters or fix data quality issues in the database

### Issue: "Out of memory during training"
**Solution**: Reduce `BATCH_SIZE` or use gradient accumulation

### Issue: "Data leakage detected"
**Solution**: Ensure `data_split` column is correctly set in the database with no overlapping sample_ids

## Files Created

1. `training/data_validation.py` - Data validation and cleaning utilities
2. `training/train_utils.py` - W&B logging and evaluation utilities
3. `notebooks/ENHANCED_TRAINING_GUIDE.md` - This guide
4. Updated `requirements.txt` - Added `wandb` and `seaborn`

## Next Steps

1. Install updated requirements: `pip install -r requirements.txt`
2. Login to W&B: `wandb login`
3. Update each notebook with the code snippets above
4. Run all three notebooks and compare results in W&B dashboard
5. Use W&B sweep for hyperparameter tuning (optional)

Happy training! 🚀
