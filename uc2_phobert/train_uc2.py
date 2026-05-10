"""
UC2 PhoBERT Vietnamese Sentiment — MLflow Pipeline
Dataset: UIT-VSFC (Vietnamese Students Feedback Corpus)
Model: vinai/phobert-base (HuggingFace)

Chạy trên Google Colab (cần GPU T4):
    python3 -u train_uc2.py --mode all

Modes:
    all    = chạy repeat (TC7) + config sweep (TC2)
    repeat = chỉ lặp 3 lần để đo TC7
    tc2    = chỉ chạy config sweep
"""
import mlflow
import torch
import time
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# CONFIG — Tương tự UC1, 1 dict duy nhất quản lý toàn bộ
# ============================================================
CONFIG = {
    "tracking_uri":  "http://<GCP_EXTERNAL_IP>:5000",  # ← ĐỔI IP GCP
    "experiment":    "UC2_PhoBERT_MLflow",
    "model_name":    "vinai/phobert-base",
    "dataset_name":  "uitnlp/vietnamese_students_feedback",
    "num_labels":    3,
    "max_length":    256,
    "num_runs":      3,           # Số lần lặp để lấy TB TC7

    # Hyperparameter mặc định
    "default": {
        "lr":         2e-5,
        "batch_size": 16,
        "epochs":     3,
    },

    # TC2 — Config sweep (3 bộ config khác nhau)
    "tc2_configs": [
        {"lr": 1e-5, "batch_size": 16, "epochs": 3},
        {"lr": 2e-5, "batch_size": 16, "epochs": 3},
        {"lr": 3e-5, "batch_size": 32, "epochs": 3},
    ],
}


# ============================================================
# DATA LOADING & TOKENIZATION
# ============================================================
def load_and_tokenize():
    """Load UIT-VSFC dataset và tokenize bằng PhoBERT tokenizer."""
    print("  Loading dataset:", CONFIG["dataset_name"])
    dataset = load_dataset(CONFIG["dataset_name"])

    print(f"  Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("sentiment", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized, tokenizer


# ============================================================
# TRAINING FUNCTION — 1 run duy nhất
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def train_one_run(tokenized, tokenizer, lr, batch_size, epochs, run_name):
    """Fine-tune PhoBERT 1 lần, log toàn bộ lên MLflow."""

    with mlflow.start_run(run_name=run_name):
        start_time = time.time()

        # ---- Log params ----
        mlflow.log_param("model", CONFIG["model_name"])
        mlflow.log_param("dataset", CONFIG["dataset_name"])
        mlflow.log_param("num_labels", CONFIG["num_labels"])
        mlflow.log_param("max_length", CONFIG["max_length"])
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("platform", "MLflow-GCP")
        mlflow.log_param("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

        # ---- Load model ----
        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["model_name"],
            num_labels=CONFIG["num_labels"],
        )

        # ---- Training args ----
        training_args = TrainingArguments(
            output_dir="./results_uc2",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=50,
            fp16=torch.cuda.is_available(),  # Mixed precision nếu có GPU
            report_to="none",  # Tắt wandb/tensorboard mặc định
        )

        # ---- Trainer ----
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
        )

        # ---- Train ----
        print(f"    Training {run_name}...")
        trainer.train()

        # ---- Evaluate on test set ----
        test_results = trainer.evaluate(tokenized["test"])
        accuracy = test_results["eval_accuracy"]
        f1 = test_results["eval_f1"]
        pipeline_time = time.time() - start_time

        # ---- Log metrics ----
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("pipeline_time_seconds", pipeline_time)

        # ---- Log model artifact ----
        mlflow.log_param("status", "completed")

        print(f"    ✓ Accuracy={accuracy:.4f} | F1={f1:.4f} | TC7={pipeline_time:.1f}s")
        return accuracy, f1, pipeline_time


# ============================================================
# MAIN
# ============================================================
def main(mode="all"):
    # Setup MLflow
    mlflow.set_tracking_uri(CONFIG["tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment"])

    # Load data (1 lần duy nhất, dùng lại cho tất cả runs)
    print("\n📦 Loading & tokenizing UIT-VSFC dataset...")
    tokenized, tokenizer = load_and_tokenize()

    d = CONFIG["default"]

    if mode in ("all", "repeat"):
        # ---- PHẦN 1: Chạy lặp num_runs lần để lấy TB TC7 ----
        print(f"\n{'='*60}")
        print(f" PHẦN 1: Chạy lặp {CONFIG['num_runs']} lần — đo TC7")
        print(f"{'='*60}")
        for i in range(1, CONFIG["num_runs"] + 1):
            print(f"\n  [PhoBERT] Lần {i}/{CONFIG['num_runs']}")
            train_one_run(
                tokenized=tokenized,
                tokenizer=tokenizer,
                lr=d["lr"],
                batch_size=d["batch_size"],
                epochs=d["epochs"],
                run_name=f"PhoBERT_run{i}",
            )

    if mode in ("all", "tc2"):
        # ---- PHẦN 2: TC2 — Config sweep ----
        print(f"\n{'='*60}")
        print(f" PHẦN 2: TC2 Config Sweep ({len(CONFIG['tc2_configs'])} configs)")
        print(f"{'='*60}")
        for cfg in CONFIG["tc2_configs"]:
            run_name = f"TC2_lr{cfg['lr']}_batch{cfg['batch_size']}"
            print(f"\n  Config: {run_name}")
            train_one_run(
                tokenized=tokenized,
                tokenizer=tokenizer,
                run_name=run_name,
                **cfg,
            )

    print(f"\n{'='*60}")
    print(f" ✅ Hoàn thành UC2! Xem kết quả: {CONFIG['tracking_uri']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UC2 PhoBERT — MLflow Pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "repeat", "tc2"],
        default="all",
        help="all=chạy tất cả | repeat=chỉ lặp TC7 | tc2=chỉ config sweep",
    )
    args = parser.parse_args()
    main(args.mode)
