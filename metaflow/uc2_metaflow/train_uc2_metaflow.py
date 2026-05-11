"""
UC2 PhoBERT Vietnamese Sentiment — Metaflow Pipeline (Local Mode)

Chạy trên Google Colab (cần GPU T4):
  python3 train_uc2_metaflow.py run
  python3 train_uc2_metaflow.py run --lr 1e-5
  python3 train_uc2_metaflow.py run --lr 3e-5 --batch_size 32
  python3 train_uc2_metaflow.py run --lr 1e-4 --batch_size 32
"""
import os
os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"

from metaflow import FlowSpec, step, Parameter
import time


class PhoBERTSentimentFlow(FlowSpec):
    """Pipeline UC2 PhoBERT Sentiment — Metaflow"""

    lr = Parameter('lr', default=2e-5, type=float,
                   help='Learning rate')
    batch_size = Parameter('batch_size', default=16, type=int,
                           help='Batch size')
    epochs = Parameter('epochs', default=3, type=int,
                       help='Number of epochs')
    max_length = Parameter('max_length', default=256, type=int,
                           help='Max token length for PhoBERT')

    @step
    def start(self):
        self.start_time = time.time()
        print(f"\n{'='*50}")
        print(f"  UC2 PhoBERT Sentiment — Metaflow (Colab GPU)")
        print(f"  Config: lr={self.lr}, batch={self.batch_size}, epochs={self.epochs}")
        print(f"{'='*50}")
        self.next(self.load_and_train)

    @step
    def load_and_train(self):
        """
        Load data + Tokenize + Train + Evaluate (gộp 1 step)

        LÝ DO GỘP: PyTorch models, DataLoaders, và HuggingFace Datasets
        KHÔNG thể pickle giữa các @step trong Metaflow.
        Đây là hạn chế của Metaflow với deep learning workloads.
        """
        import numpy as np
        import torch
        import pandas as pd
        from datasets import Dataset, DatasetDict
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from sklearn.metrics import accuracy_score, f1_score

        # ========== LOAD DATASET ==========
        print("  📦 Loading UIT-VSFC dataset...")
        urls = {
            "train": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "validation": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
            "test": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        }

        train_df = pd.read_parquet(urls["train"])
        val_df = pd.read_parquet(urls["validation"])
        test_df = pd.read_parquet(urls["test"])

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        })
        print(f"  Data: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")

        # ========== TOKENIZE ==========
        print("  🔤 Tokenizing with PhoBERT...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        def tokenize_fn(examples):
            return tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized = tokenized.rename_column("sentiment", "labels")
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # ========== TRAIN ==========
        print("  🚀 Fine-tuning PhoBERT...")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average="weighted"),
            }

        model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base", num_labels=3
        )

        training_args = TrainingArguments(
            output_dir="./results_uc2_metaflow",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to="none",
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # ========== EVALUATE ==========
        print("  📊 Evaluating on test set...")
        test_results = trainer.evaluate(tokenized["test"])

        self.accuracy = test_results["eval_accuracy"]
        self.f1 = test_results["eval_f1"]
        self.pipeline_time = time.time() - self.start_time
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        print(f"\n  ✅ Accuracy: {self.accuracy:.4f}")
        print(f"  ✅ F1: {self.f1:.4f}")
        print(f"  ✅ TC7 (pipeline time): {self.pipeline_time:.1f}s")
        print(f"  ✅ GPU: {self.gpu_name}")
        self.next(self.end)

    @step
    def end(self):
        print(f"\n{'='*50}")
        print(f"  FINAL RESULT")
        print(f"  Model: vinai/phobert-base")
        print(f"  Dataset: UIT-VSFC")
        print(f"  Accuracy: {self.accuracy:.4f}")
        print(f"  F1: {self.f1:.4f}")
        print(f"  TC7: {self.pipeline_time:.1f}s")
        print(f"  Config: lr={self.lr}, batch={self.batch_size}, epochs={self.epochs}")
        print(f"  Platform: Metaflow Local Mode (Colab {self.gpu_name})")
        print(f"{'='*50}")


if __name__ == '__main__':
    PhoBERTSentimentFlow()
