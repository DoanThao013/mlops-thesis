"""
Vẽ biểu đồ kết quả Metaflow — thay thế UI khi chạy local mode

Chạy trên VM Metaflow:
  pip install matplotlib pandas
  python3 plot_results.py

Output (lưu tại metaflow/plots/):
  uc1_accuracy_by_model.png   — Bar chart accuracy 3 model × 3 runs (UC1)
  uc1_pipeline_time.png       — Bar chart pipeline time 3 model (UC1)
  uc1_tc2_sweep.png           — Bar chart TC2 config sweep UC1
  uc2_baseline.png            — Bar chart accuracy + F1 baseline 3 runs (UC2)
  uc2_tc2_sweep.png           — Bar chart TC2 config sweep UC2
  uc1_dag.png                 — DAG flow graph UC1 (cần graphviz)
  uc2_dag.png                 — DAG flow graph UC2 (cần graphviz)
"""
import os
import sys
import subprocess

os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UC1_DIR    = os.path.join(SCRIPT_DIR, "uc1_metaflow")
UC2_DIR    = os.path.join(SCRIPT_DIR, "uc2_metaflow")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (không cần display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Màu sắc nhất quán
# ─────────────────────────────────────────────
COLORS = {
    "SimpleNN": "#4C72B0",
    "DeepNN":   "#DD8452",
    "CNN":      "#55A868",
    "baseline": "#4C72B0",
    "sweep":    "#DD8452",
}


# ─────────────────────────────────────────────
# Đọc dữ liệu từ Metaflow Client API
# ─────────────────────────────────────────────
def load_uc1_runs():
    """Đọc tất cả runs MNISTFlow, trả về DataFrame."""
    from metaflow import Flow
    original_cwd = os.getcwd()
    results = []
    try:
        os.chdir(UC1_DIR)
        flow = Flow("MNISTFlow")
        for run in flow.runs():
            try:
                results.append({
                    "run_id":          run.id,
                    "model":           run.data.model_name,
                    "lr":              run.data.lr,
                    "batch_size":      run.data.batch_size,
                    "accuracy":        round(run.data.accuracy, 4),
                    "train_time_s":    round(run.data.train_time, 1),
                    "eval_time_s":     round(run.data.eval_time, 1),
                    "pipeline_time_s": round(run.data.pipeline_time, 1),
                })
            except Exception:
                pass
    except Exception as e:
        print(f"  [UC1] Lỗi đọc flow: {e}")
    finally:
        os.chdir(original_cwd)
    return pd.DataFrame(results)


def load_uc2_runs():
    """Đọc tất cả runs PhoBERTSentimentFlow, trả về DataFrame.
    Chạy trong subprocess để Metaflow resolve đúng .metaflow/ tại UC2_DIR.
    """
    script = f"""
import os, sys, json
os.chdir(r"{UC2_DIR}")
os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DEFAULT_METADATA"]  = "local"
from metaflow import Flow
results = []
try:
    flow = Flow("PhoBERTSentimentFlow")
    for run in flow.runs():
        try:
            # training_seed: baseline=43/44/45, sweep=42 (default)
            try:
                tseed = run.data.training_seed
            except Exception:
                tseed = 42
            results.append({{
                "run_id":          run.id,
                "lr":              run.data.lr,
                "batch_size":      run.data.batch_size,
                "training_seed":   tseed,
                "accuracy":        round(run.data.accuracy, 4),
                "f1_macro":        round(run.data.f1_macro, 4),
                "train_time_s":    round(run.data.train_time, 1),
                "eval_time_s":     round(run.data.eval_time, 1),
                "pipeline_time_s": round(run.data.pipeline_time, 1),
            }})
        except Exception:
            pass
except Exception as e:
    pass
print(json.dumps(results))
"""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", script],
            stderr=subprocess.DEVNULL
        )
        import json
        data = json.loads(out.decode())
        return pd.DataFrame(data)
    except Exception as e:
        print(f"  [UC2] Lỗi đọc flow: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# UC1 — Biểu đồ 1: Accuracy 3 model × 3 runs
# ─────────────────────────────────────────────
def plot_uc1_accuracy(df):
    models = ["SimpleNN", "DeepNN", "CNN"]
    # Lọc baseline: lr=0.001, batch=64
    baseline = df[(df["lr"] == 0.001) & (df["batch_size"] == 64)].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.22

    for i, model in enumerate(models):
        subset = baseline[baseline["model"] == model].sort_values("run_id")
        accs   = subset["accuracy"].values[:3]  # tối đa 3 runs
        for j, acc in enumerate(accs):
            bar = ax.bar(x[i] + (j - 1) * width, acc, width,
                         color=COLORS[model], alpha=0.6 + j * 0.13,
                         label=f"{model} run{j+1}" if i == 0 else "")
            ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                    acc + 0.001, f"{acc:.4f}",
                    ha="center", va="bottom", fontsize=7.5)

        # Đường trung bình
        if len(accs) > 0:
            mean = accs.mean()
            ax.hlines(mean, x[i] - width * 1.6, x[i] + width * 1.6,
                      colors=COLORS[model], linestyles="--", linewidth=1.5)
            ax.text(x[i] + width * 1.7, mean, f"μ={mean:.4f}",
                    va="center", fontsize=8, color=COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("UC1 MNIST — Accuracy theo Model và Run (Metaflow)", fontsize=12, pad=12)
    ax.set_ylim(0.80, 1.01)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legend run1/run2/run3
    patches = [mpatches.Patch(facecolor="gray", alpha=0.6 + j * 0.13,
               label=f"Run {j+1}") for j in range(3)]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "uc1_accuracy_by_model.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# UC1 — Biểu đồ 2: Pipeline time trung bình
# ─────────────────────────────────────────────
def plot_uc1_pipeline_time(df):
    models    = ["SimpleNN", "DeepNN", "CNN"]
    baseline  = df[(df["lr"] == 0.001) & (df["batch_size"] == 64)].copy()

    means = []
    stds  = []
    for model in models:
        subset = baseline[baseline["model"] == model]["pipeline_time_s"]
        means.append(subset.mean())
        stds.append(subset.std() if len(subset) > 1 else 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [COLORS[m] for m in models]
    bars   = ax.bar(models, means, color=colors, width=0.5,
                    yerr=stds, capsize=6, alpha=0.85)

    for bar, mean, std in zip(bars, means, stds):
        label = f"{mean:.0f}s\n(±{std:.0f}s)" if std > 0 else f"{mean:.0f}s"
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + max(stds) * 0.05 + 20,
                label, ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Pipeline Time (giây)", fontsize=11)
    ax.set_title("UC1 MNIST — Pipeline Time trung bình (Metaflow)", fontsize=12, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "uc1_pipeline_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# UC1 — Biểu đồ 3: TC2 Config Sweep
# ─────────────────────────────────────────────
def plot_uc1_tc2(df):
    # TC2: SimpleNN với lr và batch_size khác nhau
    sweep = df[df["model"] == "SimpleNN"].copy()
    # Loại bỏ baseline (lr=0.001, batch=64)
    sweep = sweep[~((sweep["lr"] == 0.001) & (sweep["batch_size"] == 64))]
    # Thêm baseline trung bình để so sánh
    baseline_mean = df[(df["model"] == "SimpleNN") &
                       (df["lr"] == 0.001) & (df["batch_size"] == 64)]["accuracy"].mean()

    if sweep.empty:
        print("  [UC1 TC2] Chưa có sweep runs, bỏ qua biểu đồ.")
        return

    labels  = [f"lr={r['lr']}\nbatch={int(r['batch_size'])}" for _, r in sweep.iterrows()]
    accs    = sweep["accuracy"].values
    times   = sweep["pipeline_time_s"].values

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x      = np.arange(len(labels))
    width  = 0.38
    ax2    = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, accs, width, color="#4C72B0", alpha=0.85, label="Accuracy")
    bars2 = ax2.bar(x + width / 2, times, width, color="#DD8452", alpha=0.75, label="Pipeline Time (s)")

    # Đường baseline accuracy
    ax1.axhline(baseline_mean, color="#55A868", linestyle="--", linewidth=1.5,
                label=f"Baseline acc (lr=0.001, b=64): {baseline_mean:.4f}")

    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, acc + 0.003,
                 f"{acc:.4f}", ha="center", va="bottom", fontsize=8)
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, t + 5,
                 f"{t:.0f}s", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Test Accuracy", fontsize=11)
    ax2.set_ylabel("Pipeline Time (giây)", fontsize=11)
    ax1.set_title("UC1 MNIST — TC2 Config Sweep (Metaflow)", fontsize=12, pad=12)
    ax1.set_ylim(0.75, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "uc1_tc2_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# UC2 — Biểu đồ 4: Baseline accuracy + F1
# ─────────────────────────────────────────────
def plot_uc2_baseline(df):
    # Baseline: training_seed IN (43,44,45) — runs sau khi fix seed ngày 22/05
    # Fallback: lấy lr=2e-5, batch=2 sort tăng dần, 3 runs mới nhất
    if "training_seed" in df.columns:
        baseline = df[df["training_seed"].isin([43, 44, 45])].copy()
    else:
        baseline = df[(df["lr"] == 2e-5) & (df["batch_size"] == 2)].copy()

    if baseline.empty:
        baseline = df.copy()

    # Sort tăng dần theo run_id → Run 1 = run đầu tiên chạy
    baseline = baseline.sort_values("run_id", ascending=True).reset_index(drop=True)
    n    = min(len(baseline), 3)
    runs = [f"Run {i+1}" for i in range(n)]
    accs = baseline["accuracy"].values[:n]
    f1s  = baseline["f1_macro"].values[:n]

    x     = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))

    bars1 = ax.bar(x - width / 2, accs, width, color="#4C72B0", alpha=0.85, label="Accuracy")
    bars2 = ax.bar(x + width / 2, f1s,  width, color="#DD8452", alpha=0.85, label="F1-macro")

    for bar, v in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # Đường trung bình
    ax.axhline(accs.mean(), color="#4C72B0", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(f1s.mean(),  color="#DD8452", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(n - 0.05, accs.mean() + 0.003, f"μ={accs.mean():.4f}",
            ha="right", fontsize=8, color="#4C72B0")
    ax.text(n - 0.05, f1s.mean() - 0.012, f"μ={f1s.mean():.4f}",
            ha="right", fontsize=8, color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("UC2 PhoBERT — Accuracy & F1-macro (Metaflow, 3 baseline runs)", fontsize=11, pad=12)
    ax.set_ylim(0.60, 0.90)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "uc2_baseline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# UC2 — Biểu đồ 5: TC2 Config Sweep
# ─────────────────────────────────────────────
def plot_uc2_tc2(df):
    # TC2 sweep: training_seed=42 (default seed, không phải 43/44/45 của baseline)
    # Gồm 3 configs: lr=1e-5/batch=2, lr=2e-5/batch=2, lr=3e-5/batch=4
    if "training_seed" in df.columns:
        sweep = df[df["training_seed"] == 42].copy()
    else:
        # Fallback: lấy runs không phải baseline (lr≠2e-5 hoặc batch≠2)
        sweep = df[~((df["lr"] == 2e-5) & (df["batch_size"] == 2))].copy()

    if sweep.empty:
        print("  [UC2 TC2] Chưa có sweep runs, bỏ qua biểu đồ.")
        return

    # Sort theo lr tăng dần để hiển thị đúng thứ tự TC2
    sweep = sweep.sort_values("lr", ascending=True).reset_index(drop=True)

    # Baseline mean từ runs đã fix seed
    if "training_seed" in df.columns:
        bdf = df[df["training_seed"].isin([43, 44, 45])]
    else:
        bdf = df[(df["lr"] == 2e-5) & (df["batch_size"] == 2)]
    baseline_mean_acc = bdf["accuracy"].mean() if not bdf.empty else None
    baseline_mean_f1  = bdf["f1_macro"].mean()  if not bdf.empty else None

    labels = [f"lr={r['lr']:.0e}\nbatch={int(r['batch_size'])}" for _, r in sweep.iterrows()]
    accs   = sweep["accuracy"].values
    f1s    = sweep["f1_macro"].values

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width / 2, accs, width, color="#4C72B0", alpha=0.85, label="Accuracy")
    bars2 = ax.bar(x + width / 2, f1s,  width, color="#DD8452", alpha=0.85, label="F1-macro")

    ax.axhline(baseline_mean_acc, color="#4C72B0", linestyle="--", linewidth=1.3, alpha=0.7,
               label=f"Baseline acc μ={baseline_mean_acc:.4f}")
    ax.axhline(baseline_mean_f1,  color="#DD8452", linestyle="--", linewidth=1.3, alpha=0.7,
               label=f"Baseline F1 μ={baseline_mean_f1:.4f}")

    for bar, v in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("UC2 PhoBERT — TC2 Config Sweep (Metaflow)", fontsize=12, pad=12)
    ax.set_ylim(0.55, 0.95)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "uc2_tc2_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# DAG graph (cần graphviz)
# ─────────────────────────────────────────────
def export_dag(flow_script, output_name):
    """Xuất DAG PNG từ Metaflow output-dot."""
    dot_cmd = ["python3", flow_script, "output-dot"]
    png_path = os.path.join(PLOTS_DIR, output_name)
    try:
        dot_output = subprocess.check_output(dot_cmd, stderr=subprocess.DEVNULL)
        result = subprocess.run(
            ["dot", "-Tpng", "-o", png_path],
            input=dot_output, capture_output=True
        )
        if result.returncode == 0:
            print(f"  Saved: {png_path}")
        else:
            print(f"  [DAG] graphviz lỗi: {result.stderr.decode()[:100]}")
    except FileNotFoundError:
        print("  [DAG] Chưa cài graphviz. Chạy: sudo apt install graphviz")
    except Exception as e:
        print(f"  [DAG] Lỗi: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Metaflow — Vẽ biểu đồ kết quả")
    print("=" * 55)

    # ── UC1 ──
    print("\n[UC1] Đọc dữ liệu MNISTFlow...")
    df_uc1 = load_uc1_runs()
    if df_uc1.empty:
        print("  Chưa có runs UC1. Chạy train_uc1_metaflow.py trước.")
    else:
        print(f"  Tổng: {len(df_uc1)} runs")
        plot_uc1_accuracy(df_uc1)
        plot_uc1_pipeline_time(df_uc1)
        plot_uc1_tc2(df_uc1)

    # ── UC2 ──
    print("\n[UC2] Đọc dữ liệu PhoBERTSentimentFlow...")
    df_uc2 = load_uc2_runs()
    if df_uc2.empty:
        print("  Chưa có runs UC2. Chạy train_uc2_metaflow.py trước.")
    else:
        print(f"  Tổng: {len(df_uc2)} runs")
        plot_uc2_baseline(df_uc2)
        plot_uc2_tc2(df_uc2)

    # ── DAG ──
    print("\n[DAG] Xuất sơ đồ flow...")
    export_dag(os.path.join(UC1_DIR, "train_uc1_metaflow.py"), "uc1_dag.png")
    export_dag(os.path.join(UC2_DIR, "train_uc2_metaflow.py"), "uc2_dag.png")

    print("\n" + "=" * 55)
    print(f"  Xong! Ảnh lưu tại: {PLOTS_DIR}/")
    print("=" * 55)
