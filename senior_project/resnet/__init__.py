import sys
import os
from Resnet import ResNetModel, train_model, get_dataloader, save_model, load_model
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.optim as optim
from test_window import TestWindow
from torch.utils.data import get_worker_info

# 初始設定
sys.stdout.reconfigure(line_buffering=True)
torch.cuda.empty_cache()

# 將目前目錄加入系統路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

if __name__ == '__main__':
    # 設定基礎路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    dataset_path = os.path.join(PROJECT_ROOT, "IMG", "IMG_train", "method2")
    test_images_path = os.path.join(dataset_path, "test")
    model_dir = os.path.join(PROJECT_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "Resnet_model.pth")

    # 設定超參數
    batch_size = 8
    learning_rate = 3e-5
    weight_decay = 8e-4
    num_epochs = 100
    patience_epochs = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")

    # 初始化模型和優化器
    try:
        # 五類分類
        model = ResNetModel(num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print("Model and optimizer initialized successfully.")
        print(f"模型類型: {type(model).__name__}")
        print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Error initializing model: {e}")

    # 初始化資料加載器
    try:
        train_loader, test_loader, val_loader = get_dataloader(dataset_path, batch_size)
        print("Data loaders created successfully.")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        train_loader = test_loader = val_loader = None

    # GUI 相關變數
    last_message_timer = None

# 新增歷史紀錄
val_loss_history = []
val_acc_history = []
val_f1_history = []
train_loss_history = []

def load_model_with_check(model, path):
    """檢查模型檔案是否存在並加載模型"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型檔案不存在: {path}")
    return load_model(model, path)

def create_gui_elements():
    """創建GUI元素"""
    global progress_label, progress_bar, fig_loss, ax_loss, canvas_loss
    global fig_metric, ax_metric, canvas_metric
    global num_epochs_var
    global line_train_loss, line_val_loss, line_val_acc, line_val_f1
    global canvas_loss, canvas_metric

    # 創建左側按鈕區域
    left_frame = tk.Frame(root, bg="#e0e0e0", width=350)
    left_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
    left_frame.pack_propagate(False)

    # 創建按鈕容器
    button_container = tk.Frame(left_frame, bg="#e0e0e0")
    button_container.place(relx=0.5, rely=0.5, anchor="center", width=300)

    # 新增回合數設定區域
    epoch_frame = tk.Frame(button_container, bg="#e0e0e0")
    epoch_frame.pack(pady=5)
    
    tk.Label(epoch_frame, text="訓練回合數:", bg="#e0e0e0", font=("Arial", 12)).pack(side=tk.LEFT)
    num_epochs_var = tk.StringVar(value=str(num_epochs))
    epoch_entry = tk.Entry(epoch_frame, textvariable=num_epochs_var, width=5, font=("Arial", 12))
    epoch_entry.pack(side=tk.LEFT, padx=5)

    # 統一按鈕樣式
    button_width = 300
    button_style = {
        "width": 20,
        "height": 2,
        "font": ("Arial", 14),
        "relief": tk.RAISED,
        "bd": 3,
        "padx": 10
    }

    # 資料集資訊按鈕
    dataset_button = tk.Button(button_container, text="目前資料集", command=show_dataset_info,
                             bg="lightsteelblue", **button_style)
    dataset_button.pack(pady=15, ipadx=(button_width - button_style["width"]*10)//2)

    # 訓練按鈕
    global train_button
    train_button = tk.Button(button_container, text="訓練模型", command=train, 
                           bg="lightblue", **button_style)
    train_button.pack(pady=15, ipadx=(button_width - button_style["width"]*10)//2)

    # 測試按鈕
    test_button = tk.Button(button_container, text="測試模型", command=test, 
                          bg="lightgreen", **button_style)
    test_button.pack(pady=15, ipadx=(button_width - button_style["width"]*10)//2)

    # 刪除舊模型按鈕
    retrain_button = tk.Button(button_container, text="刪除舊模型", command=reset_model, 
                             bg="lightyellow", **button_style)
    retrain_button.pack(pady=15, ipadx=(button_width - button_style["width"]*10)//2)

    # 結果顯示標籤
    global result_label
    result_label = tk.Label(button_container, text="", bg="#e0e0e0",
                           font=("Arial", 14), fg="black")
    result_label.pack(pady=20)

    # 離開按鈕
    exit_button = tk.Button(button_container, text="離開", command=root.quit, 
                          bg="lightcoral", **button_style)
    exit_button.pack(pady=15, ipadx=(button_width - button_style["width"]*10)//2)

    # 創建右側進度顯示區域
    right_frame = tk.Frame(root, bg="#f0f0f0")
    right_frame.pack(side=tk.RIGHT, padx=20, fill=tk.BOTH, expand=True)

    # 訓練進度標籤
    progress_label = tk.Label(right_frame, text="", bg="#f0f0f0", font=("Arial", 12))
    progress_label.pack(pady=10)

    # 建立進度條容器以控制寬度
    progress_container = tk.Frame(right_frame, bg="#f0f0f0")
    progress_container.pack(fill=tk.X, pady=10, padx=20)

    # 訓練進度條
    progress_bar = ttk.Progressbar(progress_container, orient="horizontal", mode="determinate")
    progress_bar.pack(fill=tk.X)

    # 圖表區域
    chart_frame = tk.Frame(right_frame, bg="#f0f0f0")
    chart_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 10))

    # Loss 曲線圖 (train loss: 藍實線, val loss: 紅虛線)
    fig_loss = Figure(figsize=(6, 3))
    ax_loss = fig_loss.add_subplot(111)
    ax_loss.set_title('Train & Val Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_xlim(0, num_epochs+1)
    line_train_loss, = ax_loss.plot([], [], 'b-', label='Train Loss')
    line_val_loss, = ax_loss.plot([], [], 'r--', label='Val Loss')
    ax_loss.legend()
    canvas_loss = FigureCanvasTkAgg(fig_loss, master=chart_frame)
    canvas_loss.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    # Metric 曲線圖 (val acc: 綠線, val F1: 紫線)
    fig_metric = Figure(figsize=(6, 3))
    ax_metric = fig_metric.add_subplot(111)
    ax_metric.set_title('Val Acc & F1')
    ax_metric.set_xlabel('Epoch')
    ax_metric.set_ylabel('Score')
    ax_metric.set_ylim(0, 1)
    ax_metric.set_xlim(0, num_epochs+1)
    line_val_acc, = ax_metric.plot([], [], 'g-', label='Val Acc')
    line_val_f1, = ax_metric.plot([], [], 'm--', label='Val F1')
    ax_metric.legend()
    canvas_metric = FigureCanvasTkAgg(fig_metric, master=chart_frame)
    canvas_metric.get_tk_widget().grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

    # 設定 chart_frame 自適應
    chart_frame.grid_rowconfigure(0, weight=1)
    chart_frame.grid_rowconfigure(1, weight=1)
    chart_frame.grid_columnconfigure(0, weight=1)

def update_training_progress(epoch, total_epochs, batch, total_batches, 
                           train_loss=None, val_loss=None, val_acc=None, val_f1=None, update_history=True):
    """更新訓練進度，只更新線條數據，不清除 axes"""
    global train_loss_history, val_loss_history, val_acc_history, val_f1_history
    global line_train_loss, line_val_loss, line_val_acc, line_val_f1
    global canvas_loss, canvas_metric

    # 只在主線程更新 GUI
    if get_worker_info() is not None:
        return

    # 更新進度文字
    progress_text = f"Epoch {epoch}/{total_epochs}, Batch {batch}/{total_batches}"
    if train_loss is not None:
        progress_text += f", Train Loss: {train_loss:.4f}"
    if val_loss is not None:
        progress_text += f", Val Loss: {val_loss:.4f}"
    if val_acc is not None:
        progress_text += f", Val Acc: {val_acc:.4f}"

    progress_label.config(text=progress_text)
    progress_bar["value"] = (epoch - 1 + batch / total_batches) / total_epochs * 100

    # 當一個 epoch 完成時，更新歷史數據和圖表
    if batch == total_batches and update_history:
        # 將當前 epoch 的數據添加到歷史記錄中
        if train_loss is not None:
            train_loss_history.append(train_loss)
        if val_loss is not None:
            val_loss_history.append(val_loss)
        if val_acc is not None:
            val_acc_history.append(val_acc)
        if val_f1 is not None:
            val_f1_history.append(val_f1)

        # 更新 Loss 曲線
        if train_loss_history:
            x_train = list(range(1, len(train_loss_history)+1))
            line_train_loss.set_data(x_train, train_loss_history)
        if val_loss_history:
            x_val = list(range(1, len(val_loss_history)+1))
            line_val_loss.set_data(x_val, val_loss_history)

        # 自動調整 y 軸範圍
        ax_loss.relim()
        ax_loss.autoscale_view()
        ax_loss.set_xlim(0, total_epochs+1)
        canvas_loss.draw()

        # 更新 Metric 曲線
        if val_acc_history:
            x_acc = list(range(1, len(val_acc_history)+1))
            line_val_acc.set_data(x_acc, val_acc_history)
        if val_f1_history:
            x_f1 = list(range(1, len(val_f1_history)+1))
            line_val_f1.set_data(x_f1, val_f1_history)

        # 固定 y 軸 0~1
        ax_metric.set_ylim(0, 1)
        ax_metric.set_xlim(0, total_epochs+1)
        canvas_metric.draw()
        
    root.update()

def show_dataset_info():
    """顯示目前資料集的相關資訊"""
    try:
        # 獲取訓練集資訊
        train_classes = train_loader.dataset.classes
        class_to_idx = train_loader.dataset.class_to_idx
        total_train = len(train_loader.dataset)
        total_test = len(test_loader.dataset)
        total_val = len(val_loader.dataset)
        
        # 計算每個類別的樣本數
        class_counts = {cls: 0 for cls in train_classes}
        for _, label in train_loader.dataset.samples:
            class_counts[train_classes[label]] += 1
        
        # 準備顯示訊息
        info = f"資料集分布:\n"
        info += f"總圖片數量: {total_train + total_test + total_val}\n"
        info += f"訓練集: {total_train} 張\n"
        info += f"驗證集: {total_val} 張\n"
        info += f"測試集: {total_test} 張\n\n"
        info += "各類別圖片數量:\n"
        for cls, count in class_counts.items():
            info += f"{cls}: {count} 張\n"
        
        messagebox.showinfo("資料集資訊", info)
    except Exception as e:
        show_temp_message(f"無法獲取資料集資訊: {e}", "red")

def clear_result_message():
    """清除結果訊息"""
    result_label.config(text="")

def show_temp_message(message, color="black"):
    """顯示暫時訊息，3秒後自動消失，若有新訊息則取消舊的計時器"""
    global last_message_timer
    
    # 如果存在舊的計時器，取消它
    if last_message_timer is not None:
        root.after_cancel(last_message_timer)
    
    # 顯示新訊息
    result_label.config(text=message, fg=color)
    
    # 設定新的計時器
    last_message_timer = root.after(3000, clear_result_message)

def reset_model():
    """刪除舊模型"""
    # 顯示確認對話框
    confirm = messagebox.askokcancel("確認刪除", "確定要刪除舊模型嗎？")
    if not confirm:  # 按 X 或取消
        show_temp_message("已取消刪除", "blue")
        return

    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        show_temp_message("舊模型已刪除", "blue")
    else:
        show_temp_message("沒有舊模型可刪除", "red")

def train():
    """執行訓練步驟"""
    global model, num_epochs, optimizer
    global train_loss_history, val_loss_history, val_acc_history, val_f1_history
    progress_label.config(text=" ", fg="black")
    print("Training the model...")
    print(f"lr={learning_rate}, weight_decay={weight_decay}, batch_size={batch_size}, epochs={num_epochs}, patience={patience_epochs}")

    # 檢查是否已存在模型
    if os.path.exists(model_save_path):
        show_temp_message("已有Resnet模型存在！", "red")
        return

    # 更新回合數
    try:
        num_epochs = int(num_epochs_var.get())
        if num_epochs <= 0:
            raise ValueError("回合數必須大於0")
    except ValueError as e:
        show_temp_message(f"回合數設定錯誤: {e}", "red")
        return

    train_button.config(state=tk.DISABLED)
    
    # 重置圖表與歷史紀錄
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    val_f1_history = []
    
    # 重新創建線條對象
    global line_train_loss, line_val_loss, line_val_acc, line_val_f1
    
    ax_loss.clear()
    ax_loss.set_title('Train & Val Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_xlim(0, num_epochs+1)
    line_train_loss, = ax_loss.plot([], [], 'b-', label='Train Loss')
    line_val_loss, = ax_loss.plot([], [], 'r--', label='Val Loss')
    ax_loss.legend()
    
    ax_metric.clear()
    ax_metric.set_title('Val Acc & F1')
    ax_metric.set_xlabel('Epoch')
    ax_metric.set_ylabel('Score')
    ax_metric.set_ylim(0, 1)
    ax_metric.set_xlim(0, num_epochs+1)
    line_val_acc, = ax_metric.plot([], [], 'g-', label='Val Acc')
    line_val_f1, = ax_metric.plot([], [], 'm--', label='Val F1')
    ax_metric.legend()
    
    canvas_loss.draw()
    canvas_metric.draw()
    
    try:
        # 訓練模型並接收歷史數據
        model, best_val_acc, minutes, seconds, val_loss_hist, val_acc_hist, val_f1_hist, train_loss_hist = train_model(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            num_epochs,
            update_training_progress,
            val_loader=val_loader,
            patience=patience_epochs
        )
        
        # 更新全域變數（雖然圖表已經在訓練過程中更新了）
        train_loss_history = train_loss_hist
        val_loss_history = val_loss_hist
        val_acc_history = val_acc_hist
        val_f1_history = val_f1_hist
        
        save_model(model, model_save_path)
        completion_text = f"訓練完成！花費時間: {minutes}分{seconds}秒，best val acc: {best_val_acc:.4f}"
        progress_label.config(text=completion_text, fg="green")
        
    except Exception as e:
        show_temp_message(f"訓練失敗: {e}", "red")
        print(f"Training error: {str(e)}")
    finally:
        train_button.config(state=tk.NORMAL)
        progress_bar["value"] = 100

def test():
    """執行測試步驟"""
    print("Testing the model...")
    global model
    try:
        model_path = os.path.join(model_dir, "Resnet_model.pth")
        
        # 創建新的模型實例用於載入
        model = ResNetModel(num_classes=5).to(device)
        model = load_model_with_check(model, model_path)
        model.eval()
        
        # 使用 method2 的 test 資料夾
        if not os.path.exists(test_images_path):
            os.makedirs(test_images_path, exist_ok=True)
            print(f"已創建測試圖片目錄: {test_images_path}")
            
        # 傳入測試圖片目錄路徑
        with torch.no_grad():
            test_window = TestWindow(model, device, test_images_path)
        
    except Exception as e:
        show_temp_message(f"測試失敗: {e}", "red")
        print(f"Testing error: {str(e)}")

def main():
    """建立 tkinter 使用者介面"""
    global root
    root = tk.Tk()
    root.title("Resnet 訓練介面")
    root.geometry("1000x1000")
    root.configure(bg="#f0f0f0")

    create_gui_elements()
    root.mainloop()

if __name__ == "__main__":
    main()