import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import shutil
import random
from PIL import Image
import traceback

SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def k_fold_split(input_dir, k=5, output_dir=None):
    """對資料夾中所有圖片做 K-Fold 分割，並對應遮罩（忽略子資料夾類別）"""
    if output_dir is None:
        output_dir = input_dir.parent / f"IMG_kfold_{k}"
    output_dir.mkdir(exist_ok=True)

    folds_dir = output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)

    # 找遮罩資料夾
    mask_dir = input_dir.parent / "IMG_Mask"
    mask_files = {}
    if mask_dir.exists():
        for mask_file in mask_dir.rglob("*.*"):
            if mask_file.suffix.lower() in SUPPORTED_FORMATS:
                mask_files[mask_file.stem] = mask_file
                if mask_file.stem.endswith("_mask"):
                    mask_files[mask_file.stem[:-5]] = mask_file

    # 收集所有圖片
    all_files = [f for f in input_dir.rglob("*.*") if f.suffix.lower() in SUPPORTED_FORMATS]
    if not all_files:
        return ["未發現任何有效圖片"]
    
    random.shuffle(all_files)
    fold_size = len(all_files) // k

    for fold in range(k):
        img_fold_path = folds_dir / f"fold_{fold+1}" / "images"
        mask_fold_path = folds_dir / f"fold_{fold+1}" / "masks"
        img_fold_path.mkdir(parents=True, exist_ok=True)
        mask_fold_path.mkdir(parents=True, exist_ok=True)

        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < k - 1 else len(all_files)

        for f in all_files[start_idx:end_idx]:
            shutil.copy2(f, img_fold_path / f.name)
            mask_file = mask_files.get(f.stem)
            if mask_file:
                mask_dest = mask_fold_path / f"{f.stem}.png"
                if mask_file.suffix.lower() != '.png':
                    Image.open(mask_file).save(mask_dest, 'PNG')
                else:
                    shutil.copy2(mask_file, mask_dest)
            else:
                print(f"警告: 找不到圖片 {f.name} 的遮罩")

    return [f"共 {len(all_files)} 張圖片分成 {k} 個 fold"]

def generate_train_val_test_packages(base_output_dir, k=5):
    """依照 folds 資料夾生成 dataset_x (train/val/test) 結構"""
    folds_dir = base_output_dir / "folds"
    if not folds_dir.exists():
        raise FileNotFoundError(f"找不到 folds 目錄: {folds_dir}")

    fold_paths = [folds_dir / f"fold_{i+1}" for i in range(k)]

    for i in range(k):
        dataset_dir = base_output_dir / f"dataset_{i+1}"
        for split in ["train", "val", "test"]:
            (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "masks").mkdir(parents=True, exist_ok=True)

        # val/test fold
        val_test_fold = fold_paths[i]
        val_img_dir = val_test_fold / "images"
        val_mask_dir = val_test_fold / "masks"

        for img_file in val_img_dir.glob("*.*"):
            shutil.copy2(img_file, dataset_dir / "val" / "images" / img_file.name)
            shutil.copy2(img_file, dataset_dir / "test" / "images" / img_file.name)

        for mask_file in val_mask_dir.glob("*.*"):
            shutil.copy2(mask_file, dataset_dir / "val" / "masks" / mask_file.name)
            shutil.copy2(mask_file, dataset_dir / "test" / "masks" / mask_file.name)

        # train folds
        for j, fold in enumerate(fold_paths):
            if j == i:
                continue
            train_img_dir = fold / "images"
            train_mask_dir = fold / "masks"

            for img_file in train_img_dir.glob("*.*"):
                shutil.copy2(img_file, dataset_dir / "train" / "images" / img_file.name)
            for mask_file in train_mask_dir.glob("*.*"):
                shutil.copy2(mask_file, dataset_dir / "train" / "masks" / mask_file.name)

        print(f"完成 dataset_{i+1}: train={len(list((dataset_dir/'train'/'images').glob('*')))} "
              f"val={len(list((dataset_dir/'val'/'images').glob('*')))} "
              f"test={len(list((dataset_dir/'test'/'images').glob('*')))}")

class KFoldGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("K-Fold 圖片分類工具")
        self.window.geometry("600x400")

        self.selected_folder = None
        self.k_value = tk.IntVar(value=5)

        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="選擇資料夾", font=('微軟正黑體', 12, 'bold')).grid(row=0, column=0, pady=10, sticky='w')
        self.folder_entry = ttk.Entry(frame, width=40)
        self.folder_entry.grid(row=0, column=1, pady=10, padx=10)
        ttk.Button(frame, text="瀏覽", command=self.browse_folder).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="K 值 (fold 數量)", font=('微軟正黑體', 12, 'bold')).grid(row=1, column=0, pady=10, sticky='w')
        ttk.Entry(frame, textvariable=self.k_value, width=10).grid(row=1, column=1, pady=10, sticky='w')

        self.status_label = ttk.Label(frame, text="", font=('微軟正黑體', 12))
        self.status_label.grid(row=2, column=0, columnspan=3, pady=10)

        ttk.Button(frame, text="開始 K-Fold 分割", command=self.run_kfold).grid(row=3, column=0, columnspan=3, pady=10)
        ttk.Button(frame, text="生成 train/val/test 資料集", command=self.run_generate_dataset).grid(row=4, column=0, columnspan=3, pady=10)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="選擇資料夾")
        if folder:
            self.selected_folder = Path(folder)
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)

    def run_kfold(self):
        if not self.selected_folder or not self.selected_folder.exists():
            messagebox.showerror("錯誤", "請先選擇有效資料夾")
            return

        k = self.k_value.get()
        if k <= 1:
            messagebox.showerror("錯誤", "K 值必須大於 1")
            return

        self.status_label.config(text="正在執行 K-Fold 分割...")
        self.window.update()

        try:
            results = k_fold_split(self.selected_folder, k)
            summary = "\n".join(results)
            messagebox.showinfo("完成", f"K-Fold 分割完成!\n\n{summary}")
            self.status_label.config(text="完成!")
        except Exception as e:
            self.status_label.config(text="分割失敗")
            traceback.print_exc()
            messagebox.showerror("錯誤", f"K-Fold 分割失敗:\n{e}")

    def run_generate_dataset(self):
        if not self.selected_folder or not self.selected_folder.exists():
            messagebox.showerror("錯誤", "請先選擇有效資料夾")
            return

        k = self.k_value.get()
        base_output_dir = self.selected_folder.parent / f"IMG_kfold_{k}"

        self.status_label.config(text="正在生成 train/val/test 資料集...")
        self.window.update()

        try:
            generate_train_val_test_packages(base_output_dir, k)
            messagebox.showinfo("完成", "成功生成 dataset_1 ~ dataset_{k} 資料集")
            self.status_label.config(text="完成!")
        except Exception as e:
            self.status_label.config(text="生成失敗")
            traceback.print_exc()
            messagebox.showerror("錯誤", f"生成 train/val/test 資料集失敗:\n{e}")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = KFoldGUI()
    app.run()
