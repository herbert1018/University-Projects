import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import sys
import traceback
from image_RB import process_folder as process_rb, process_image as process_rb_image
from auto_assort import process_folder as process_assort
from auto_mask import process_folder as process_mask
from batch_crop import process_folder as process_crop
from image_stats import ImageStats
import os

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

class ImageProcessorGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("圖片處理工具")
        self.window.geometry("800x600")
        
        # 初始化進度條相關屬性
        self.progress_window = None
        self.progress_bar = None
        self.status_label = None
        self.current_process = None
        self.total_files = 0
        self.processed_files = 0
        
        # 設定主題樣式
        self.style = ttk.Style()
        self.style.configure('Custom.TButton',
                           padding=10,
                           font=('微軟正黑體', 12, 'bold'),  # 加粗字體
                           width=20,
                           relief='raised',
                           borderwidth=3)
        
        # 設定按鈕懸停效果
        self.style.map('Custom.TButton',
                      foreground=[('pressed', '#000066'), 
                                ('active', '#000099'),
                                ('!disabled', '#000033')],
                      background=[('pressed', '#99CCFF'), 
                                ('active', '#CCE5FF'),
                                ('!disabled', '#E6F3FF')],
                      relief=[('pressed', 'sunken'),
                             ('!pressed', 'raised')],
                      borderwidth=[('pressed', 2),
                                 ('!pressed', 3)])

        # 設定狀態標籤的樣式
        self.style.configure('success.TLabel', 
                           foreground='#2E7D32',  # 深綠色
                           font=('微軟正黑體', 12))
        self.style.configure('error.TLabel', 
                           foreground='#C62828',  # 深紅色
                           font=('微軟正黑體', 12))
        self.style.configure('info.TLabel', 
                           foreground='#1565C0',  # 深藍色
                           font=('微軟正黑體', 12))
        
        self.message_timer = None  # 用於追蹤訊息計時器

        self.setup_ui()
        self.current_dialog = None  # 新增追蹤目前的對話框

        # 為每個按鈕設定獨特的樣式
        button_styles = {
            'crop.TButton': {
                'background': '#81C784',  # 綠色
                'hover': '#66BB6A',
                'pressed': '#4CAF50'
            },
            'rb.TButton': {
                'background': '#4FC3F7',  # 天藍色
                'hover': '#29B6F6',
                'pressed': '#03A9F4'
            },
            'assort.TButton': {
                'background': '#BA68C8',  # 紫色
                'hover': '#AB47BC',
                'pressed': '#9C27B0'
            },
            'mask.TButton': {
                'background': '#FFB74D',  # 橙色
                'hover': '#FFA726',
                'pressed': '#FF9800'
            }
        }
        
        # 配置每個按鈕的樣式
        for style_name, colors in button_styles.items():
            self.style.configure(style_name,
                               padding=10,
                               font=('微軟正黑體', 12, 'bold'),
                               width=20,
                               background=colors['background'])
            self.style.map(style_name,
                         background=[('active', colors['hover']),
                                   ('pressed', colors['pressed'])],
                         relief=[('pressed', 'sunken')])

        self.stats = ImageStats()
    def setup_ui(self):
        # 建立主框架並設定背景
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 設定視窗可調整大小
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 添加標題
        title_label = ttk.Label(main_frame, 
                              text="圖片處理工具", 
                              font=('微軟正黑體', 24, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # 建立按鈕框架
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky='ew')
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        # 調整按鈕順序：批次裁切、藍線去除、遮罩生成、圖片分類
        buttons = [
            ("批次裁切", self.run_batch_crop, 'crop.TButton'),
            ("藍線去除", self.run_rb_process, 'rb.TButton'),
            ("遮罩生成", self.run_mask_generator, 'mask.TButton'),
            ("圖片分類", self.run_assort, 'assort.TButton'),
        ]

        for i, (text, command, style) in enumerate(buttons):
            btn = ttk.Button(button_frame, 
                           text=text,
                           command=command,
                           style=style)
            # 置中顯示，維持長寬
            btn.grid(row=i, column=0, columnspan=2, pady=12, padx=245, sticky='ew')

        # 統計報告與使用說明按鈕
        help_btn = ttk.Button(button_frame, text="使用說明", command=self.run_help, style='Custom.TButton')
        help_btn.grid(row=len(buttons)+1, column=0, pady=(12, 0), padx=(25, 5), sticky='e')

        stats_btn = ttk.Button(button_frame, text="統計報告", command=self.run_stats, style='Custom.TButton')
        stats_btn.grid(row=len(buttons)+1, column=1, pady=(12, 0), padx=(5, 25), sticky='w')

        # 在按鈕框架下方添加進度條和狀態標籤
        status_frame = ttk.Frame(main_frame, padding="10")
        status_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky='ew')
        
        # 修改狀態標籤的建立方式
        self.status_label = ttk.Label(status_frame, text="", 
                                    style='info.TLabel')
        self.status_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(status_frame, length=300, mode='indeterminate')
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()  # 初始時隱藏進度條

    def create_progress_window(self):
        if self.progress_bar:
            self.progress_bar.stop()
            self.progress_bar["mode"] = "determinate"  # 改為確定模式
            self.progress_bar["value"] = 0
        self.status_label.config(text="正在處理圖片...")
        self.progress_bar.pack(pady=5)
        self.window.update()

    def update_progress(self):
        """更新進度條"""
        if self.total_files > 0:
            progress = (self.processed_files / self.total_files) * 100
            self.progress_bar["value"] = progress
            self.status_label.config(text=f"處理進度: {progress:.1f}% ({self.processed_files}/{self.total_files})")
            self.window.update()

    def close_progress_window(self):
        if self.progress_bar:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()  # 隱藏進度條
            self.status_label.config(text="")  # 清空文字
            self.window.update()

    def get_project_root(self):
        """取得專案根目錄"""
        current_file = Path(__file__)
        return current_file.parent.parent

    def show_status_message(self, message, style='info'):
        """在狀態列顯示訊息"""
        # 如果有執行中的計時器，先取消它
        if self.message_timer:
            self.window.after_cancel(self.message_timer)
        
        # 更新訊息和樣式
        self.status_label.config(text=message, style=f'{style}.TLabel')
        
        # 設定5秒後清空文字
        def reset_status():
            self.status_label.config(text="", style='info.TLabel')
            self.message_timer = None
            
        self.message_timer = self.window.after(5000, reset_status)

    def process_with_progress(self, process_func, folder, process_subdirs, output_dir):
        try:
            # 計算總檔案數
            self.total_files = sum(1 for path in folder.rglob("*") 
                                 if path.is_file() 
                                 and path.suffix.lower() in SUPPORTED_FORMATS
                                 and not any('移出圖片' in p.parts for p in path.parents))
            self.processed_files = 0
            
            self.create_progress_window()
            
            def update_callback():
                self.processed_files += 1
                self.update_progress()
            
            # 根據不同的處理函數使用不同的調用方式
            if process_func == process_rb:
                base_output_dir = folder.parent / output_dir
                base_output_dir.mkdir(parents=True, exist_ok=True)
                result = process_func(folder, base_output_dir, update_callback)
            elif process_func == process_crop:
                result = process_func(folder, update_callback)
            else:
                result = process_func(folder, update_callback)
            
            self.close_progress_window()
            
            # 處理回傳結果
            if result:
                if isinstance(result, tuple) and len(result) == 3:
                    success, failure, error_files = result
                    if error_files:
                        error_msg = f"失敗: {len(error_files)}張"
                        self.show_status_message(error_msg, 'error')
                else:
                    success, failure = result
                
                if failure == 0:
                    status_text = f"✅ 完成！成功處理 {success} 張圖片"
                    self.show_status_message(status_text, 'success')
                else:
                    status_text = f"⚠️ 完成 (成功:{success} 失敗:{failure})"
                    style = 'error' if failure > success else 'info'
                    self.show_status_message(status_text, style)
            
            return result
            
        except Exception as e:
            self.close_progress_window()
            self.show_status_message("錯誤", 'error')
            traceback.print_exc()
            return None

    def select_folder(self):
        # 使用 filedialog.askdirectory，若按叉叉會回傳空字串
        folder_path = filedialog.askdirectory(title="選擇要處理的圖片資料夾")
        if not folder_path or folder_path == "":
            return None
        folder = Path(folder_path)
        if not folder.exists():
            return None

        # 使用自訂對話框，允許按叉叉直接關閉
        confirm_dialog = tk.Toplevel(self.window)
        confirm_dialog.title("確認")
        confirm_dialog.grab_set()
        confirm_dialog.resizable(False, False)
        confirm_dialog.geometry("+%d+%d" % (self.window.winfo_rootx()+200, self.window.winfo_rooty()+200))

        result = {"value": None}

        def on_yes():
            result["value"] = True
            confirm_dialog.destroy()

        def on_no():
            result["value"] = False
            confirm_dialog.destroy()

        def on_close():
            result["value"] = None
            confirm_dialog.destroy()

        label = ttk.Label(confirm_dialog, text="是否處理所有子資料夾中的圖片？\n\n選擇「是」將處理所有子資料夾\n選擇「否」只處理選擇的資料夾", padding=20)
        label.pack()
        btn_frame = ttk.Frame(confirm_dialog)
        btn_frame.pack(pady=10)
        yes_btn = ttk.Button(btn_frame, text="是", command=on_yes)
        yes_btn.pack(side=tk.LEFT, padx=10)
        no_btn = ttk.Button(btn_frame, text="否", command=on_no)
        no_btn.pack(side=tk.LEFT, padx=10)
        confirm_dialog.protocol("WM_DELETE_WINDOW", on_close)
        confirm_dialog.wait_window()

        if result["value"] is None:
            return None
        return (folder, result["value"])

    def run_rb_process(self):
        result = self.select_folder()
        if result:
            folder, process_subdirs = result
            self.process_with_progress(process_rb, folder, process_subdirs, "IMG_RB")

    def run_assort(self):
        result = self.select_folder()
        if result:
            folder, process_subdirs = result
            results = self.process_with_progress(process_assort, folder, process_subdirs, "IMG_train")
            if results:
                # 主視窗顯示簡單訊息
                self.status_label.config(text="自動隨機分類完成!")
                
                # 彈出視窗顯示詳細結果
                method1_summary, method2_summary = results
                summary_message = "方法一分配結果:\n" + "\n".join(method1_summary) + "\n\n"
                summary_message += "方法二分配結果:\n" + "\n".join(method2_summary)
                messagebox.showinfo("分類結果統計", summary_message)

    def run_mask_generator(self):
        result = self.select_folder()
        if result:
            folder, process_subdirs = result
            self.process_with_progress(process_mask, folder, process_subdirs, "IMG_Mask")

    def run_batch_crop(self):
        result = self.select_folder()
        if result:
            folder, process_subdirs = result
            self.process_with_progress(process_crop, folder, process_subdirs, "IMG_Cut")

    def run_stats(self):
        """執行統計功能（自動分析IMG資料夾，兩兩並排顯示）"""
        self.stats.run_stats_gui(self.window)

    def run_help(self):
        """顯示使用說明"""
        self.stats.show_help_dialog(self.window)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ImageProcessorGUI()
    app.run()
