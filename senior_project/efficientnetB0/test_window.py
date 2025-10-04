import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import torch
from torchvision import transforms
import glob
import torch.nn as nn

class TestWindow:
    def __init__(self, model, device, test_dir):
        self.window = tk.Toplevel()
        self.window.title("模型測試視窗")
        
        # 配置視窗行列
        self.window.grid_rowconfigure(0, weight=1)
        for i in range(3):
            self.window.grid_columnconfigure(i, weight=1)
        
        # 確保測試目錄存在
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"測試圖片目錄不存在: {test_dir}")
        
        self.model = model
        self.device = device
        self.test_dir = os.path.abspath(test_dir)
        self.correct_count = 0
        self.total_count = 0
        self.image_cache = {}  # 初始化 image_cache 字典
        self.selected_item = None  # 新增選取項目追蹤
        
        # GUI 相關常數設定
        self.GUI_CONFIG = {
            'window_size': "1150x900",  # 再次縮減視窗寬度
            'thumbnail_size': (100, 100),
            'display_size': (400, 400),  # 調整預覽圖為正方形
            'button_width': 15,
            'button_height': 2,
            'section_weights': [3, 4, 4],  # 調整區塊比例
            'right_frame_width': 420,  # 縮減右側區塊寬度
            'right_frame_padding': 10,  # 調整內邊距
            'preview_size': 400  # 新增預覽圖顯示尺寸
        }
        
        # 加入樣式設定
        self.style = ttk.Style()
        styles = {
            "ImageItem.TFrame": {
                "padding": 5,
                "background": "white"
            },
            "Selected.ImageItem.TFrame": {
                "padding": 5,
                "background": "lightblue"
            },
            "ImageItem.TLabel": {
                "background": "white"
            },
            "Selected.ImageItem.TLabel": {
                "background": "lightblue"
            },
            "Action.TButton": {
                "padding": (20, 10),
                "font": ("Arial", 12, "bold")
            },
            "Result.TLabel": {
                "font": ("Arial", 16, "bold"),  # 放大標題
                "foreground": "black",
                "padding": (5, 5)  # 縮減間距
            },
            "Probability.TLabel": {
                "font": ("Arial", 11),  # 調整機率文字
                "foreground": "#666666",
                "padding": (20, 1)  # 縮減行距
            }
        }
        
        for style_name, props in styles.items():
            self.style.configure(style_name, **props)
        
        # 修改圖片轉換，與 AlexNet.py 保持一致
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 新增類別名稱對照表與目錄名稱對照表
        self.class_names = {
            "0": "eGFR 15-30",
            "1": "eGFR 30-45",
            "2": "eGFR 45-60",
            "3": "eGFR 60-90",
            "4": "eGFR 大於90",
        }
        self.dir_to_class = {
            "eGFR 15-30": "0",
            "eGFR 30-45": "1",
            "eGFR 45-60": "2",
            "eGFR 60-90": "3",
            "eGFR 大於90": "4",
        }
        
        self.setup_gui()
        self.load_images()
    
    def setup_gui(self):
        # 基本視窗設定
        self.window.geometry(self.GUI_CONFIG['window_size'])
        self.window.resizable(True, True)
        
        # 建立主要框架並設定填滿
        frames = {}
        for idx, name in enumerate(['圖片列表', '功能表', '預測結果']):
            frame = ttk.LabelFrame(self.window, text=name)
            frame.grid(row=0, column=idx, padx=2, pady=2, sticky="nsew")
            frame.grid_rowconfigure(0, weight=1)  # 確保內容可以垂直擴展
            frame.grid_columnconfigure(0, weight=1)  # 確保內容可以水平擴展
            frames[name] = frame
            self.window.grid_columnconfigure(idx, weight=self.GUI_CONFIG['section_weights'][idx])
        
        self._setup_left_frame(frames['圖片列表'])
        self._setup_middle_frame(frames['功能表'])
        self._setup_right_frame(frames['預測結果'])

    def _setup_left_frame(self, frame):
        list_container = ttk.Frame(frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=2)  # 減少容器邊距
        
        self.canvas = tk.Canvas(list_container, width=350)  # 增加畫布寬度
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, width=350)
        
        self.scrollable_frame.bind( 
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all"),
                width=350  # 確保一致的寬度
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=350)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _setup_middle_frame(self, frame):
        # 建立一個容器來集中按鈕
        button_container = ttk.Frame(frame)
        # 改用 grid 布局並設定權重實現完美置中
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        button_container.grid(row=0, column=0)
        
        # 建立功能按鈕，增加內邊距
        buttons = [
            ("預測此圖片", self.run_next),
            ("全部預測", self.auto_run_all),
            ("重置狀態", self.reset_state)
        ]
        
        for idx, (text, command) in enumerate(buttons):
            btn = ttk.Button(
                button_container,
                text=text,
                command=command,
                style="Action.TButton",
                width=self.GUI_CONFIG['button_width']
            )
            btn.pack(pady=12)  # 稍微縮減按鈕間距

    def reset_state(self):
        self.correct_count = 0
        self.total_count = 0
        self.load_images()  # 重新載入所有圖片
        
        # 清空預覽和結果
        self.display_label.configure(image='')
        self.result_label.config(text='')
        self.entropy_label.config(text='')
        
        # 清空所有機率標籤
        for label in self.prob_labels:
            label.config(text='')
            label.configure(foreground="#666666")
        
        self.accuracy_label.config(text='')
        
        # 更新視窗
        self.window.update()

    def _setup_right_frame(self, frame):
        frame.grid_propagate(False)
        frame.configure(width=self.GUI_CONFIG['right_frame_width'])
        
        # 計算置中對齊的位置
        content_width = self.GUI_CONFIG['preview_size']
        side_padding = (self.GUI_CONFIG['right_frame_width'] - content_width) // 2
        
        # 配置右側框架
        frame.grid_rowconfigure(0, weight=0)  # 預覽圖區域
        frame.grid_rowconfigure(1, weight=1)  # 結果區域
        frame.grid_rowconfigure(2, weight=1)  # 機率資訊區域
        frame.grid_rowconfigure(3, weight=1)  # 準確率區域
        frame.grid_columnconfigure(0, weight=1)
        
        # 調整預覽圖區域
        display_frame = ttk.Frame(frame, width=content_width, height=content_width)
        display_frame.grid(row=0, column=0, padx=side_padding, pady=(15, 10))
        display_frame.grid_propagate(False)
        
        self.display_label = tk.Label(display_frame, bg="white")
        self.display_label.place(relwidth=1, relheight=1)
        
        # 文字對齊預覽圖左側 - 增加 padding 補償
        text_left_margin = side_padding + 20  # 增加左側邊距使其對齊預覽圖
        
        # 配置結果標籤
        self.result_label = ttk.Label(frame, font=("Arial", 16, "bold"))
        self.result_label.grid(row=1, column=0, padx=(text_left_margin, 0), pady=5, sticky="w")
        
        entropy_frame = ttk.Frame(frame)
        entropy_frame.grid(row=2, column=0, padx=(text_left_margin, 0), pady=5, sticky="w")
        
        self.entropy_label = ttk.Label(entropy_frame, style="Result.TLabel")
        self.entropy_label.pack(side=tk.TOP, anchor="w", pady=(0, 5))
        
        prob_container = ttk.Frame(entropy_frame)
        prob_container.pack(fill=tk.X, expand=True)
        
        self.prob_labels = []
        for i in range(5):
            label = ttk.Label(prob_container, style="Probability.TLabel")
            label.pack(side=tk.TOP, anchor="w", pady=1)
            self.prob_labels.append(label)
        
        # 配置準確率標籤
        self.accuracy_label = ttk.Label(frame, font=("Arial", 14, "bold"))
        self.accuracy_label.grid(row=3, column=0, padx=(text_left_margin, 0), pady=5, sticky="w")

    def load_images(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.image_items = []
        self.image_paths = []
        
        for class_dir in sorted(os.listdir(self.test_dir)):
            class_path = os.path.join(self.test_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            for img_path in glob.glob(os.path.join(class_path, "*.[jp][pn][g]")):
                try:
                    self._create_image_item(img_path, class_dir)
                    self.image_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # 自動選取第一個項目
        if self.image_items:
            self.on_image_select(self.image_items[0])

    def _create_image_item(self, img_path, class_dir):
        item_frame = ttk.Frame(self.scrollable_frame, style="ImageItem.TFrame")
        item_frame.pack(fill=tk.X, padx=5, pady=2)
        item_frame.path = img_path
        
        # 建立縮圖和標籤的容器，使用與父框架相同的樣式
        content_frame = ttk.Frame(item_frame, style="ImageItem.TFrame")
        content_frame.pack(fill=tk.X, expand=True)
        
        # 建立縮圖
        image = Image.open(img_path).convert('RGB')
        thumb = image.copy()
        thumb.thumbnail(self.GUI_CONFIG['thumbnail_size'])
        photo = ImageTk.PhotoImage(thumb)
        self.image_cache[img_path] = photo
        
        img_label = ttk.Label(content_frame, image=photo, style="ImageItem.TLabel")
        img_label.image = photo
        img_label.pack(side=tk.LEFT, padx=5)
        
        name_label = ttk.Label(
            content_frame,
            text=f"{class_dir}/{os.path.basename(img_path)}",
            wraplength=160,  # 增加文字換行寬度
            style="ImageItem.TLabel"
        )
        name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 綁定點擊事件到所有元件
        for widget in (item_frame, content_frame, img_label, name_label):
            widget.bind('<Button-1>', lambda e, f=item_frame: self.on_image_select(f))
        
        self.image_items.append(item_frame)

    def on_image_select(self, selected_frame):
        if not selected_frame or not selected_frame.winfo_exists():
            return
            
        def update_style(widget, is_selected):
            style_suffix = "Selected.ImageItem" if is_selected else "ImageItem"
            if isinstance(widget, ttk.Frame):
                widget.configure(style=f"{style_suffix}.TFrame")
            elif isinstance(widget, ttk.Label):
                widget.configure(style=f"{style_suffix}.TLabel")
            # 遞迴處理所有子元件
            for child in widget.winfo_children():
                update_style(child, is_selected)
        
        # 重設之前選取項目的樣式
        if self.selected_item and self.selected_item.winfo_exists():
            update_style(self.selected_item, False)
        
        # 設定新選取項目的樣式
        update_style(selected_frame, True)
        self.selected_item = selected_frame
        
        selected_path = selected_frame.path
        if selected_path in self.image_paths:
            idx = self.image_paths.index(selected_path)
            self.image_paths = (
                [self.image_paths[idx]] + 
                self.image_paths[:idx] + 
                self.image_paths[idx+1:]
            )
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def run_next(self):
        if not self.image_paths:
            return False
            
        try:
            image_path = self.image_paths.pop(0)
            input_tensor = self.transform(
                Image.open(image_path).convert('RGB')
            ).unsqueeze(0).to(self.device)
            
            # 修改標籤處理邏輯：從目錄名稱獲取類別索引
            dir_name = os.path.basename(os.path.dirname(image_path))
            true_label = self.dir_to_class[dir_name]  # 取得類別索引
            true_label_mapped = self.class_names[true_label]  # 取得可讀的類別名稱
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                pred_idx = torch.max(output, 1)[1].item()
                pred_label = self.class_names[str(pred_idx)]
                
                # 使用正確的數字索引計算交叉熵損失
                criterion = torch.nn.CrossEntropyLoss()
                target = torch.tensor([int(true_label)]).to(self.device)
                loss = criterion(output, target).item()
            
            self.total_count += 1
            if pred_label == true_label_mapped:
                self.correct_count += 1
                
            self.update_display(
                Image.open(image_path).convert('RGB'),
                true_label_mapped,
                pred_label,
                probabilities[0],  # 傳遞機率分布
                loss  # 傳遞交叉熵損失
            )
            self.remove_image_item(image_path)
            
            # 自動選取下一張圖片
            if self.image_items:
                self.on_image_select(self.image_items[0])
            
            return True
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return False

    def auto_run_all(self):
        while self.run_next():
            self.window.update()
    
    def update_display(self, image, true_label, pred_label, probs, loss):
        display_image = image.resize(self.GUI_CONFIG['display_size'])
        photo = ImageTk.PhotoImage(display_image)
        self.display_label.configure(image=photo)
        self.display_label.image = photo
        
        result_text = f"True: {true_label} | Pred: {pred_label}"
        self.result_label.config(
            text=result_text,
            foreground="green" if pred_label == true_label else "red"
        )
        
        # 更新交叉熵損失和機率顯示
        self.entropy_label.config(text=f"Cross Entropy Loss: {loss:.4f}")
        
        # 更新每個類別的機率顯示
        max_prob = torch.max(probs).item()
        for i, (prob, label) in enumerate(zip(probs, self.prob_labels)):
            prob_value = prob.item()
            prob_text = f"{self.class_names[str(i)]}: {prob_value:.4f}"
            
            # 修正：直接使用傳入的 true_label 和 pred_label 比較
            if str(i) == self.dir_to_class[true_label] and true_label == pred_label:
                label.configure(foreground="green", font=("Arial", 11, "bold"))
            elif str(i) == self.dir_to_class[true_label]:
                label.configure(foreground="blue", font=("Arial", 11, "bold"))
            elif prob_value == max_prob:
                label.configure(foreground="red", font=("Arial", 11, "bold"))
            else:
                label.configure(foreground="#666666", font=("Arial", 11))
            
            label.configure(text=prob_text)
        
        accuracy = (self.correct_count / self.total_count * 100) if self.total_count > 0 else 0
        self.accuracy_label.config(
            text=f"Accuracy: {self.correct_count}/{self.total_count} ({accuracy:.2f}%)"
        )

    def remove_image_item(self, image_path):
        for item in self.image_items[:]:
            if item.path == image_path:
                # 如果移除的是目前選取的項目，找下一個可選取的項目
                if item == self.selected_item and self.image_items:
                    next_idx = self.image_items.index(item) + 1
                    if next_idx < len(self.image_items):
                        self.on_image_select(self.image_items[next_idx])
                item.destroy()
                self.image_items.remove(item)
                break

