import cv2
import numpy as np
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
import sys

# 設定系統編碼
sys.stdout.reconfigure(encoding='utf-8')

# 裁剪參數：設定裁剪區域
CROP_PARAMS = {
    'left': 235,
    'upper': 90,
    'width': 1024,
    'height': 1024
}
EXCLUDE_FOLDERS = ['移出圖片']
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

def process_folder(input_folder, progress_callback=None):
    """處理資料夾主函數"""
    if not isinstance(input_folder, Path):
        input_folder = Path(input_folder).resolve()
    
    if not input_folder.is_dir():
        raise ValueError(f"錯誤: {input_folder} 不是有效資料夾")
    
    # 修改輸出目錄，直接使用 IMG_Cut
    output_base = input_folder.parent / 'IMG_Cut'
    output_base.mkdir(parents=True, exist_ok=True)
    
    success_count = failure_count = 0
    error_files = []  # 新增錯誤檔案清單
    
    # 遍歷所有子目錄
    for path in input_folder.rglob("*"):
        if any(excluded in path.parts for excluded in EXCLUDE_FOLDERS):
            continue
            
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
            try:
                # 修改輸出路徑計算方式
                rel_path = path.parent.relative_to(input_folder)
                if str(rel_path) == '.':  # 如果是根目錄
                    output_folder = output_base
                else:
                    output_folder = (output_base / rel_path).resolve()
                output_folder.mkdir(parents=True, exist_ok=True)
                
                result = crop_image(path, output_folder)
                if result is True:
                    success_count += 1
                else:
                    failure_count += 1
                    error_files.append((path.name, str(result)))
            except Exception as e:
                failure_count += 1
                error_files.append((path.name, str(e)))
            
            if progress_callback:
                progress_callback()
    
    return success_count, failure_count, error_files

def crop_image(image_path, output_folder):
    """裁切圖片"""
    try:
        # 使用完整的 Unicode 路徑讀取圖片
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return f"無法讀取圖片: {image_path}"
        
        # 使用固定區域裁切
        right = CROP_PARAMS['left'] + CROP_PARAMS['width']
        lower = CROP_PARAMS['upper'] + CROP_PARAMS['height']
        cropped = img[CROP_PARAMS['upper']:lower, CROP_PARAMS['left']:right]
        
        # 使用 imencode 處理中文路徑的寫入
        output_path = output_folder / image_path.name
        _, img_encoded = cv2.imencode(image_path.suffix, cropped)
        img_encoded.tofile(str(output_path))
        return True
        
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    try:
        # 建立 root 視窗但隱藏它
        root = tk.Tk()
        root.withdraw()
        
        # 使用對話框選擇資料夾
        folder_path = filedialog.askdirectory(title="選擇要處理的圖片資料夾")
        if not folder_path:  # 使用者取消選擇
            print("已取消操作")
            exit()
            
        test_folder = Path(folder_path)
        print(f"\n開始處理資料夾: {test_folder}")
        
        # 執行處理
        success, failure, error_files = process_folder(test_folder)
        
        # 顯示結果
        print("\n處理完成！")
        print(f"成功處理: {success} 張圖片")
        print(f"處理失敗: {failure} 張圖片")
        print(f"輸出位置: {test_folder.parent / 'IMG_Cut' / test_folder.name}")
        
        # 顯示錯誤檔案清單
        if error_files:
            print("\n處理失敗的檔案：")
            for filename, error in error_files:
                print(f"- {filename}: {error}")
        
    except Exception as e:
        print(f"執行時發生錯誤: {str(e)}")


