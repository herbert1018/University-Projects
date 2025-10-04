import cv2
import numpy as np
import os
from pathlib import Path

SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def process_image(image_path, output_folder):
    """處理單張圖片"""
    try:
        # 使用 os.path.abspath 處理中文路徑
        safe_path = str(image_path.absolute())
        
        # 使用 cv2.imdecode 來讀取圖片
        with open(safe_path, 'rb') as f:
            image_array = np.frombuffer(f.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
        if image is None:
            raise ValueError("圖片讀取失敗")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([1, 0, 0]), np.array([179, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        # 處理輸出路徑：保持原始檔名
        output_path = output_folder / image_path.name
        
        # 使用 cv2.imencode 來儲存圖片
        is_success, buffer = cv2.imencode(image_path.suffix, result)
        if is_success:
            with open(str(output_path.absolute()), 'wb') as f:
                f.write(buffer)
        else:
            raise ValueError("圖片儲存失敗")
            
        return True, ""
    except Exception as e:
        return False, str(e)

def process_folder(input_folder, output_folder, progress_callback=None):
    """處理資料夾的主要函數
    
    Args:
        input_folder (str/Path): 輸入資料夾路徑
        output_folder (str/Path): 輸出資料夾路徑
        progress_callback (callable): 進度回調函數
    
    Returns:
        tuple: (成功數, 失敗數)
    """
    input_folder = Path(input_folder) if not isinstance(input_folder, Path) else input_folder
    if not input_folder.is_dir():
        raise ValueError(f"錯誤: {input_folder} 不是有效資料夾")

    output_folder = Path(output_folder) if not isinstance(output_folder, Path) else output_folder
    output_folder.mkdir(exist_ok=True)

    success_count, failure_count = 0, 0
    
    # 遞迴處理所有檔案和子資料夾
    for item in input_folder.rglob("*"):
        if item.is_file() and item.suffix.lower() in SUPPORTED_FORMATS:
            # 計算相對路徑以保持資料夾結構
            rel_path = item.relative_to(input_folder)
            target_folder = output_folder / rel_path.parent
            target_folder.mkdir(parents=True, exist_ok=True)
            
            success, error_msg = process_image(item, target_folder)
            if success:
                success_count += 1
            else:
                failure_count += 1
            
            if progress_callback:
                progress_callback()

    return success_count, failure_count
