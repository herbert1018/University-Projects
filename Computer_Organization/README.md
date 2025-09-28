# Computer Organization – 期末專題

## 專案簡介
本專案為「計算機組織」課程的期末專題，主題為 **最大子陣列問題 (Maximum Subarray Problem)**。  
使用circuitverse作為實作網站，內容包含演算法實作、程式碼示範，以及簡報文件。
![問題示意圖](../suport/CO_Problem.png)
![電路展示圖](../suport/CO_Demo.png)

## 檔案說明
- `max subarray.cpp`：最大子陣列演算法的程式碼實作 (C++)  
- `__maxSubarray__F.cv`：電路設計檔，可直接導入circuitverse使用  
- `Demo.png`：整個數位邏輯電路的實體圖
- `計算機組織 - 期末專題.pptx`：期末簡報檔  
- `README.md`：專案說明文件（本檔案）

## 執行方式
1. 開啟 circuitverse 並登錄後創建空白專案
2. 使用"Import file"匯入`__maxSubarray__F.cv`
3. 將對應數字串轉成二補數 (本專案未實作自動轉換)  
4. 將轉換後字串貼進RAM後，拉一下後面線路確保存取(網站bug)
5. 點擊start按鈕觀看處理過程並獲取結果 

## 專案特色
- 使用 Dynamic programming 演算法解決最大子陣列問題
![C++演算法圖](../suport/CO_Algorithm.png)
- 透過圖示與簡報說明演算法流程