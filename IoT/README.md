# IoT – 捷運站驗票閘門模擬專案

## 專案簡介
本專案為「物聯網 (IoT)」課程的期末專題，設計一個 **捷運站驗票閘門系統**，模擬人流管制與異常狀況偵測。

<p align="center">
  <img src="../suport/IOT_UI.png" alt="前端UI" width="200" />
  <img src="../suport/IOT_Demo.png" alt="實體展示" width="200" />
</p>
<p align="center">
  <img src="../suport/IOT_NodeRed.png" alt="資料流" width="200" />
  <img src="../suport/IOT_Use.png" alt="使用元件" width="200" />
</p>

## 檔案說明
- `MRT_Gate/MRT_Gate.ino`：ESP32 (Arduino IDE) 程式碼，控制閘門模擬  
- `flows.json`：Node-RED 流程檔，實現即時資料顯示與控制  
- `database.txt`：資料庫相關設定或測試資料  
- `DemoVideo.mp4`：專案展示影片  
- `捷運站驗票閘門.pdf`：期末專案報告  

## 系統架構
1. ESP32 + 感測器 → 偵測通行與異常狀況  
2. Node-RED → 負責資料收集轉傳、視覺化與遠端控制  
3. HeidiSQL → 儲存人流紀錄與異常事件  
4. 前端介面 → 即時顯示人流、閘門與異常狀態  

## 執行方式
1. 將 `MRT_Gate.ino` 上傳至 ESP32  
2. 開啟 MQTTX 並依據要求設定並訂閱相關參數名
2. 匯入 `flows.json` 至 Node-RED，並啟動  
3. 開啟Node-RED給的UI介面並測試功能  

## 專案特色
- 模擬 **捷運站閘門驗票系統**  
- 具備 **人流紀錄**、**異常狀況警示**、**遠端控制** 三大功能  
- Node-RED 儀表板整合感測器數據與控制介面  
- 重要元件: RC522 IC卡感應模組、KY-025 磁簧開關