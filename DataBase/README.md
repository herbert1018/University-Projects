# Database System – 期末專案

## 專案簡介
本專案為「資料庫系統」課程的期末專題，內容涵蓋 **使用 Node.js 建立後端伺服器**，並搭配 SQL 資料庫進行使用者登入、註冊與基本介面操作。

## 檔案說明
- `server.js`：Node.js 後端伺服器程式  
- `com.sql`：SQL 腳本，包含資料表建立與測試資料  
- `public/`：前端靜態檔案
  - `home.html` / `home_style.css`：首頁頁面與樣式
  - `login_register.html` / `style.css`：登入、註冊頁面與樣式
- `資料庫系統_期末報告.pptx`：期末簡報檔  
- `node_modules.zip`：依賴模組壓縮檔（需解壓縮後使用）  
- `package-lock.json`：NPM 依賴管理檔  

## 執行方式
```bash
# 解壓縮 node_modules.zip
unzip node_modules.zip -d node_modules

# 啟動伺服器
node server.js
伺服器啟動後，可在瀏覽器訪問 http://localhost:3000 查看網站。
```

## 專案特色
- 使用 Node.js + Express 建立後端
- 前後端分離：HTML/CSS 前端與 Node.js 後端結合
- SQL 資料庫進行使用者資料管理