/*CREATE DATABASE test1;*/
USE test1;

/*網站用戶*/
CREATE TABLE users (
    username VARCHAR(100) PRIMARY KEY,  -- 用戶名稱(主鍵)
    password VARCHAR(255) NOT NULL  	-- 密碼
);
/*用戶資料*/
CREATE TABLE user_profiles (
    username VARCHAR(50) PRIMARY KEY,   -- 用戶名(主鍵)
    email VARCHAR(100) DEFAULT '無',  	-- 電子郵件
    phone VARCHAR(20)  DEFAULT '無',    -- 電話
    address TEXT ,       			    -- 地址
    balance int DEFAULT 0,			    -- 錢錢
    FOREIGN KEY (username) REFERENCES users(username)
    ON DELETE CASCADE
);
/*倉庫*/
CREATE TABLE warehouse (
	own_name VARCHAR(50),                		-- 擁有者名稱
    product_name VARCHAR(100) NOT NULL,         -- 商品名稱
    quantity INT DEFAULT 0,                     -- 庫存數量
    PRIMARY KEY (own_name, product_name),		-- 複合主鍵
    FOREIGN KEY (own_name) REFERENCES users(username) 
    ON DELETE CASCADE
);
/*架上商品*/
CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,  -- 商品編號(主鍵)
    product_name VARCHAR(100) NOT NULL,         -- 商品名稱
    quantity INT DEFAULT 0,                     -- 庫存數量
    price int,              					-- 金額
    seller_username VARCHAR(50),                -- 賣家名稱
    FOREIGN KEY (seller_username) REFERENCES users(username) 
    ON DELETE CASCADE
);



/*新增測試
INSERT INTO user_profiles (username, email, phone, address, balance) 
VALUES ('123', 'isu@gmail.com', '1234567890', '台北市某某街道', '100');
INSERT INTO user_profiles (username, email, phone, address, balance)
VALUES ('456', 'isu2@gmail.com', '1234567899', '台北市某街道', '100');

INSERT INTO products (product_name, quantity, price, seller_username) 
VALUES 
('蘋果', 50, 10, '123'),
('香蕉', 30, 5, '123');

INSERT INTO warehouse (own_name, product_name, quantity) 
VALUES 
("123",'蘋果', 50);


/*查詢+修改
SET SQL_SAFE_UPDATES = 0;
UPDATE user_profiles SET balance = 2000 WHERE username = '456';
Select * from users;
Select * from user_profiles;
Select * from warehouse;
Select * from products;


/*刪除
delete from users where username='456';
delete from warehouse where own_name='456';
delete from products where seller_username='123';
drop table warehouse;
drop table products;
drop table user_profiles;
drop table users;
