// 資料庫系統
const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');
const session = require('express-session');
const path = require('path');

const app = express();
const port = 3000;

// 設定 MySQL 資料庫連接
const db = mysql.createConnection({
    host: 'localhost', // 資料庫地址
    user: 'root',      // 使用者名稱
    password: '123456',// 使用者密碼
    database: 'test1'  // 資料庫名稱
});

// 連接資料庫
db.connect((err) => {
    if (err) {
        console.error('資料庫連接失敗:', err.stack);
        return;
    }
    console.log('成功連接到 MySQL 資料庫');
});

// 設定 Express 中間件
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use(session({
    secret: 'your_secret_key',
    resave: false,
    saveUninitialized: true
}));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'login_register.html'));
});


// 登錄檢查API
app.get('/check-login', (req, res) => {
    if (req.session.loggedIn) {
        res.json({
            loggedIn: true,
            username: req.session.username
        });
    } else {
        res.json({
            loggedIn: false
        });
    }
});

// 登錄API
app.post('/login', (req, res) => {
    const { username, password } = req.body;
    const query = 'SELECT * FROM users WHERE username = ? AND password = ?';
    db.query(query, [username, password], (err, results) => {
        if (err) {
            return res.status(500).json({ message: '資料庫錯誤' });
        }

        if (results.length > 0) {
            req.session.loggedIn = true;
            req.session.username = username;
            res.json({ success: true, message: '登錄成功' });
        } else {
            res.status(400).json({ success: false, message: '帳號或密碼錯誤' });
        }
    });
});

// 註冊API
app.post('/register', (req, res) => {
    const { username, password } = req.body;

    // 檢查用戶名是否已經存在
    const checkQuery = 'SELECT * FROM users WHERE username = ?';
    db.query(checkQuery, [username], (err, results) => {
        if (err) {
            return res.status(500).json({ message: '資料庫錯誤' });
        }

        if (results.length > 0) {
            // 用戶名已經存在
            return res.status(400).json({ success: false, message: '該用戶名已經註冊過' });
        }
        else {
            // 用戶名未註冊，執行註冊
            const insertQuery = 'INSERT INTO users (username, password) VALUES (?, ?); ';
            db.query(insertQuery, [username, password], (err, results) => {
                if (err) {
                    return res.status(500).json({ message: '註冊失敗' });
                }
            });
            // 註冊成功後，在 user_profiles 表中插入用戶資料
            const insertProfileQuery = `
            INSERT INTO user_profiles (username, email, phone, address, balance)
            VALUES (?, '無', '無', NULL, 0);
            `;

            db.query(insertProfileQuery, [username], (err, results) => {
                if (err) {
                    return res.status(500).json({ message: '創建用戶資料失敗' });
                }
                res.json({ success: true, message: '註冊成功' });
            });
        }
    });
});
//獲取個人資料API
app.get('/user-info/:username', (req, res) => {
    const username = req.params.username;
    const query = 'SELECT email, phone, address, balance FROM user_profiles WHERE username = ?';

    db.query(query, [username], (err, results) => {
        if (err) {
            console.error('Database error:', err);  // 打印数据库错误
            return res.status(500).json({ message: '資料庫錯誤' });
        }

        if (results.length > 0) {
            const user = results[0];
            res.json({
                email: user.email || '未設定',
                phone: user.phone || '未設定',
                address: user.address || '未設定',
                balance: user.balance || 0
            });
        } else {
            console.log('User not found');  // 打印用户未找到的情况
            res.status(404).json({ message: '用戶未找到' });
        }
    });
});
// 更新個人資料API
app.post('/update-profile', (req, res) => {
    const { username, email, phone, address } = req.body;

    const query = 'UPDATE user_profiles SET email = ?, phone = ?, address = ? WHERE username = ?';
    db.query(query, [email, phone, address, username], (err, result) => {
        if (err) {
            console.error('更新個人資料失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤，無法更新個人資料' });
        }

        res.json({ success: true, message: '個人資料更新成功' });
    });
});
// 登出API
app.post('/logout', (req, res) => {
    if (req.session.loggedIn) {
        // 清除會話資料
        req.session.destroy((err) => {
            if (err) {
                return res.status(500).json({ message: '登出失敗' });
            }
            res.json({ success: true, message: '登出成功' });
        });
    } else {
        res.status(400).json({ message: '未登入，無法登出' });
    }
});
// 獲取商品列表API
app.get('/get-products', (req, res) => {
    const buyer_username = req.session.username;
    const { sortCriteria = 'product_name', sortOrder = 'asc' } = req.query;

    if (!buyer_username) {
        return res.status(401).json({ message: '未登入' });
    }

    const validSortCriteria = ['product_name', 'price', 'quantity', 'seller_username'];
    if (!validSortCriteria.includes(sortCriteria)) {
        return res.status(400).json({ message: '無效的排序標準' });
    }

    const validSortOrder = ['asc', 'desc'];
    if (!validSortOrder.includes(sortOrder)) {
        return res.status(400).json({ message: '無效的排序順序' });
    }

    const query = `SELECT product_id, product_name, quantity, price, seller_username FROM products WHERE seller_username != ? ORDER BY ${sortCriteria} ${sortOrder}`;
    db.query(query, [buyer_username], (err, results) => {
        if (err) {
            console.error('查詢商品列表失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        // 如果倉庫為空，返回空列表
        res.json(results || []);
    });
});
// 獲取私人倉庫商品列表API
app.get('/get-warehouse-products', (req, res) => {
    const username = req.session.username;
    const { sortCriteria = 'product_name', sortOrder = 'asc' } = req.query;

    if (!username) {
        return res.status(401).json({ message: '未登入' });
    }

    const validSortCriteria = ['product_name', 'price', 'quantity'];
    if (!validSortCriteria.includes(sortCriteria)) {
        return res.status(400).json({ message: '無效的排序標準' });
    }

    const validSortOrder = ['asc', 'desc'];
    if (!validSortOrder.includes(sortOrder)) {
        return res.status(400).json({ message: '無效的排序順序' });
    }

    const query = `SELECT * FROM warehouse WHERE own_name = ? ORDER BY ${sortCriteria} ${sortOrder}`;
    db.query(query, [username], (err, results) => {
        if (err) {
            console.error('查詢倉庫商品列表失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        res.json(results);
    });
});

// 搜尋倉庫商品API
app.get('/search-warehouse-products', (req, res) => {
    const username = req.session.username;
    const keyword = req.query.keyword;
    const criteria = req.query.criteria;

    if (!username) {
        return res.status(401).json({ message: '未登入' });
    }

    if (!keyword) {
        return res.status(400).json({ message: '缺少關鍵字' });
    }

    let query;
    if (criteria === 'product_name') {
        query = 'SELECT * FROM warehouse WHERE own_name = ? AND product_name LIKE ?';
    } else if (criteria === 'quantity') {
        query = 'SELECT * FROM warehouse WHERE own_name = ? AND quantity = ?';
    } else {
        return res.status(400).json({ message: '無效的搜尋標準' });
    }

    db.query(query, [username, criteria === 'product_name' ? `%${keyword}%` : keyword], (err, results) => {
        if (err) {
            console.error('搜尋倉庫商品失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        res.json(results);
    });
});

// 搜尋交易頁面商品API
app.get('/search-products', (req, res) => {
    const buyer_username = req.session.username;
    const keyword = req.query.keyword;
    const criteria = req.query.criteria;

    if (!buyer_username) {
        return res.status(401).json({ message: '未登入' });
    }

    if (!keyword) {
        return res.status(400).json({ message: '缺少關鍵字' });
    }

    let query;
    if (criteria === 'product_name') {
        query = 'SELECT * FROM products WHERE seller_username != ? AND product_name LIKE ?';
    } else if (criteria === 'price') {
        query = 'SELECT * FROM products WHERE seller_username != ? AND price = ?';
    } else if (criteria === 'quantity') {
        query = 'SELECT * FROM products WHERE seller_username != ? AND quantity = ?';
    } else if (criteria === 'seller_username') {
        query = 'SELECT * FROM products WHERE seller_username != ? AND seller_username LIKE ?';
    } else {
        return res.status(400).json({ message: '無效的搜尋標準' });
    }

    db.query(query, [buyer_username, criteria === 'product_name' || criteria === 'seller_username' ? `%${keyword}%` : keyword], (err, results) => {
        if (err) {
            console.error('搜尋交易頁面商品失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        res.json(results);
    });
});

// 搜尋販售中的商品API
app.get('/search-selling-products', (req, res) => {
    const username = req.session.username;
    const keyword = req.query.keyword;
    const criteria = req.query.criteria;

    if (!username) {
        return res.status(401).json({ message: '未登入' });
    }

    if (!keyword) {
        return res.status(400).json({ message: '缺少關鍵字' });
    }

    let query;
    if (criteria === 'product_name') {
        query = 'SELECT * FROM products WHERE seller_username = ? AND product_name LIKE ?';
    } else if (criteria === 'price') {
        query = 'SELECT * FROM products WHERE seller_username = ? AND price = ?';
    } else if (criteria === 'quantity') {
        query = 'SELECT * FROM products WHERE seller_username = ? AND quantity = ?';
    } else {
        return res.status(400).json({ message: '無效的搜尋標準' });
    }

    db.query(query, [username, criteria === 'product_name' ? `%${keyword}%` : keyword], (err, results) => {
        if (err) {
            console.error('搜尋販售中的商品失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        res.json(results);
    });
});

// 上傳商品API
app.post('/add-product', (req, res) => {
    const { product_name, quantity, seller_username } = req.body;

    // 驗證必填欄位
    if (!product_name || !quantity || !seller_username) {
        return res.status(400).json({ message: '缺少必要的欄位' });
    }

    // 驗證 quantity 是否為有效的整數
    if (isNaN(quantity) || quantity < 0) {
        return res.status(400).json({ message: '庫存數量無效' });
    }

    // 檢查 seller_username 是否存在於 users 表中
    const checkUserQuery = 'SELECT * FROM users WHERE username = ?';
    db.query(checkUserQuery, [seller_username], (err, results) => {
        if (err) {
            console.error('檢查用戶錯誤:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        if (results.length === 0) {
            return res.status(400).json({ message: '賣家名稱不存在' });
        }

        // 檢查倉庫中是否已經存在相同用戶名和商品名的商品
        const checkWarehouseQuery = 'SELECT * FROM warehouse WHERE own_name = ? AND product_name = ?';
        db.query(checkWarehouseQuery, [seller_username, product_name], (err, results) => {
            if (err) {
                console.error('檢查倉庫失敗:', err);
                return res.status(500).json({ message: '伺服器錯誤' });
            }

            if (results.length > 0) {
                // 如果存在相同用戶名和商品名的商品，則更新數量
                const updateWarehouseQuery = 'UPDATE warehouse SET quantity = quantity + ? WHERE own_name = ? AND product_name = ?';
                db.query(updateWarehouseQuery, [quantity, seller_username, product_name], (err) => {
                    if (err) {
                        console.error('更新倉庫失敗:', err);
                        return res.status(500).json({ message: '伺服器錯誤' });
                    }

                    // 更新成功後重新查詢倉庫商品列表
                    const getUpdatedWarehouseQuery = 'SELECT * FROM warehouse WHERE own_name = ?';
                    db.query(getUpdatedWarehouseQuery, [seller_username], (err, updatedResults) => {
                        if (err) {
                            console.error('查詢更新後倉庫商品列表失敗:', err);
                            return res.status(500).json({ message: '伺服器錯誤' });
                        }

                        res.json({ success: true, message: '商品上傳成功並更新倉庫數量', warehouse: updatedResults });
                    });
                });
            } else {
                // 如果不存在相同用戶名和商品名的商品，則新增新的倉庫列
                const addWarehouseQuery = 'INSERT INTO warehouse (own_name, product_name, quantity) VALUES (?, ?, ?)';
                db.query(addWarehouseQuery, [seller_username, product_name, quantity], (err) => {
                    if (err) {
                        console.error('新增倉庫失敗:', err);
                        return res.status(500).json({ message: '伺服器錯誤' });
                    }

                    // 新增成功後重新查詢倉庫商品列表
                    const getUpdatedWarehouseQuery = 'SELECT * FROM warehouse WHERE own_name = ?';
                    db.query(getUpdatedWarehouseQuery, [seller_username], (err, updatedResults) => {
                        if (err) {
                            console.error('查詢更新後倉庫商品列表失敗:', err);
                            return res.status(500).json({ message: '伺服器錯誤' });
                        }

                        res.json({ success: true, message: '商品新增成功並新增新的倉庫列', warehouse: updatedResults });
                    });
                });
            }
        });
    });
});

// 上架商品API
app.post('/list-product', (req, res) => {
    const { product_name, quantity, price, seller_username } = req.body;

    // 驗證必填欄位
    if (!product_name || !quantity || !price || !seller_username) {
        return res.status(400).json({ message: '缺少必要的欄位' });
    }

    // 驗證 quantity 和 price 是否為有效的數字
    if (isNaN(quantity) || quantity <= 0 || isNaN(price) || price <= 0) {
        return res.status(400).json({ message: '數量或價格無效' });
    }

    // 檢查 seller_username 是否存在於 users 表中
    const checkUserQuery = 'SELECT * FROM users WHERE username = ?';
    db.query(checkUserQuery, [seller_username], (err, results) => {
        if (err) {
            console.error('檢查用戶錯誤:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        if (results.length === 0) {
            return res.status(400).json({ message: '賣家名稱不存在' });
        }

        // 檢查倉庫中是否有足夠的庫存
        const checkWarehouseQuery = 'SELECT quantity FROM warehouse WHERE own_name = ? AND product_name = ?';
        db.query(checkWarehouseQuery, [seller_username, product_name], (err, results) => {
            if (err) {
                console.error('檢查倉庫失敗:', err);
                return res.status(500).json({ message: '伺服器錯誤' });
            }

            if (results.length === 0) {
                return res.status(400).json({ message: '倉庫中沒有該商品' });
            }

            const warehouseQuantity = results[0].quantity;

            if (quantity > warehouseQuantity) {
                return res.status(400).json({ message: '上架數量超過倉庫庫存' });
            }

            // 檢查是否有相同名字和價錢的商品
            const checkProductQuery = 'SELECT * FROM products WHERE product_name = ? AND price = ? AND seller_username = ?';
            db.query(checkProductQuery, [product_name, price, seller_username], (err, results) => {
                if (err) {
                    console.error('檢查商品失敗:', err);
                    return res.status(500).json({ message: '伺服器錯誤' });
                }

                if (results.length > 0) {
                    // 如果存在相同名字和價錢的商品，則合併數量
                    const updateProductQuery = 'UPDATE products SET quantity = quantity + ? WHERE product_name = ? AND price = ? AND seller_username = ?';
                    db.query(updateProductQuery, [quantity, product_name, price, seller_username], (err) => {
                        if (err) {
                            console.error('合併商品失敗:', err);
                            return res.status(500).json({ message: '伺服器錯誤' });
                        }

                        // 更新倉庫中的商品數量
                        const newWarehouseQuantity = warehouseQuantity - quantity;
                        if (newWarehouseQuantity > 0) {
                            const updateWarehouseQuery = 'UPDATE warehouse SET quantity = ? WHERE own_name = ? AND product_name = ?';
                            db.query(updateWarehouseQuery, [newWarehouseQuantity, seller_username, product_name], (err) => {
                                if (err) {
                                    console.error('更新倉庫失敗:', err);
                                    return res.status(500).json({ message: '伺服器錯誤' });
                                }

                                res.json({ success: true, message: '商品上架成功並合併' });
                            });
                        } else {
                            const deleteWarehouseQuery = 'DELETE FROM warehouse WHERE own_name = ? AND product_name = ?';
                            db.query(deleteWarehouseQuery, [seller_username, product_name], (err) => {
                                if (err) {
                                    console.error('刪除倉庫商品失敗:', err);
                                    return res.status(500).json({ message: '伺服器錯誤' });
                                }

                                res.json({ success: true, message: '商品上架成功並從倉庫中刪除' });
                            });
                        }
                    });
                } else {
                    // 將商品插入到 products 表中
                    const addProductQuery = 'INSERT INTO products (product_name, quantity, price, seller_username) VALUES (?, ?, ?, ?)';
                    db.query(addProductQuery, [product_name, quantity, price, seller_username], (err) => {
                        if (err) {
                            console.error('新增商品失敗:', err);
                            return res.status(500).json({ message: '伺服器錯誤' });
                        }

                        // 更新倉庫中的商品數量
                        const newWarehouseQuantity = warehouseQuantity - quantity;
                        if (newWarehouseQuantity > 0) {
                            const updateWarehouseQuery = 'UPDATE warehouse SET quantity = ? WHERE own_name = ? AND product_name = ?';
                            db.query(updateWarehouseQuery, [newWarehouseQuantity, seller_username, product_name], (err) => {
                                if (err) {
                                    console.error('更新倉庫失敗:', err);
                                    return res.status(500).json({ message: '伺服器錯誤' });
                                }

                                res.json({ success: true, message: '商品上架成功' });
                            });
                        } else {
                            const deleteWarehouseQuery = 'DELETE FROM warehouse WHERE own_name = ? AND product_name = ?';
                            db.query(deleteWarehouseQuery, [seller_username, product_name], (err) => {
                                if (err) {
                                    console.error('刪除倉庫商品失敗:', err);
                                    return res.status(500).json({ message: '伺服器錯誤' });
                                }

                                res.json({ success: true, message: '商品上架成功並從倉庫中刪除' });
                            });
                        }
                    });
                }
            });
        });
    });
});

// 刪除私人倉庫商品API
app.post('/delete-warehouse-product', (req, res) => {
    const { product_name, seller_username } = req.body;

    // 驗證必填欄位
    if (!product_name || !seller_username) {
        return res.status(400).json({ message: '缺少必要的欄位' });
    }

    const deleteWarehouseProductQuery = 'DELETE FROM warehouse WHERE own_name = ? AND product_name = ?';
    db.query(deleteWarehouseProductQuery, [seller_username, product_name], (err) => {
        if (err) {
            console.error('刪除倉庫商品失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        res.json({ success: true, message: '商品刪除成功' });
    });
});

// 編輯私人倉庫商品API
app.post('/edit-warehouse-product', (req, res) => {
    const { original_product_name, product_name, quantity, seller_username } = req.body;

    // 驗證必填欄位
    if (!original_product_name || !product_name || !quantity || !seller_username) {
        return res.status(400).json({ message: '缺少必要的欄位' });
    }

    // 檢查是否存在同名的商品
    const checkWarehouseQuery = 'SELECT * FROM warehouse WHERE own_name = ? AND product_name = ?';
    db.query(checkWarehouseQuery, [seller_username, product_name], (err, results) => {
        if (err) {
            console.error('檢查倉庫失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        if (results.length > 0 && original_product_name !== product_name) {
            // 如果存在同名的商品，則合併數量
            const updateWarehouseQuery = 'UPDATE warehouse SET quantity = quantity + ? WHERE own_name = ? AND product_name = ?';
            db.query(updateWarehouseQuery, [quantity, seller_username, product_name], (err) => {
                if (err) {
                    console.error('更新倉庫失敗:', err);
                    return res.status(500).json({ message: '伺服器錯誤' });
                }

                // 刪除原始商品
                const deleteOriginalProductQuery = 'DELETE FROM warehouse WHERE own_name = ? AND product_name = ?';
                db.query(deleteOriginalProductQuery, [seller_username, original_product_name], (err) => {
                    if (err) {
                        console.error('刪除原始商品失敗:', err);
                        return res.status(500).json({ message: '伺服器錯誤' });
                    }

                    res.json({ success: true, message: '商品合併成功' });
                });
            });
        } else {
            // 更新商品資料
            const updateWarehouseQuery = 'UPDATE warehouse SET product_name = ?, quantity = ? WHERE own_name = ? AND product_name = ?';
            db.query(updateWarehouseQuery, [product_name, quantity, seller_username, original_product_name], (err) => {
                if (err) {
                    console.error('更新倉庫失敗:', err);
                    return res.status(500).json({ message: '伺服器錯誤' });
                }

                res.json({ success: true, message: '商品更新成功' });
            });
        }
    });
});

// 下架商品API
app.post('/unlist-product', (req, res) => {
    const { product_name, quantity, price, seller_username } = req.body;

    // 驗證必填欄位
    if (!product_name || !quantity || !price || !seller_username) {
        return res.status(400).json({ message: '缺少必要的欄位' });
    }

    // 從 products 表中查詢商品
    const getProductQuery = 'SELECT * FROM products WHERE seller_username = ? AND product_name = ? AND price = ?';
    db.query(getProductQuery, [seller_username, product_name, price], (err, results) => {
        if (err) {
            console.error('查詢販售中的商品失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        if (results.length === 0) {
            return res.status(404).json({ message: '商品未找到' });
        }

        const product = results[0];

        // 將商品插入到 warehouse 表中
        const addWarehouseQuery = 'INSERT INTO warehouse (own_name, product_name, quantity) VALUES (?, ?, ?) ON DUPLICATE KEY UPDATE quantity = quantity + VALUES(quantity)';
        db.query(addWarehouseQuery, [product.seller_username, product.product_name, product.quantity], (err) => {
            if (err) {
                console.error('新增倉庫商品失敗:', err);
                return res.status(500).json({ message: '伺服器錯誤' });
            }

            // 從 products 表中刪除商品
            const deleteProductQuery = 'DELETE FROM products WHERE seller_username = ? AND product_name = ? AND price = ?';
            db.query(deleteProductQuery, [seller_username, product_name, price], (err) => {
                if (err) {
                    console.error('刪除販售中的商品失敗:', err);
                    return res.status(500).json({ message: '伺服器錯誤' });
                }

                res.json({ success: true, message: '商品下架成功' });
            });
        });
    });
});

// 獲取販售中的商品列表API
app.get('/get-selling-products', (req, res) => {
    const username = req.session.username;
    const { sortCriteria = 'product_name', sortOrder = 'asc' } = req.query;

    if (!username) {
        return res.status(401).json({ message: '未登入' });
    }

    const validSortCriteria = ['product_name', 'price', 'quantity'];
    if (!validSortCriteria.includes(sortCriteria)) {
        return res.status(400).json({ message: '無效的排序標準' });
    }

    const validSortOrder = ['asc', 'desc'];
    if (!validSortOrder.includes(sortOrder)) {
        return res.status(400).json({ message: '無效的排序順序' });
    }

    const query = `SELECT * FROM products WHERE seller_username = ? ORDER BY ${sortCriteria} ${sortOrder}`;
    db.query(query, [username], (err, results) => {
        if (err) {
            console.error('查詢販售中的商品列表失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        res.json(results);
    });
});

// 搜尋販售中的商品API
app.get('/search-selling-products', (req, res) => {
    const username = req.session.username;
    const keyword = req.query.keyword;
    const criteria = req.query.criteria;

    if (!username) {
        return res.status(401).json({ message: '未登入' });
    }

    if (!keyword) {
        return res.status(400).json({ message: '缺少關鍵字' });
    }

    let query;
    if (criteria === 'product_name') {
        query = 'SELECT * FROM products WHERE seller_username = ? AND product_name LIKE ?';
    } else if (criteria === 'price') {
        query = 'SELECT * FROM products WHERE seller_username = ? AND price = ?';
    } else if (criteria === 'quantity') {
        query = 'SELECT * FROM products WHERE seller_username = ? AND quantity = ?';
    } else {
        return res.status(400).json({ message: '無效的搜尋標準' });
    }

    db.query(query, [username, criteria === 'product_name' ? `%${keyword}%` : keyword], (err, results) => {
        if (err) {
            console.error('搜尋販售中的商品失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        res.json(results);
    });
});

// 商品數量更新API
app.post('/update-num', (req, res) => {
    const { product_id } = req.body;

    // 驗證商品ID是否有效
    if (!product_id || isNaN(product_id)) {
        return res.status(400).json({ message: '商品ID無效' });
    }

    const updateQuery = 'UPDATE products SET quantity = quantity - 1 WHERE product_id = ? AND quantity > 0';
    db.query(updateQuery, [product_id], (err, result) => {
        if (err) {
            console.error('資料庫更新失敗:', err);
            return res.status(500).send('資料庫更新失敗');
        }

        if (result.affectedRows === 0) {
            return res.status(400).json({ message: '商品庫存不足，無法更新' });
        }

        // 查詢更新後的資料
        db.query('SELECT * FROM products WHERE product_id = ?', [product_id], (err, results) => {
            if (err) {
                console.error('查詢失敗:', err);
                return res.status(500).send('查詢更新後資料失敗');
            }
            res.json(results[0]); // 返回更新後的資料
        });
    });
});

// 頁面跳轉至購買頁面
app.get('/home', (req, res) => {
    if (req.session.loggedIn) {
        res.sendFile(path.join(__dirname, 'public', 'home.html')); // 返回購買頁面
    } else {
        res.redirect('/login_register.html'); // 如果未登入，跳轉回登入頁面
    }
});
// 購買商品API
app.post('/buy-product', (req, res) => {
    const { productId, quantity, productPrice } = req.body;
    const buyer_username = req.session.username;

    if (!buyer_username) {
        return res.status(401).json({ message: '未登入' });
    }

    // 查詢用戶餘額
    const getUserBalanceQuery = 'SELECT balance FROM user_profiles WHERE username = ?';
    db.query(getUserBalanceQuery, [buyer_username], (err, results) => {
        if (err) {
            console.error('查詢用戶餘額失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        if (results.length === 0) {
            return res.status(404).json({ message: '用戶未找到' });
        }

        const userBalance = results[0].balance;
        const totalCost = quantity * productPrice;

        if (userBalance < totalCost) {
            return res.status(400).json({ message: '餘額不足' });
        }

        // 查詢商品剩餘數量
        const getProductQuantityQuery = 'SELECT quantity FROM products WHERE product_id = ?';
        db.query(getProductQuantityQuery, [productId], (err, results) => {
            if (err) {
                console.error('查詢商品數量失敗:', err);
                return res.status(500).json({ message: '伺服器錯誤' });
            }

            if (results.length === 0) {
                return res.status(404).json({ message: '商品未找到' });
            }

            const remainingQuantity = results[0].quantity;

            if (quantity > remainingQuantity) {
                return res.status(400).json({ message: '購買數量超過剩餘數量' });
            }

            // 更新用戶餘額
            const updateUserBalanceQuery = 'UPDATE user_profiles SET balance = balance - ? WHERE username = ?';
            db.query(updateUserBalanceQuery, [totalCost, buyer_username], (err) => {
                if (err) {
                    console.error('更新用戶餘額失敗:', err);
                    return res.status(500).json({ message: '伺服器錯誤' });
                }

                // 更新商品庫存
                const updateProductQuantityQuery = 'UPDATE products SET quantity = quantity - ? WHERE product_id = ?';
                db.query(updateProductQuantityQuery, [quantity, productId], (err) => {
                    if (err) {
                        console.error('更新商品庫存失敗:', err);
                        return res.status(500).json({ message: '伺服器錯誤' });
                    }

                    // 查詢賣家名稱
                    const getSellerQuery = 'SELECT seller_username, product_name FROM products WHERE product_id = ?';
                    db.query(getSellerQuery, [productId], (err, results) => {
                        if (err) {
                            console.error('查詢賣家名稱失敗:', err);
                            return res.status(500).json({ message: '伺服器錯誤' });
                        }

                        if (results.length === 0) {
                            return res.status(404).json({ message: '商品未找到' });
                        }

                        const seller_username = results[0].seller_username;
                        const product_name = results[0].product_name;

                        // 將商品加入到買家的倉庫中
                        const addToWarehouseQuery = `
                            INSERT INTO warehouse (own_name, product_name, quantity)
                            VALUES (?, ?, ?)
                            ON DUPLICATE KEY UPDATE quantity = quantity + VALUES(quantity)
                        `;
                        db.query(addToWarehouseQuery, [buyer_username, product_name, quantity], (err) => {
                            if (err) {
                                console.error('加入倉庫失敗:', err);
                                return res.status(500).json({ message: '伺服器錯誤' });
                            }

                            // 檢查商品是否需要下架
                            const checkProductQuantityQuery = 'SELECT quantity FROM products WHERE product_id = ?';
                            db.query(checkProductQuantityQuery, [productId], (err, results) => {
                                if (err) {
                                    console.error('檢查商品數量失敗:', err);
                                    return res.status(500).json({ message: '伺服器錯誤' });
                                }

                                if (results.length === 0) {
                                    return res.status(404).json({ message: '商品未找到' });
                                }

                                const remainingQuantity = results[0].quantity;

                                if (remainingQuantity <= 0) {
                                    // 下架商品
                                    const deleteProductQuery = 'DELETE FROM products WHERE product_id = ?';
                                    db.query(deleteProductQuery, [productId], (err) => {
                                        if (err) {
                                            console.error('下架商品失敗:', err);
                                            return res.status(500).json({ message: '伺服器錯誤' });
                                        }

                                        // 移除賣家倉庫中的此物品
                                        const deleteFromWarehouseQuery = 'DELETE FROM warehouse WHERE own_name = ? AND product_name = ?';
                                        db.query(deleteFromWarehouseQuery, [seller_username, product_name], (err) => {
                                            if (err) {
                                                console.error('移除賣家倉庫物品失敗:', err);
                                                return res.status(500).json({ message: '伺服器錯誤' });
                                            }

                                            // 更新賣家餘額
                                            const updateSellerBalanceQuery = 'UPDATE user_profiles SET balance = balance + ? WHERE username = ?';
                                            db.query(updateSellerBalanceQuery, [totalCost, seller_username], (err) => {
                                                if (err) {
                                                    console.error('更新賣家餘額失敗:', err);
                                                    return res.status(500).json({ message: '伺服器錯誤' });
                                                }

                                                res.json({ success: true, message: '購買成功並下架商品', seller: seller_username });
                                            });
                                        });
                                    });
                                } else {
                                    // 更新賣家餘額
                                    const updateSellerBalanceQuery = 'UPDATE user_profiles SET balance = balance + ? WHERE username = ?';
                                    db.query(updateSellerBalanceQuery, [totalCost, seller_username], (err) => {
                                        if (err) {
                                            console.error('更新賣家餘額失敗:', err);
                                            return res.status(500).json({ message: '伺服器錯誤' });
                                        }

                                        res.json({ success: true, message: '購買成功', seller: seller_username });
                                    });
                                }
                            });
                        });
                    });
                });
            });
        });
    });
});

// 打工API
app.post('/work', (req, res) => {
    const { username } = req.body;

    if (!username) {
        return res.status(400).json({ message: '缺少必要的欄位' });
    }

    const updateBalanceQuery = 'UPDATE user_profiles SET balance = balance + 100 WHERE username = ?';
    db.query(updateBalanceQuery, [username], (err) => {
        if (err) {
            console.error('更新用戶餘額失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤' });
        }

        res.json({ success: true, message: '打工成功，獲得100元' });
    });
});

// 刪除帳號API
app.post('/delete-account', (req, res) => {
    const { username } = req.body;

    if (!username) {
        return res.status(400).json({ message: '缺少必要的欄位' });
    }

    const deleteUserQuery = 'DELETE FROM users WHERE username = ?';
    db.query(deleteUserQuery, [username], (err) => {
        if (err) {
            console.error('刪除帳號失敗:', err);
            return res.status(500).json({ message: '伺服器錯誤，無法刪除帳號' });
        }

        // 刪除成功後，清除會話資料
        req.session.destroy((err) => {
            if (err) {
                return res.status(500).json({ message: '刪除帳號成功，但登出失敗' });
            }
            res.json({ success: true, message: '帳號已刪除' });
        });
    });
});

// 啟動伺服器
app.listen(port, () => {
    console.log(`伺服器運行中： http://localhost:${port}`);
});
