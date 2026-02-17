# 环境变量配置使用指南

## 配置文件位置

MySQL数据库配置现在存储在 `config/.env` 文件中。

## 配置项说明

```env
# MySQL数据库配置
MYSQL_HOST=127.0.0.1      # 数据库主机地址
MYSQL_PORT=5455           # 数据库端口
MYSQL_DATABASE=my_rag     # 数据库名称
MYSQL_USER=root           # 用户名
MYSQL_PASSWORD=infini_rag_flow  # 密码
```

## 使用方式

### 1. 默认使用环境变量（推荐）

```python
from src.database.mysql_storage import MySQLStorage

# 自动从config/.env加载配置
storage = MySQLStorage()
storage.connect()
```

### 2. 使用上下文管理器

```python
from src.database.mysql_storage import MySQLStorageManager

# 自动管理连接和断开
with MySQLStorageManager() as storage:
    # 执行数据库操作
    stats = storage.get_statistics()
    print(stats)
```

### 3. 混合配置（部分覆盖）

```python
# 环境变量提供基础配置，代码中覆盖特定项
custom_config = {
    'database': 'production_db',  # 覆盖数据库名
    'user': 'app_user'            # 覆盖用户名
}

storage = MySQLStorage(config=custom_config, use_env=True)
# host, port, password 来自环境变量
# database, user 来自配置字典
```

### 4. 完全使用代码配置

```python
# 不使用环境变量，完全通过代码配置
config = {
    'host': 'localhost',
    'port': 3306,
    'database': 'my_app',
    'user': 'admin',
    'password': 'secret'
}

storage = MySQLStorage(config=config, use_env=False)
```

### 5. 自定义环境变量文件路径

```python
from src.database.mysql_storage import load_env_config

# 使用自定义的环境变量文件
custom_env_path = '/path/to/custom.env'
config = load_env_config(custom_env_path)
storage = MySQLStorage(config=config)
```

## 配置优先级

当同时使用环境变量和配置字典时，优先级如下：
1. **配置字典中的值**（最高优先级）
2. **环境变量中的值**
3. **默认值**（最低优先级）

## 安全建议

1. **不要将密码提交到版本控制**
   - 在 `.gitignore` 中添加 `config/.env`
   - 提供 `config/.env.example` 作为模板

2. **生产环境配置**
   ```bash
   # 生产环境建议使用更安全的方式
   export MYSQL_PASSWORD=$(cat /secure/location/password.txt)
   ```

3. **权限设置**
   ```bash
   chmod 600 config/.env  # 限制文件访问权限
   ```

## 故障排除

### 1. 配置文件未找到
```
错误: FileNotFoundError
解决: 确保 config/.env 文件存在
```

### 2. 连接失败
```
错误: MySQL连接失败
解决: 检查配置项是否正确，网络是否可达
```

### 3. 权限问题
```
错误: Access denied
解决: 检查用户名和密码是否正确
```

## 测试配置

运行测试脚本验证配置是否正确：

```bash
python test/test_env_config.py
```

这将测试：
- 环境变量加载功能
- 自定义路径配置
- MySQL连接测试
- 配置混合功能