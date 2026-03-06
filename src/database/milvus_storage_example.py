from src.database.milvus_storage import create_milvus_storage
from pymilvus import utility, Collection, db

from src.embedding.M3EEmbedding import generate_dense_embeddings_with_m3e


def get_collection_count(database: str = "default"):
    """
    获取当前数据库中的集合数量
    
    Returns:
        int: 集合数量
    """
    # 1. 连接到 Milvus
    # 注意：根据实际 Milvus 服务配置修改用户名和密码
    storage = create_milvus_storage(user="root", password="Milvus", database=database)
    print("✅ 已连接到 Milvus")

    # 2. 获取所有集合名称
    all_collections = utility.list_collections()
    collection_count = len(all_collections)
    print(f"📋 当前共有 {collection_count} 个集合：")
    for i, coll_name in enumerate(all_collections, 1):
        print(f"   {i}. {coll_name}")

    # 3. 断开连接
    storage.close()
    
    return collection_count


def get_collection_stats(collection_name):
    """
    获取指定集合的数据条数，并展示前2条数据信息
    
    Args:
        collection_name: 集合名称
    
    Returns:
        dict: 包含数据条数和前2条数据的字典
    """
    # 1. 连接到 Milvus
    # 注意：根据实际 Milvus 服务配置修改用户名和密码
    storage = create_milvus_storage(collection_name=collection_name, user="root", password="Milvus")
    print("✅ 已连接到 Milvus")

    # 2. 检查集合是否存在
    if not utility.has_collection(collection_name):
        print(f"❌ 集合 '{collection_name}' 不存在")
        storage.close()
        return {"count": 0, "records": []}

    # 3. 获取集合
    collection = Collection(collection_name)
    
    # 4. 获取数据条数
    record_count = collection.num_entities
    print(f"📊 集合 '{collection_name}' 共有 {record_count} 条数据")
    
    # 5. 获取前2条数据
    records = []
    if record_count > 0:
        # 加载集合
        if not collection.is_loaded:
            collection.load()
        
        # 查询前2条数据
        # 注意：这里使用空表达式获取所有数据，然后限制为2条
        # 实际使用中可能需要根据集合的主键字段调整查询条件
        result = collection.query(
            expr="",
            limit=2,
            output_fields=["*"]
        )
        
        records = result
        print("📋 前2条数据信息：")
        for i, record in enumerate(records, 1):
            print(f"   第 {i} 条: {record}")

    # 6. 断开连接
    storage.close()
    
    return {"count": record_count, "records": records}


def get_database_count(database: str = "default"):
    """
    获取当前连接的数据库数量
    
    Returns:
        int: 数据库数量
    """
    # 1. 连接到 Milvus（不需要指定 database，因为要查询所有 database）
    # 注意：根据实际 Milvus 服务配置修改用户名和密码
    storage = create_milvus_storage(user="root", password="Milvus", database=database)
    print("✅ 已连接到 Milvus")

    # 2. 获取所有数据库名称
    all_databases = db.list_database()
    database_count = len(all_databases)
    print(f"📋 当前共有 {database_count} 个数据库：")
    for i, db_name in enumerate(all_databases, 1):
        print(f"   {i}. {db_name}")

    # 3. 断开连接
    storage.close()
    
    return database_count


def drop_database(database_name: str, drop_collections: bool = False) -> bool:
    """
    删除指定的数据库
    
    Args:
        database_name: 要删除的数据库名称
        drop_collections: 是否先删除数据库中的所有集合，默认为 False
    
    Returns:
        bool: 删除是否成功
    """
    # 1. 连接到 Milvus（不需要指定 database，因为要删除指定的 database）
    # 注意：根据实际 Milvus 服务配置修改用户名和密码
    storage = create_milvus_storage(user="root", password="Milvus")
    print("✅ 已连接到 Milvus")

    try:
        # 2. 检查数据库是否存在
        all_databases = db.list_database()
        if database_name not in all_databases:
            print(f"❌ 数据库 '{database_name}' 不存在")
            return False

        # 3. 如果需要先删除集合
        if drop_collections:
            print(f"🔄 正在删除数据库 '{database_name}' 中的所有集合...")
            try:
                # 切换到指定数据库
                db.using_database(database_name)
                # 获取所有集合
                collections = utility.list_collections()
                if collections:
                    for coll_name in collections:
                        print(f"   删除集合: {coll_name}")
                        utility.drop_collection(coll_name)
                    print(f"✅ 成功删除数据库 '{database_name}' 中的所有集合")
                else:
                    print(f"ℹ️  数据库 '{database_name}' 中没有集合")
            except Exception as coll_error:
                print(f"❌ 删除集合失败: {str(coll_error)}")
                return False

        # 4. 删除数据库
        db.drop_database(database_name)
        print(f"✅ 数据库 '{database_name}' 删除成功")
        return True
    except Exception as e:
        print(f"❌ 删除数据库失败: {str(e)}")
        return False
    finally:
        # 5. 断开连接
        storage.close()


def create_database(database_name: str) -> bool:
    """
    创建指定名称的数据库
    
    Args:
        database_name: 要创建的数据库名称
    
    Returns:
        bool: 创建是否成功
    """
    # 1. 连接到 Milvus（不需要指定 database，因为要创建新的 database）
    # 注意：根据实际 Milvus 服务配置修改用户名和密码
    storage = create_milvus_storage(user="root", password="Milvus")
    print("✅ 已连接到 Milvus")

    try:
        # 2. 检查数据库是否已存在
        all_databases = db.list_database()
        if database_name in all_databases:
            print(f"ℹ️  数据库 '{database_name}' 已存在")
            return True

        # 3. 创建数据库
        db.create_database(database_name)
        print(f"✅ 数据库 '{database_name}' 创建成功")
        return True
    except Exception as e:
        print(f"❌ 创建数据库失败: {str(e)}")
        return False
    finally:
        # 4. 断开连接
        storage.close()
def test_milvus_storage():
    """测试Milvus存储功能"""
    print("=== Milvus存储功能测试 ===\n")

    try:
        # 创建存储实例
        storage = create_milvus_storage()

        # 测试数据
        test_texts = [
            "这是一个测试文本块1，用于验证Milvus存储功能。",
            "这是第二个测试文本块，包含心理咨询相关内容。",
            "第三个测试文本，用于测试向量搜索功能。"
        ]

        test_metadata = [
            {'tag': '测试标签1', 'total_turns': 2},
            {'tag': '心理咨询', 'total_turns': 3},
            {'tag': '搜索测试', 'total_turns': 1}
        ]

        # 生成向量
        print("正在生成测试向量...")
        embeddings = generate_dense_embeddings_with_m3e(test_texts)
        print(f"生成了 {len(embeddings)} 个向量，维度: {embeddings[0].shape}")

        # 插入数据
        print("\n正在插入测试数据...")
        inserted_count = storage.insert_data(
            texts=test_texts,
            embeddings=embeddings,
            metadata_list=test_metadata,
            batch_size=10
        )
        print(f"成功插入 {inserted_count} 条记录")

        # 测试搜索
        print("\n正在测试搜索功能...")
        search_results = storage.search_similar(
            query_text="心理咨询技巧",
            top_k=2
        )

        print(f"搜索到 {len(search_results)} 个相似结果:")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. 距离: {result['distance']:.4f}")
            print(f"   内容: {result['content'][:50]}...")
            print(f"   标签: {result['tag']}")

        # 获取统计信息
        print("\n获取集合统计信息...")
        stats = storage.get_collection_stats()
        print(f"集合统计: {stats}")

        # 关闭连接
        storage.close()
        print("\n✅ Milvus存储测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")


if __name__ == "__main__":
    database_name = "dify_data"
    print("=== 1. 查询当前数据库数量 ===")
    get_database_count(database=database_name)
    
    print("\n=== 2. 创建新数据库 ===")
    # 测试创建 dify_date 数据库
    # create_database("dify_data")
    
    print("\n=== 3. 查询当前数据库集合数量 ===")
    get_collection_count(database=database_name)  # 暂时注释，避免因为数据库不存在而报错
    
    print("\n=== 4. 再次查询当前数据库数量 ===")
    # get_database_count()
    
    print("\n=== 5. 删除指定数据库 ===")
    # 测试删除 dify_date 数据库（如果存在），先清空集合
    # drop_database("dify_data", drop_collections=True)
    
    print("\n=== 6. 最终查询当前数据库数量 ===")
    # get_database_count()
    
    print("\n=== 7. 查询指定集合数据信息 ===")
    # 替换为实际的集合名称
    target_collection = "psychology_dialogues"
    # get_collection_stats(target_collection)