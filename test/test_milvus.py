from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

if __name__ == "__main__":
    # 1. 连接到本地 Milvus
    print("正在连接 Milvus...")
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("✅ Milvus 连接成功！")

    # 2. 定义集合字段（类似数据库表结构）
    book_id = FieldSchema(name="book_id", dtype=DataType.INT64, is_primary=True)
    word_count = FieldSchema(name="word_count", dtype=DataType.INT64)
    book_intro = FieldSchema(name="book_intro", dtype=DataType.FLOAT_VECTOR, dim=8)  # 8维向量
    schema = CollectionSchema(fields=[book_id, word_count, book_intro], description="测试书籍集合")

    # 3. 创建集合
    collection_name = "book_test"
    print(f"正在创建集合 {collection_name}...")
    collection = Collection(name=collection_name, schema=schema)
    print(f"✅ 集合 {collection_name} 创建成功！")

    # 4. 插入测试数据
    import random

    data = [
        [i for i in range(10)],  # book_id
        [random.randint(1000, 10000) for _ in range(10)],  # word_count
        [[random.random() for _ in range(8)] for _ in range(10)]  # 8维向量
    ]
    print("正在插入测试数据...")
    insert_result = collection.insert(data)
    print(f"✅ 成功插入 {insert_result.insert_count} 条数据！")

    # ==========================================
    # 🔧 修复点：必须为向量字段创建索引！
    # ==========================================
    print("正在为向量字段创建索引...")
    index_params = {
        "metric_type": "L2",  # 距离度量方式：L2(欧氏距离)、IP(内积)
        "index_type": "IVF_FLAT",  # 索引类型：IVF_FLAT(最常用的倒排索引)
        "params": {"nlist": 128}  # IVF参数：聚类中心数量，数据量大时可设为1024
    }
    collection.create_index(
        field_name="book_intro",  # 要创建索引的向量字段名
        index_params=index_params
    )
    print("✅ 索引创建成功！")
    # ==========================================

    # 5. 加载集合到内存（必须加载才能查询）
    print("正在加载集合到内存...")
    collection.load()
    print("✅ 集合加载成功！")

    # 6. 执行向量查询
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    query_vector = [random.random() for _ in range(8)]  # 随机生成查询向量
    print("正在执行向量查询...")
    results = collection.search(
        data=[query_vector],
        anns_field="book_intro",
        param=search_params,
        limit=3,  # 返回最相似的3条结果
        expr=None,
        output_fields=["book_id", "word_count"]
    )

    # 7. 打印查询结果
    print("\n🎉 查询成功！最相似的3本书：")
    for hits in results:
        for hit in hits:
            print(f"书ID: {hit.entity.get('book_id')}, 字数: {hit.entity.get('word_count')}, 距离: {hit.distance}")

    # 8. 删除刚插入的全部数据
    print("\n🗑️  正在删除刚插入的全部数据...")
    
    # 方法1: 使用 delete 方法删除指定ID的数据
    # 获取刚插入的所有book_id
    all_book_ids = [i for i in range(10)]
    
    # 构造删除表达式
    delete_expr = f"book_id in {all_book_ids}"
    print(f"删除条件: {delete_expr}")
    
    # 执行删除操作
    delete_result = collection.delete(expr=delete_expr)
    print(f"✅ 删除操作完成！受影响的实体数: {delete_result.delete_count}")
    
    # 方法2: 可选的清空整个集合的方法（注释掉，避免误操作）
    # collection.delete(expr="book_id >= 0")  # 删除所有记录
    
    # 验证删除结果
    print("\n🔍 验证删除结果...")
    try:
        # 重新加载集合以反映删除操作
        collection.release()  # 先释放
        collection.load()     # 重新加载
        
        # 查询剩余数据数量
        query_result = collection.query(
            expr="book_id >= 0",
            output_fields=["book_id"]
        )
        remaining_count = len(query_result)
        print(f"删除后剩余数据条数: {remaining_count}")
        
        if remaining_count == 0:
            print("✅ 数据已全部删除成功！")
        else:
            print(f"⚠️  仍有 {remaining_count} 条数据未被删除")
            
    except Exception as e:
        print(f"验证删除结果时出错: {e}")
    
    # 9. 断开连接
    connections.disconnect("default")
