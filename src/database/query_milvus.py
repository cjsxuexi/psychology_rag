from pymilvus import connections, utility, Collection


def get_collection_count():
    """
    获取当前数据库中的集合数量
    
    Returns:
        int: 集合数量
    """
    # 1. 连接到 Milvus
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("✅ 已连接到 Milvus")

    # 2. 获取所有集合名称
    all_collections = utility.list_collections()
    collection_count = len(all_collections)
    print(f"📋 当前共有 {collection_count} 个集合：")
    for i, coll_name in enumerate(all_collections, 1):
        print(f"   {i}. {coll_name}")

    # 3. 断开连接
    connections.disconnect("default")
    
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
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("✅ 已连接到 Milvus")

    # 2. 检查集合是否存在
    if not utility.has_collection(collection_name):
        print(f"❌ 集合 '{collection_name}' 不存在")
        connections.disconnect("default")
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
    connections.disconnect("default")
    
    return {"count": record_count, "records": records}


if __name__ == "__main__":
    print("=== 1. 查询当前数据库集合数量 ===")
    get_collection_count()
    
    print("\n=== 2. 查询指定集合数据信息 ===")
    # 替换为实际的集合名称
    target_collection = "psychology_dialogues"
    get_collection_stats(target_collection)