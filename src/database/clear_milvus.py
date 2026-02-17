from pymilvus import connections, utility

if __name__ == "__main__":
    # 1. 连接到 Milvus
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("✅ 已连接到 Milvus")

    # 2. 获取所有集合名称
    all_collections = utility.list_collections()
    print(f"📋 当前共有 {len(all_collections)} 个集合：{all_collections}")

    if not all_collections:
        print("ℹ️ Milvus 已经是空的，无需清理")
    else:
        # 3. 循环删除每个集合
        for coll_name in all_collections:
            print(f"🗑️  正在删除集合：{coll_name}")
            utility.drop_collection(coll_name)

        print("\n🎉 所有集合已删除完毕！Milvus 已清空。")

    # 4. 断开连接
    connections.disconnect("default")