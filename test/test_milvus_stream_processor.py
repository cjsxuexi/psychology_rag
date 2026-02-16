#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Milvus流处理器测试脚本
验证Milvus替代MySQL+FAISS的流式处理功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_process.milvus_stream_processor import MilvusStreamProcessor, stream_process_psychology_data_milvus
from loguru import logger
import time


def test_basic_functionality():
    """测试基本功能"""
    logger.info("=== 测试Milvus流处理器基本功能 ===")
    
    try:
        # 创建处理器实例
        processor = MilvusStreamProcessor(
            json_file="PsyDTCorpus_train_mulit_turn_packing.json",
            batch_size=10,
            embedding_batch_size=5,
            max_workers=1,
            memory_limit_mb=1000,
            buffer_size=20
        )
        
        logger.success("✓ Milvus流处理器初始化成功")
        # 获取统计信息
        stats = processor._get_final_stats()
        logger.info(f"统计信息: {stats}")
        
        # 关闭处理器
        processor.close()
        logger.success("✓ Milvus流处理器关闭成功")
        
    except Exception as e:
        logger.error(f"基本功能测试失败: {str(e)}")
        raise


def test_stream_processing():
    """测试流式处理功能"""
    logger.info("=== 测试流式处理功能 ===")
    
    try:
        # 处理少量数据进行测试
        result = stream_process_psychology_data_milvus(
            max_items=50,  # 处理50条数据
            batch_size=10,
            embedding_batch_size=5,
            memory_limit_mb=1000,
            auto_configure=False
        )
        
        logger.success("✓ 流式处理完成")
        logger.info(f"处理结果: {result}")
        
        # 验证结果
        assert result['processed_items'] > 0, "处理的数据项数应该大于0"
        assert result['stored_chunks'] > 0, "存储的文本块数应该大于0"
        assert 'milvus_entities' in result, "结果应该包含Milvus实体数"
        
        logger.success("✓ 流式处理结果验证通过")
        
    except Exception as e:
        logger.error(f"流式处理测试失败: {str(e)}")
        raise


def test_search_functionality():
    """测试搜索功能"""
    logger.info("=== 测试Milvus搜索功能 ===")
    
    processor = None
    try:
        # 创建处理器实例
        processor = MilvusStreamProcessor(collection_name="psychology_dialogues")
        
        # 执行搜索
        query_text = ("我不确定他是不是还爱我")
        search_results = processor.search_similar_chunks(query_text, top_k=3)
        
        logger.success("✓ Milvus搜索执行成功")
        logger.info(f"搜索查询: {query_text}")
        logger.info(f"搜索结果数量: {len(search_results)}")
        
        # 显示搜索结果
        for i, result in enumerate(search_results, 1):
            logger.info(f"结果 {i}:")
            logger.info(f"  距离: {result['distance']:.4f}")
            logger.info(f"  标签: {result['tag']}")
            logger.info(f"  content: {result['content']}")

        # 验证搜索结果
        assert len(search_results) > 0, "应该返回至少一个搜索结果"
        assert all('content' in result for result in search_results), "所有结果都应该包含文本内容"
        assert all('distance' in result for result in search_results), "所有结果都应该包含距离信息"
        
        logger.success("✓ 搜索功能验证通过")
        
    except Exception as e:
        logger.error(f"搜索功能测试失败: {str(e)}")
        raise
    finally:
        if processor:
            processor.close()


def test_data_reader_method():
    """测试_data_reader方法，按照新要求调整输出逻辑"""
    logger.info("=== 测试_data_reader方法（调整后）===")
    
    processor = None
    try:
        # 创建处理器实例
        processor = MilvusStreamProcessor(
            json_file="PsyDTCorpus_train_mulit_turn_packing.json",
            batch_size=10,
            max_workers=1,
            memory_limit_mb=1000
        )
        
        logger.success("✓ Milvus流处理器初始化成功")
        
        # 调用_data_reader方法并获取前5个数据项
        logger.info("开始读取JSON数据...")
        reader_generator = processor._data_reader(max_items=5)
        
        item_count = 0
        for item in reader_generator:
            item_count += 1
            logger.info(f"\n--- 第 {item_count} 个完整item对象 ---")
            logger.info(f"对象类型: {type(item)}")
            logger.info(f"对象键数量: {len(item) if isinstance(item, dict) else 'N/A'}")
            
            # 输出前3个键值对，根据不同类型采用不同输出策略
            if isinstance(item, dict) and len(item) > 0:
                logger.info("前3个字段内容:")
                keys_list = list(item.keys())
                for i, key in enumerate(keys_list[:3]):
                    value = item[key]
                    value_type = type(value).__name__
                    
                    if isinstance(value, list):
                        # 列表类型：打印前2条内容，最大200字
                        if len(value) > 0:
                            logger.info(f"  {key} (list[{len(value)}]):")
                            for j, list_item in enumerate(value[:2]):  # 前2条
                                if isinstance(list_item, dict):
                                    # 如果列表项是字典，显示其键名和部分内容
                                    item_keys = list(list_item.keys())
                                    item_preview = f"{{{', '.join(item_keys[:3])}}}" if len(item_keys) > 3 else str(list_item)
                                    logger.info(f"    [{j}]: {item_preview[:100]}...")
                                else:
                                    # 普通类型，直接转字符串并限制长度
                                    item_str = str(list_item)
                                    item_preview = item_str[:100] + "..." if len(item_str) > 100 else item_str
                                    logger.info(f"    [{j}]: {item_preview}")
                            if len(value) > 2:
                                logger.info(f"    ... 还有 {len(value) - 2} 条")
                        else:
                            logger.info(f"  {key} (empty list): []")
                    
                    elif isinstance(value, (str, int, float, bool)) or value is None:
                        # 普通类型：直接输出打印
                        value_str = str(value)
                        if len(value_str) > 200:
                            value_str = value_str[:200] + "..."
                        logger.info(f"  {key} ({value_type}): {value_str}")
                    
                    else:
                        # 其他复杂类型：显示类型和简要信息
                        value_str = str(value)
                        value_preview = value_str[:100] + "..." if len(value_str) > 100 else value_str
                        logger.info(f"  {key} ({value_type}): {value_preview}")
                
                if len(keys_list) > 3:
                    logger.info(f"  ... 还有 {len(keys_list) - 3} 个字段")
            
            logger.info("-" * 50)
        
        # 验证结果
        assert item_count == 5, f"应该读取5个数据项，实际读取了{item_count}个"
        logger.success(f"✓ 成功读取并展示了 {item_count} 个完整item对象")
        
    except Exception as e:
        logger.error(f"_data_reader方法测试失败: {str(e)}")
        raise
    finally:
        if processor:
            processor.close()


def test_error_handling():
    """测试错误处理"""
    logger.info("=== 测试错误处理机制 ===")
    
    try:
        # 测试无效的JSON文件
        try:
            processor = MilvusStreamProcessor(json_file="nonexistent.json")
            processor.close()
            assert False, "应该抛出文件不存在异常"
        except FileNotFoundError:
            logger.success("✓ 正确处理文件不存在错误")
        
        # 测试无效的Milvus连接
        try:
            processor = MilvusStreamProcessor(
                milvus_host="invalid_host",
                milvus_port="99999"
            )
            processor.close()
            assert False, "应该抛出连接错误"
        except Exception:
            logger.success("✓ 正确处理Milvus连接错误")
            
        logger.success("✓ 错误处理机制验证通过")
        
    except Exception as e:
        logger.error(f"错误处理测试失败: {str(e)}")
        raise


def performance_comparison():
    """性能对比测试"""
    logger.info("=== 性能对比测试 ===")
    
    test_sizes = [10, 50, 100]
    
    for size in test_sizes:
        logger.info(f"\n测试处理 {size} 条数据的性能:")
        
        try:
            start_time = time.time()
            result = stream_process_psychology_data_milvus(
                max_items=size,
                batch_size=10,
                embedding_batch_size=5,
                memory_limit_mb=1000,
                auto_configure=False
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            speed = result['processed_items'] / processing_time if processing_time > 0 else 0
            
            logger.info(f"  处理时间: {processing_time:.2f} 秒")
            logger.info(f"  处理速度: {speed:.2f} 项/秒")
            logger.info(f"  存储实体数: {result['milvus_entities']}")
            logger.info(f"  内存峰值: {result['memory_peak_mb']} MB")
            
        except Exception as e:
            logger.error(f"性能测试失败 (size={size}): {str(e)}")


def main():
    """主测试函数"""
    logger.info("开始Milvus流处理器全面测试")
    logger.info("=" * 50)
    
    try:
        # 1. _data_reader方法测试
        # test_data_reader_method()
        
        # 2. 基本功能测试
        # test_basic_functionality()
        
        # 3. 流式处理测试
        # test_stream_processing()
        
        # 4. 搜索功能测试
        test_search_functionality()
        #
        # # 5. 错误处理测试
        # test_error_handling()
        #
        # # 6. 性能对比测试
        # performance_comparison()
        #
        # logger.success("=" * 50)
        # logger.success("🎉 所有测试通过！Milvus流处理器工作正常")
        
    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"❌ 测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()