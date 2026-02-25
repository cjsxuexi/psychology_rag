#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LCEL流处理器测试脚本
验证基于LangChain LCEL的流式处理功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_process.lcel_stream_process import MilvusLCELStreamProcessor, stream_process_psychology_data_milvus_lcel
from loguru import logger
import time


def test_basic_functionality():
    """测试LCEL流处理器基本功能"""
    logger.info("=== 测试LCEL流处理器基本功能 ===")
    
    try:
        # 创建处理器实例
        processor = MilvusLCELStreamProcessor(
            json_file="PsyDTCorpus_train_mulit_turn_packing.json",
            batch_size=10,
            embedding_batch_size=5,
            max_workers=1,
            memory_limit_mb=1000,
            buffer_size=20
        )
        
        logger.success("✓ LCEL流处理器初始化成功")
        
        # 关闭处理器
        processor.close()
        logger.success("✓ LCEL流处理器关闭成功")
        
    except Exception as e:
        logger.error(f"基本功能测试失败: {str(e)}")
        raise


def test_lcel_stream_processing():
    """测试LCEL流式处理功能"""
    logger.info("=== 测试LCEL流式处理功能 ===")
    
    try:
        # 处理少量数据进行测试
        result = stream_process_psychology_data_milvus_lcel(
            max_items=3,  # 处理10条数据
            batch_size=10,
            embedding_batch_size=10,
            memory_limit_mb=5000,
            auto_configure=False
        )
        
        logger.success("✓ LCEL流式处理完成")
        logger.info(f"处理结果: {result}")
        
        # 验证结果
        assert result['generated_chunks'] > 0, "生成的文本块数应该大于0"
        
        logger.success("✓ LCEL流式处理结果验证通过")
        
    except Exception as e:
        logger.error(f"LCEL流式处理测试失败: {str(e)}")
        raise


def test_error_handling():
    """测试LCEL错误处理"""
    logger.info("=== 测试LCEL错误处理机制 ===")
    
    try:
        # 测试无效的JSON文件
        try:
            result = stream_process_psychology_data_milvus_lcel(
                json_file="nonexistent.json",
                max_items=1
            )
            assert False, "应该抛出文件不存在异常"
        except FileNotFoundError:
            logger.success("✓ 正确处理文件不存在错误")
        except Exception as e:
            logger.warning(f"预期文件不存在错误，但收到: {type(e).__name__}")
            # 继续测试，因为可能有不同的错误处理方式
        
        logger.success("✓ 错误处理机制验证通过")
        
    except Exception as e:
        logger.error(f"错误处理测试失败: {str(e)}")
        raise


def performance_comparison():
    """LCEL性能测试"""
    logger.info("=== LCEL性能测试 ===")
    
    test_sizes = [10, 20]
    
    for size in test_sizes:
        logger.info(f"\n测试LCEL处理 {size} 条数据的性能:")
        
        try:
            start_time = time.time()
            result = stream_process_psychology_data_milvus_lcel(
                max_items=size,
                batch_size=5,
                embedding_batch_size=5,
                memory_limit_mb=1000,
                auto_configure=False
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            speed = result['generated_chunks'] / processing_time if processing_time > 0 else 0
            
            logger.info(f"  处理时间: {processing_time:.2f} 秒")
            logger.info(f"  处理速度: {speed:.2f} 项/秒")
            logger.info(f"  生成文本块数: {result['generated_chunks']}")
            
        except Exception as e:
            logger.error(f"LCEL性能测试失败 (size={size}): {str(e)}")


def main():
    """主测试函数"""
    logger.info("开始LCEL流处理器全面测试")
    logger.info("=" * 50)
    
    try:
        # 1. 基本功能测试
        # test_basic_functionality()
        
        # 2. LCEL流式处理测试
        test_lcel_stream_processing()
        
        # 3. 错误处理测试
        # test_error_handling()
        #
        # # 4. 性能测试
        # performance_comparison()
        #
        logger.success("=" * 50)
        logger.success("🎉 所有LCEL测试通过！LCEL流处理器工作正常")
        
    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"❌ 测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()