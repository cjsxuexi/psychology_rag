#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

PROJECT_BASE = os.getenv("RAG_PROJECT_BASE") or os.getenv("RAG_DEPLOY_BASE")


def get_project_base_directory(*args):
    global PROJECT_BASE
    if PROJECT_BASE is None:
        # 获取对应的根路径
        PROJECT_BASE = os.path.abspath(
            # 获取当前文件所在文件夹的父文件夹的绝对路径
            os.path.join(
                # __file__ 表示当前文件.无法在notebook直接使用
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
            )
        )

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


def get_resource_path(file_name):
    return os.path.join(get_project_base_directory(), "resource_package", file_name)

def get_config_path(file_name):
    return os.path.join(get_project_base_directory(), "config", file_name)


def get_config(file_name, encoding='utf-8'):
    """
    获取配置文件内容
    
    Args:
        file_name (str): 配置文件名
        encoding (str): 文件编码，默认为utf-8
    
    Returns:
        str: 配置文件内容
        
    Raises:
        FileNotFoundError: 当配置文件不存在时
        IOError: 当读取文件出错时
    """
    config_path = get_config_path(file_name)
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding=encoding) as f:
            content = f.read()
        return content
    except Exception as e:
        raise IOError(f"读取配置文件失败 {config_path}: {str(e)}")

def get_storage_path(file_name):
    return os.path.join(get_project_base_directory(), "resource_package", "storage", file_name)

# base：相对路径、绝对路径都ok
def traversal_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            # yield 返回当前值后暂停，直到下次迭代
            yield fullname


def file_path():
    return os.path.realpath(__file__)
