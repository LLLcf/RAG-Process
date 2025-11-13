import os
import re
import json
import time
import asyncio
import warnings
from typing import List, Dict, Tuple, Optional, AsyncIterator, Iterator, AsyncGenerator
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
import jieba
import docx
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# LlamaIndex核心组件
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    ServiceContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.llms import (
    LLM, ChatMessage, CompletionResponse, 
    CompletionResponseGen, ChatResponse, ChatResponseGen
)
from openai import OpenAI
from vllm import LLM, SamplingParams
from utils import *

# 忽略警告
warnings.filterwarnings('ignore')


def main():
    """主函数 - 批量处理测试集问题，生成问答结果"""

    DOC_PATH = "../data/文档数据"  # 政策文档存放目录
    KNOWLEDGE_GRAPH_PATH = "../data/文档数据/扬州市人工智能产业图谱.xlsx"  # 知识图谱文件路径
    TEST_PATH = "../data/初赛A榜测试集.xlsx"  # 测试集文件
    OUTPUT_PATH = "../data/output/result.csv"  # 最终结果输出路径
    
    # 1. 读取测试集
    print("读取测试集...")
    try:
        test_data = pd.read_excel(TEST_PATH)
        print(f"成功读取Excel格式测试集")
    except Exception as e:
        # Excel读取失败时，尝试读取CSV格式
        print(f"Excel读取失败: {e}，尝试CSV格式...")
        TEST_PATH = "../data/初赛A榜测试集.csv"
        try:
            test_data = pd.read_csv(TEST_PATH)
            print(f"成功读取CSV格式测试集")
        except Exception as e:
            print(f"测试集读取失败: {e}")
            return

    # 校验测试集格式
    required_cols = ['ID', 'question']
    if not all(col in test_data.columns for col in required_cols):
        print(f"✗ 测试集格式错误，需包含{required_cols}列")
        print(f"当前列: {test_data.columns.tolist()}")
        return

    print(f"✓ 共加载 {len(test_data)} 个测试问题\n")

    # 2. 初始化问答系统
    print("初始化问答系统...")
    
    qa = FullyOptimizedQASystem()
    qa.knowledge_graph_path = KNOWLEDGE_GRAPH_PATH

    qa.initialize(DOC_PATH)
    print("✓ 问答系统初始化成功")

    # 3. 批量处理问答
    print("\n" + "=" * 60)
    print("开始批量问答处理...")
    print("=" * 60 + "\n")

    results = []
    # test_data = test_data.iloc[5: 6]
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="处理进度"):
            
        qid = row['ID']
        question = str(row['question']).strip()

        answer = qa.answer(question)
        # 清理特殊字符（根据比赛要求调整）
        clean_chars = ["\t", "\r", '"', "*", "#"]
        for char in clean_chars:
            answer = answer.replace(char, "")
        # 保留必要的换行和空格
        answer = re.sub(r'\n+', '\n', answer).strip()
        
        results.append({
            'ID': qid, 
            'answer': answer})
        
        print(f"\n{'='*80}")
        print(f"问题ID: {qid}")
        print(f"问题: {question}")
        print(f"{'='*80}")
        print(f"答案: {answer}")
        print(f"{'='*80}")
        # break
        

    # 4. 保存最终结果
    print("\n" + "=" * 60)
    print("正在保存最终结果...")

    result_df = pd.DataFrame(results)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # 保存结果
    result_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    
    print(f"✓ 最终结果已保存至: {OUTPUT_PATH}")

def check_environment():
    """检查运行环境"""
    print("检查运行环境...")
    
    # 检查必要目录
    required_dirs = [
        "../data/文档数据",
        "../data/output"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"⚠ 目录不存在: {dir_path}")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✓ 已创建目录: {dir_path}")
            except Exception as e:
                print(f"✗ 创建目录失败: {e}")
        else:
            print(f"✓ 目录存在: {dir_path}")
    
    # 检查必要文件
    required_files = [
        "../data/扬州市人工智能产业图谱.xlsx",
        "../data/初赛A榜测试集.xlsx"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ 文件存在: {file_path}")
        else:
            print(f"⚠ 文件不存在: {file_path}")
    
    print("环境检查完成\n")


if __name__ == "__main__":
    # 先检查环境
    check_environment()
    
    # 运行主函数
    main()