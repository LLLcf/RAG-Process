import os
import pandas as pd
from tqdm import tqdm
from build_rag import FullyOptimizedQASystem, EnhancedConfig

def main():
    """主函数 - 批量处理测试集问题，生成问答结果"""

    # --- 路径配置 ---
    DOC_PATH = "../data/"  
    TEST_PATH = "../data/科目二初赛集人物名单-50个.xlsx"
    OUTPUT_PATH = "../data/result2.csv"

    # 1. 读取测试集
    print("读取测试集...")
    try:
        test_data = pd.read_excel(TEST_PATH, header=None)
        test_data.columns = ['name']
        print(f"成功读取Excel格式测试集: {len(test_data)} 条")
    except Exception as e:
        print(f"Excel读取失败，尝试CSV格式...")
        TEST_PATH = "../data/科目二初赛集人物名单-50个.csv"
        try:
            test_data = pd.read_csv(TEST_PATH, header=None)
            test_data.columns = ['name']
            print(f"成功读取CSV格式测试集: {len(test_data)} 条")
        except Exception as e2:
            print(f"测试集读取完全失败: {e2}")
            return

    # 2. 初始化问答系统
    print("\n" + "=" * 40)
    print("初始化问答系统...")
    print("=" * 40)
    
    qa = FullyOptimizedQASystem()
    qa.initialize(DOC_PATH)

    # 3. 批量处理问答
    print("\n" + "=" * 40)
    print("开始批量生成人物画像...")
    print("=" * 40 + "\n")

    results = []
    
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="处理进度"):
        name = str(row['name']).strip()
        if not name or str(name).lower() == 'nan':
            continue
            
        try:
            # 生成画像字典
            profile = qa.generate_person_profile(name)
            
            # 确保包含所有标准列，缺失的填 "未知"
            final_record = {col: "未知" for col in EnhancedConfig.OUTPUT_COLUMNS}
            final_record.update(profile)
            print('答案\n', final_record)
            results.append(final_record)
            
            # 简略打印提取的原文名以供观察
            print(f"已处理: {name} -> 原文名: {profile.get('姓名（原文）', '未知')}")
            
        except Exception as e:
            print(f"处理 {name} 时发生错误: {e}")
            empty_record = {col: "处理失败" for col in EnhancedConfig.OUTPUT_COLUMNS}
            empty_record['姓名（中文）'] = name
            results.append(empty_record)
        # break
    # 4. 保存最终结果
    print("\n" + "=" * 40)
    print("正在保存最终结果...")
    
    if results:
        result_df = pd.DataFrame(results)
        
        # 再次确保列顺序
        result_df = result_df[EnhancedConfig.OUTPUT_COLUMNS]
        
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        result_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        
        print(f"✓ 最终结果已保存至: {OUTPUT_PATH}")
        print(f"  共处理 {len(result_df)} 条数据")
    else:
        print("未生成数据。")

if __name__ == "__main__":
    main()