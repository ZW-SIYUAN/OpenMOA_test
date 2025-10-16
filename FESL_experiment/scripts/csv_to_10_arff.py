#!/usr/bin/env python3
"""
将CSV文件夹中所有文件根据10个perm列转换为ARFF文件

用法:
    python csv_to_10_arff.py input_dir/ output_dir/
    python csv_to_10_arff.py input.csv output_dir/  (单文件)
"""

import os
import sys
import pandas as pd
from pathlib import Path


def csv_to_10_arff_single(csv_path, output_dir):
    """
    将单个CSV转换为10个ARFF文件
    """
    df = pd.read_csv(csv_path)
    dataset_name = Path(csv_path).stem
    
    # 检测特征列
    x1_cols = [col for col in df.columns if col.startswith('X1_')]
    x2_cols = [col for col in df.columns if col.startswith('X2_')]
    label_col = 'label'
    
    if not x1_cols or not x2_cols or label_col not in df.columns:
        print(f"⚠ 跳过 {csv_path}: CSV格式错误")
        return
    
    feature_cols = x1_cols + x2_cols
    label_values = sorted(df[label_col].unique())
    
    # 为每个perm生成ARFF
    for perm_idx in range(1, 11):
        perm_col = f'perm_{perm_idx}'
        
        if perm_col not in df.columns:
            print(f"⚠ {dataset_name}: 找不到列 {perm_col}")
            continue
        
        # 按照perm打乱（perm从1开始，需要转换为0-based索引）
        perm_indices = df[perm_col].values.astype(int) - 1
        df_shuffled = df.iloc[perm_indices].reset_index(drop=True)
        
        # 生成ARFF文件
        arff_path = os.path.join(output_dir, f'{dataset_name}_perm_{perm_idx}.arff')
        
        with open(arff_path, 'w') as f:
            # 写头
            f.write(f'@relation {dataset_name}\n\n')
            for col in feature_cols:
                f.write(f'@attribute {col} numeric\n')
            label_str = ','.join(str(v) for v in label_values)
            f.write(f'@attribute class {{{label_str}}}\n\n@data\n')
            
            # 写数据
            for _, row in df_shuffled.iterrows():
                values = [str(row[col]) for col in feature_cols]
                label = str(int(row[label_col]))
                f.write(','.join(values) + ',' + label + '\n')
        
        print(f"  ✓ {os.path.basename(arff_path)}")


def csv_to_10_arff_batch(csv_dir, output_dir):
    """
    批量转换整个目录的CSV文件
    """
    csv_files = sorted(Path(csv_dir).glob('*.csv'))
    
    if not csv_files:
        print(f"错误: {csv_dir} 中找不到CSV文件")
        return
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"找到 {len(csv_files)} 个CSV文件\n")
    
    for csv_file in csv_files:
        print(f"处理: {csv_file.name}")
        csv_to_10_arff_single(str(csv_file), output_dir)
    
    print(f"\n✅ 转换完成！ARFF文件已保存到 {output_dir}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法:")
        print("  批量: python csv_to_10_arff.py <csv_dir> <output_dir>")
        print("  单文件: python csv_to_10_arff.py <input.csv> <output_dir>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"错误: {input_path} 不存在")
        sys.exit(1)
    
    try:
        if os.path.isdir(input_path):
            # 目录模式：批量处理
            csv_to_10_arff_batch(input_path, output_dir)
        else:
            # 文件模式：单文件处理
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            print(f"处理: {Path(input_path).name}")
            csv_to_10_arff_single(input_path, output_dir)
            print(f"\n✅ 转换完成！ARFF文件已保存到 {output_dir}")
    
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)