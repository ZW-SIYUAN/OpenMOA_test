#!/usr/bin/env python3
import sys, csv, os

def csv2arff(csv_path, arff_path=None):
    if arff_path is None:
        arff_path = csv_path.replace('.csv', '.arff')
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header, data = rows[0], rows[1:]          # 第一行当属性名
    relation = os.path.splitext(os.path.basename(csv_path))[0]

    with open(arff_path, 'w') as fw:
        fw.write(f'@relation {relation}\n\n')
        # 全部属性设为 numeric（71 维连续）
        for name in header[:-1]:
            fw.write(f'@attribute {name} numeric\n')
        # 最后一列是类别 {0,1}
        fw.write(f'@attribute class {{0,1}}\n\n@data\n')
        # 逐行写入数据
        for row in data:
            fw.write(','.join(row) + '\n')
    print(f'✅ 已生成 {arff_path}')

if __name__ == '__main__':
    csv2arff(sys.argv[1])   # 命令行：python csv2arff.py australian.csv