import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
import time

try:
    # 尝试导入，这里应该会加载您本地文件夹里的 capymoa
    # 也就是您刚修改过 __init__.py 和 _datasets.py 的那个文件夹
    import capymoa.datasets as datasets
    from capymoa.stream import MOAStream
    
    # 打印一下实际加载的路径，验证是否加载了本地代码
    print(f"✅ Successfully loaded local package from: {os.path.dirname(datasets.__file__)}")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("请检查项目根目录下是否存在 'capymoa' 或 'openmoa' 文件夹。")
    sys.exit(1)

def test_datasets():
    # 这里列出您之前确认添加的 10 个二分类数据集
    # 因为加载的是本地代码，现在 capymoa.datasets 下应该有这些类了
    binary_datasets_list = [
        datasets.RCV1,
        datasets.W8a,
        datasets.Adult,      # a8a
        datasets.Magic04,
        datasets.Spambase,
        datasets.Musk,
        datasets.SVMGuide3,
        datasets.German,
        datasets.Australian,
        datasets.Ionosphere
    ]

    print(f"\n🚀 Starting Benchmark Sanity Check for {len(binary_datasets_list)} Binary Datasets...\n")
    print(f"{'Dataset Name':<15} | {'Status':<10} | {'Samples':<10} | {'Features':<10} | {'Classes':<10}")
    print("-" * 70)

    failed_datasets = []

    for dataset_cls in binary_datasets_list:
        name = dataset_cls.__name__
        try:
            # 1. 初始化流 (会自动触发下载/解压/读取Header)
            stream: MOAStream = dataset_cls()
            
            # 2. 尝试读取第一条数据 (验证数据解析是否正常)
            instance = stream.next_instance()
            
            # 获取统计信息
            n_samples = len(stream) if hasattr(stream, '__len__') else "Unknown"
            n_features = stream.schema.get_num_attributes()
            n_classes = stream.schema.get_num_classes()
            
            # 打印成功信息
            print(f"{name:<15} | ✅ PASS    | {str(n_samples):<10} | {str(n_features):<10} | {str(n_classes):<10}")
            
            if n_classes != 2:
                print(f"  ⚠️ WARNING: {name} has {n_classes} classes (Expected 2)")

        except Exception as e:
            print(f"{name:<15} | ❌ FAIL    | {'-':<10} | {'-':<10} | {'-':<10}")
            print(f"  └── Error: {e}")
            failed_datasets.append(name)
    
    print("-" * 70)
    if not failed_datasets:
        print("\n🎉 Congratulations! All local datasets are loaded correctly.")
    else:
        print(f"\n⚠️ Found issues in {len(failed_datasets)} datasets: {', '.join(failed_datasets)}")

if __name__ == "__main__":
    test_datasets()