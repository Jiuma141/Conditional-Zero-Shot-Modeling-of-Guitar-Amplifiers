import os
import pandas as pd
from pathlib import Path

def consolidate_amp_space(root_dir: str, output_csv: str = "all_data.csv"):
    dfs = []
    root = Path(root_dir)
    for csv_path in root.rglob("*.csv"):
        s = str(csv_path)
        # 跳过 macOS metadata 目录 和 Jupyter 检查点
        if "__MACOSX" in s or ".ipynb_checkpoints" in s:
            continue

        try:
            df = pd.read_csv(csv_path, encoding='utf8')
        except Exception as e:
            print(f"⚠️ 无法读取 {csv_path}: {e}")
            continue

        # 统一小写列名
        df.columns = df.columns.str.lower()

        # 取父目录名作为 model
        parent_name = csv_path.parent.name
        df["model"] = parent_name

        for col in ("filename", "di_file"):
            if col in df.columns:
                def fix_path(fn: str) -> str:
                    # 绝对路径直接返回
                    if os.path.isabs(fn):
                        return fn
                    fn = fn.replace("\\", "/")
                    prefix = parent_name + "/"
                    if fn.startswith(prefix):
                        fn = fn[len(prefix):]
                    return str((csv_path.parent / fn).resolve())
                df[col] = df[col].astype(str).apply(fix_path)

        dfs.append(df)

    if not dfs:
        print("❌ 未找到任何 CSV 文件")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(output_csv, index=False)
    print(f"✅ 合并完成，共 {len(full_df)} 行，已保存到 {output_csv}")

if __name__ == "__main__":
    consolidate_amp_space("/root/autodl-tmp/unzipped", output_csv="all_data.csv")
