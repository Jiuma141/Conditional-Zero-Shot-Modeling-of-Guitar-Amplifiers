# split_all_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

def split_all_data(
    csv_path: str,
    out_train: str = "train.csv",
    out_val:   str = "val.csv",
    out_test:  str = "test.csv",
    val_frac:  float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
):
    """
    1) 读 all_data.csv
    2) 过滤掉 gain/treble/mid/bass or gain 任意一列缺失的行
    3) 基于 device 做分层 split: 先划 test，再在剩下的划 val
    4) 保存到 train.csv / val.csv / test.csv
    """
    # 1) 读取
    df = pd.read_csv(csv_path)
    print(f"原始行数：{len(df)}")

    # # 2) 只保留四个参数都不缺失的行
    # required = ["gain","treble","mid","bass"]
    # df = df.dropna(subset=required, how="any").reset_index(drop=True)
    # print(f"过滤后 (gain/treble/mid/bass 都有值) 行数：{len(df)}")
    
    # 2) 只保留gain不缺失的行
    required = ["gain"]
    df = df.dropna(subset=required, how="any").reset_index(drop=True)
    print(f"过滤后 (gain有值) 行数：{len(df)}")
    

    # 3) 如果 device 不是数字，则先给它打标签
    if df["model"].dtype == object:
        df["model"] = df["model"].astype(str)
        df["device_idx"] = df["model"].factorize()[0]
    else:
        df["device_idx"] = df["model"].astype(int)

    # 4) 先划出 test
    idx = df.index.values
    test_idx = train_idx_val_idx = None
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=test_frac,
        stratify=df["device_idx"],
        random_state=random_state,
    )
    # 再从 train_val 划出 val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_frac/(1-test_frac),
        stratify=df.loc[train_val_idx, "device_idx"],
        random_state=random_state,
    )

    # 5) 保存
    df.loc[train_idx].drop(columns="device_idx").to_csv(out_train, index=False)
    df.loc[val_idx].  drop(columns="device_idx").to_csv(out_val,   index=False)
    df.loc[test_idx]. drop(columns="device_idx").to_csv(out_test,  index=False)

    print(f"已保存：")
    print(f"  train -> {out_train}, 行数 {len(train_idx)}")
    print(f"  val   -> {out_val}, 行数 {len(val_idx)}")
    print(f"  test  -> {out_test}, 行数 {len(test_idx)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str, default="all_data.csv")
    parser.add_argument("--train_out",  type=str, default="train.csv")
    parser.add_argument("--val_out",    type=str, default="val.csv")
    parser.add_argument("--test_out",   type=str, default="test.csv")
    parser.add_argument("--val_frac",   type=float, default=0.1)
    parser.add_argument("--test_frac",  type=float, default=0.1)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    split_all_data(
        csv_path     = args.csv,
        out_train    = args.train_out,
        out_val      = args.val_out,
        out_test     = args.test_out,
        val_frac     = args.val_frac,
        test_frac    = args.test_frac,
        random_state = args.seed,
    )
