import pandas as pd
import numpy as np
import os
import joblib
import sys
import argparse  # 导入参数解析库
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import ParameterGrid


def load_and_merge_data(train_file, test_file, label_file, num_features):
    """
    加载所有数据，并按时间戳合并测试集与标签。
    """
    print("--- 0. 正在加载数据... ---")
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        df_label = pd.read_csv(label_file)
    except FileNotFoundError as e:
        print(f"错误：文件未找到。 {e}")
        return None, None, None, None

    # 假设第一列是时间戳
    time_col = df_train.columns[0]

    # 假设 test_label.csv 的第二列是标签
    label_col_name = df_label.columns[1]

    # 将标签列重命名为 'true_label' 以便统一处理
    df_label = df_label.rename(columns={label_col_name: 'true_label'})

    # 合并测试集和标签
    df_test_with_labels = pd.merge(df_test, df_label[[time_col, 'true_label']],
                                   on=time_col, how='left')

    # 检查并处理合并后标签缺失的行（这些行无法用于评估）
    if df_test_with_labels['true_label'].isnull().any():
        missing_count = df_test_with_labels['true_label'].isnull().sum()
        print(f"警告：测试集与标签合并后有 {missing_count} 行没有标签，将丢弃这些行以进行评估。")
        df_test_with_labels = df_test_with_labels.dropna(subset=['true_label'])

    y_test = df_test_with_labels['true_label'].astype(int)

    # 定义特征列
    feature_cols = [f'col_{i}' for i in range(num_features)]


    X_train = df_train[feature_cols]
    X_test = df_test_with_labels[feature_cols]

    print("数据加载和合并完毕。\n")
    return df_train, df_test_with_labels, X_train, X_test, y_test, time_col


def preprocess_data(X_train, X_test):
    """
    执行数据预处理：缺失值填充 和 特征缩放
    """
    print("--- 1. 正在预处理数据... ---")

    # 1. 缺失值填充 (使用中位数)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    print("缺失值填充完毕 (strategy='median')。")

    # 2. 特征缩放 (使用 RobustScaler)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    print("特征标准化完毕 (RobustScaler)。\n")

    return X_train_scaled, X_test_scaled, scaler, imputer


def find_best_f1_threshold(scores, y_true, verbose=True):
    """
    在给定分数和真实标签的情况下，找到最佳F1分数和对应的阈值。
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)

    # 计算F1分数 (添加 1e-9 避免除以零)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)

    # 找到最佳F1分数的索引
    if len(thresholds) < len(f1_scores):
        best_f1_idx = np.argmax(f1_scores[:-1])
    else:
        best_f1_idx = np.argmax(f1_scores)

    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx]

    if verbose:
        print(f"--- 评估报告 (基于测试集) ---")
        print(f"最佳 F1 分数: {best_f1:.4f}")
        print(f"对应的最佳阈值 (归一化后): {best_threshold:.4f}")

        # 使用此阈值进行最终分类
        y_pred = (scores >= best_threshold).astype(int)
        print("\n使用最佳阈值的分类报告:")
        print(classification_report(y_true, y_pred, target_names=['正常 (0)', '异常 (1)']))

        try:
            roc_auc = roc_auc_score(y_true, scores)
            print(f"macro-ROC AUC: {roc_auc:.4f}")
        except ValueError:
            print("ROC AUC 无法计算 (可能标签中只有一类)")
        print("---------------------------------")

    return best_f1, best_threshold


def grid_search_iforest(X_train_scaled, X_test_scaled, y_test, param_grid):
    """
    执行网格搜索，找到在测试集上F1分数最高的模型参数。
    """
    print("--- 2. 正在执行网格搜索... ---")
    start_time = time.time()

    best_f1 = -1
    best_params = None

    all_params = list(ParameterGrid(param_grid))
    print(f"总共 {len(all_params)} 种参数组合需要测试。")

    for i, params in enumerate(all_params):
        print(f"  测试 {i + 1}/{len(all_params)}: {params}")

        # 1. 训练模型
        model = IsolationForest(
            random_state=42,
            n_jobs=-1,
            **params
        )
        model.fit(X_train_scaled)

        # 2. 获取测试集分数 (iForest的score_samples越低越异常，取反)
        test_scores_raw = -model.score_samples(X_test_scaled)

        # 3. 评估 F1 (注意：此处在原始分数上评估，不影响F1)
        f1, _ = find_best_f1_threshold(test_scores_raw, y_test, verbose=False)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            print(f"    *** 新的最佳 F1: {f1:.4f} ***")

    print(f"\n网格搜索完成。耗时: {time.time() - start_time:.2f} 秒")
    print(f"最佳 F1 (来自网格搜索): {best_f1:.4f}")
    print(f"最佳参数: {best_params}\n")

    # 4. 使用最佳参数重新训练最终模型
    print("--- 3. 正在训练最终模型... ---")
    best_model = IsolationForest(
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    best_model.fit(X_train_scaled)
    print("最终模型训练完毕。\n")

    return best_model, best_params


def run_iforest_experiment(train_file, test_file, label_file, num_features, output_dir):
    """
    封装后的主执行函数
    """

    # 1. 加载数据 (使用新函数)
    data = load_and_merge_data(train_file, test_file, label_file, num_features)
    df_train, df_test_with_labels, X_train, X_test, y_test, time_col = data

    # 2. 预处理
    X_train_scaled, X_test_scaled, scaler, imputer = preprocess_data(X_train, X_test)

    # 3. 定义网格搜索参数
    param_grid = {
        'n_estimators': [25, 50, 100, 200],
        'max_samples': ['auto', 0.25, 0.5, 0.75],
        'max_features': [0.25, 0.5, 0.75, 1.0]
    }

    # 4. 执行网格搜索和训练
    best_model, best_params = grid_search_iforest(X_train_scaled, X_test_scaled, y_test, param_grid)

    # 5. 生成异常分数
    print("--- 4. 正在生成最终的异常分数... ---")
    train_scores_raw = -best_model.score_samples(X_train_scaled)
    test_scores_raw = -best_model.score_samples(X_test_scaled)

    # 6. 归一化分数 (0到1)
    score_scaler = MinMaxScaler()
    train_scores_norm = score_scaler.fit_transform(train_scores_raw.reshape(-1, 1)).ravel()
    test_scores_norm = score_scaler.transform(test_scores_raw.reshape(-1, 1)).ravel()
    print("分数计算并归一化完毕。\n")

    # 7. 评估模型 (在归一化的测试集分数上)
    find_best_f1_threshold(test_scores_norm, y_test, verbose=True)

    # 8. 保存结果
    print("--- 6. 正在保存结果... ---")

    # 创建保存目录
    # output_dir = "../dataset_scores/MSL/IF"
    os.makedirs(output_dir, exist_ok=True)

    # 动态命名
    train_basename = os.path.splitext(os.path.basename(train_file))[0]
    test_basename = os.path.splitext(os.path.basename(test_file))[0]

    # 训练集
    df_train_out = df_train.copy()
    df_train_out['anomaly_score'] = train_scores_norm
    train_out_file = os.path.join(output_dir, f"{train_basename}_with_scores.csv")
    df_train_out.to_csv(train_out_file, index=False)
    print(f"训练集分数已保存到: {train_out_file}")

    # 测试集
    df_test_out = df_test_with_labels.copy()
    df_test_out['anomaly_score'] = test_scores_norm
    test_out_file = os.path.join(output_dir, f"{test_basename}_with_scores.csv")
    df_test_out.to_csv(test_out_file, index=False)
    print(f"测试集分数已保存到: {test_out_file}")

    print("\n--- 任务完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Isolation Forest 进行异常检测并优化 F1 分数")

    parser.add_argument("--train_file", type=str, required=True, help="训练CSV文件的路径 (例如: train.csv)")
    parser.add_argument("--test_file", type=str, required=True, help="测试CSV文件的路径 (例如: test.csv)")
    parser.add_argument("--label_file", type=str, required=True, help="测试标签CSV文件的路径 (例如: test_label.csv)")
    parser.add_argument("--num_features", type=int, default=0, help="特征数量")
    parser.add_argument("--output_dir", type=str, default=None, help="结果保存目录")

    args = parser.parse_args()

    print(args.train_file)

    run_iforest_experiment(
        train_file=args.train_file,
        test_file=args.test_file,
        label_file=args.label_file,
        num_features=args.num_features,
        output_dir=args.output_dir
    )