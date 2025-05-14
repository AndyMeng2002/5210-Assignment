import cudf
import pandas as pd
import cupy as cp
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt



# 本函数用于将指定文件夹中所有的 pqt 文件合成一个大的 DataFrame 并加入日期信息作为一列
def concat_factor(folder_path: str) -> pd.DataFrame:
    # 找出目标文件夹中所有 .pqt 文件
    all_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pqt')])
    all_dfs = []
    for fp in all_files:
        df = pd.read_parquet(fp)
        base_name = os.path.basename(fp)
        # 根据文件名称提取日期信息
        date_str = base_name.split('_')[1].split('.')[0]
        df['Date'] = int(date_str)
        if df.index.name == 'code':
            df = df.reset_index() 
        cols = ['Date', 'code'] + [col for col in df.columns if col not in ['Date', 'code']]
        df = df[cols]
        all_dfs.append(df)
    factor_all = pd.concat(all_dfs, axis=0, ignore_index=True).sort_values(by=['Date','code'])
    return factor_all



# 本函数用于计算时间序列 IC
def cal_time_series_ic(df_merge: pd.DataFrame, ic_type: str) -> pd.DataFrame:
    '''
    df_merge 输入的是已经将 return 和 factor 粘贴到一起的dataframe
    要求 df_merge 有两列分别是'code'和'date',并且所有收益率的列要以 return_xxx 来命名
    ic_type只有两种输入值: ic 或者 rankic
    '''
    factor_cols = [col for col in df_merge.columns if col not in ['Date', 'code'] and not col.startswith('return')]
    return_cols = [col for col in df_merge.columns if col.startswith('return')]
    # 创建：外层 dict 的 key 是股票代码，内层是每个 fac|ret 的 IC 值
    result_dict = {}

    if ic_type == 'rankic':
        method = 'spearman'
    elif ic_type == 'ic':
        method = 'pearson' 

    for stock, group in df_merge.groupby('code'):
        for fac in factor_cols:
            for ret in return_cols:
                col_name = f"{fac}|{ret}"  # 把因子和收益列组合成唯一的列名
                x = group[fac]
                y = group[ret]
                ic = x.corr(y, method=method)
                if stock not in result_dict: # 初始化该股票行
                    result_dict[stock] = {}
                result_dict[stock][col_name] = ic
        
    # 构造成 DataFrame，行是股票代码，列是 fac|ret 组合
    # time_series_ic = pd.DataFrame.from_dict(result_dict, orient='index')
    time_series_ic = pd.DataFrame.from_dict(result_dict, orient='index')
    time_series_ic.index.name = 'code'
    time_series_ic = time_series_ic.sort_values(by=['code'])
    return time_series_ic



# 本函数用于计算横截面 IC
def cal_cross_sectional_ic(df_merge: pd.DataFrame, ic_type: str) -> pd.DataFrame:
    '''
    df_merge 输入的是已经将 return 和 factor 粘贴到一起的dataframe
    要求 df_merge 有两列分别是'code'和'date',并且所有收益率的列要以 return_xxx 来命名
    ic_type只有两种输入值: ic 或者 rankic
    '''
    factor_cols = [col for col in df_merge.columns if col not in ['Date', 'code'] and not col.startswith('return')]
    return_cols = [col for col in df_merge.columns if col.startswith('return')]
    # 创建：外层 dict 的 key 是股票代码，内层是每个 fac|ret 的 IC 值
    result_dict = {}

    if ic_type == 'rankic':
        method = 'spearman'
    elif ic_type == 'ic':
        method = 'pearson' 

    for date, group in df_merge.groupby('Date'):
        for fac in factor_cols:
            for ret in return_cols:
                col_name = f"{fac}|{ret}"  # 把因子和收益列组合成唯一的列名
                x = group[fac]
                y = group[ret]
                ic = x.corr(y, method=method)
                if date not in result_dict: # 初始化该股票行
                    result_dict[date] = {}
                result_dict[date][col_name] = ic

    # 构造成 DataFrame，行是股票代码，列是 fac|ret 组合
    cross_sectional_ic = pd.DataFrame.from_dict(result_dict, orient='index')
    cross_sectional_ic.index.name = 'Date'
    cross_sectional_ic = cross_sectional_ic.sort_values(by=['Date'])
    return cross_sectional_ic



def cal_icir(df_ic: pd.DataFrame) -> pd.DataFrame:
    '''
    输入:
        df_ic: pd.DataFrame
        每一列是一个因子的 IC 时间序列（如每天一个值），列名形如 'factor1'、'factor2' 等
    输出:
        pd.DataFrame
        行是因子名，列包括 IC Mean、IC Std、ICIR
    '''
    ic_mean = df_ic.mean(axis=0)
    ic_std = df_ic.std(axis=0)
    icir = ic_mean / ic_std
    result_df = pd.DataFrame({
        'IC Mean': ic_mean.to_pandas(),
        'IC Std': ic_std.to_pandas(),
        'ICIR': icir.to_pandas()
    })
    result_df.index.name = 'factor'
    return result_df



def cal_pair_diff(factor: pd.DataFrame) -> pd.DataFrame:
    '''
    输入的是原始因子 dataframe
    输出的是原始因子 pairwise-difference 的 dataframe
    '''
    # 1. 找出 dataframe 的因子列（排除 'Date' 和 'code'）
    factor_cols = [col for col in factor.columns if col not in ['Date', 'code']]
    # 2. 枚举所有两两组合：C(12, 2)
    pairs = list(itertools.combinations(factor_cols, 2))
    # 3. 创建一个新 DataFrame 保存差值
    diff_df = factor[['Date', 'code']].copy()
    # 4. 对每一对因子计算差值
    for col1, col2 in pairs:
        diff_col_name = f'{col1} & {col2}'
        diff_df[diff_col_name] = factor[col1] - factor[col2]
    return diff_df



# 本函数用于对比两个不同时间段的 IC 值在图像上的差异
def plot_compare_ic_mean(ic_mean1, ic_mean2, label1='Period 1', label2='Period 2', title='IC Mean Comparison'):
    """
    对比两个不同时间段的因子 IC 均值
    参数:
        ic_mean1, ic_mean2: pd.Series,每个因子的 IC 均值, index 为因子名
        label1, label2: 图例标签
        title: 图标题
    """
    sorted_index = ic_mean1.sort_values(ascending=True).index
    # 对两个 Series 重新排序
    df1_sorted = ic_mean1[sorted_index]
    df2_sorted = ic_mean2.reindex(sorted_index)
    # 画图
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(df1_sorted)), df1_sorted.values, width=0.4, label=label1, color='skyblue')
    plt.bar([x + 0.4 for x in range(len(df2_sorted))], df2_sorted.values, width=0.4, label=label2, color='orange')
    
    plt.title(title)
    plt.legend()
    plt.xticks([])
    plt.tight_layout()
    plt.show()


def plot_all(ic_mean):
    # 所有持仓时间点要保持这个顺序
    horizon_order = ['return_1min', 'return_5min', 'return_10min', 'return_30min',
                    'return_1hr', 'return_2hr', 'return_inday', 'return_overnight']

    # 提取所有因子名
    factors = sorted(set(j.split('|')[0] for j in ic_mean.index))

    # 每行一个因子图
    n_cols = 3  # 每行放几个图
    n_rows = (len(factors) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for idx, factor in enumerate(factors):
        # 选出该因子相关的数据
        sub_idx = [j for j in ic_mean.index if j.split('|')[0] == factor]
        sub_df = ic_mean.loc[sub_idx]
        
        # 重新排序
        sub_df = sub_df.reindex([f"{factor}|{h}" for h in horizon_order if f"{factor}|{h}" in sub_df.index])

        # 横轴：持仓周期，纵轴：IC值
        axes[idx].plot([i.split('|')[1].replace('return_', '') for i in sub_df.index], sub_df.values, marker='o')
        axes[idx].set_title(factor, fontsize=10)
        xticks = range(len(sub_df))
        xticklabels = [i.split('|')[1].replace('return_', '') for i in sub_df.index]
        axes[idx].set_xticks(xticks)
        axes[idx].set_xticklabels(xticklabels, rotation=45)
        axes[idx].grid(True)

    # 清除多余子图
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle('Mean RankIC per Factor vs Holding Period', fontsize=16, y=1.02)
    plt.show()
