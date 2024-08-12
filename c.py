import pandas as pd
from collections import Counter
# 读取训练数据和提交样本数据
df_original = pd.read_csv("./train_data.csv")
n_original = df_original.shape[0]  # 获取原始数据的行数
df_submit = pd.read_csv("./sample_submission2.csv")
 
# 合并数据并重置索引
df = pd.concat([df_original, df_submit], axis=0).reset_index(drop=True)
class GenomicTokenizer:
    def __init__(self, ngram=5, stride=2):
        # 初始化分词器，设置n-gram长度和步幅
        self.ngram = ngram
        self.stride = stride
        
    def tokenize(self, t):
 
        # 字符串变list
        if isinstance(t, str):
            t = list(t)
 
        if self.ngram == 1:
            # 如果n-gram长度为1，直接将序列转换为字符列表
            toks = t
        else:
            # 否则，按照步幅对序列进行n-gram分词
            toks = [t[i:i+self.ngram] for i in range(0, len(t), self.stride) if len(t[i:i+self.ngram]) == self.ngram]
        
            # 如果最后一个分词长度小于n-gram，移除最后一个分词
            if len(toks[-1]) < self.ngram:
                toks = toks[:-1]
 
            # sub list to str
            toks = [''.join(x) for x in toks]
 
        # 返回分词结果
        return toks
 
class GenomicVocab:
    def __init__(self, itos):
        # 初始化词汇表，itos是一个词汇表列表
        self.itos = itos
        # 创建从词汇到索引的映射
        self.stoi = {v: k for k, v in enumerate(self.itos)}
        
    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        # 创建词汇表类方法
        # 统计每个token出现的频率
        freq = Counter(tokens)
        # 选择出现频率大于等于min_freq的token，并且最多保留max_vocab个token
        # itos = ['<pad>'] + [o for o, c in freq.most_common(max_vocab - 1) if c >= min_freq]
        itos = [o for o, c in freq.most_common(max_vocab - 1) if c >= min_freq]
        # 返回包含词汇表的类实例
        return cls(itos)
    
def siRNA_feat_builder_substr(se, name, patterns):
    
    # 创建一个空字典来存储特征
    features = {}
 
    for pattern in patterns:
        try:
            # escaped_pattern = re.escape(pattern)  # 转义模式中的特殊字符
            escaped_pattern = pattern
            features[f"feat_{name}_seq_pattern_{escaped_pattern}"] = se.str.count(escaped_pattern)
        except re.error as e:
            print(f"Error in pattern {pattern}: {e}")
 
    # 将字典转换为DataFrame
    feature_df = pd.DataFrame(features)
 
    return feature_df

# 构建siRNA特征的函数
def siRNA_feat_builder(s: pd.Series, anti: bool = False):
    name = "anti" if anti else "sense"  # 根据参数设置名称
    df = s.to_frame()  # 将系列转换为DataFrame
    df[f"feat_siRNA_{name}_seq_len"] = s.str.len()  # 序列长度特征
 
    # 前后位置的碱基特征
    for pos in [0, -1]:
        for c in list("AUGC"):
            df[f"feat_siRNA_{name}_seq_{c}_{'front' if pos == 0 else 'back'}"] = (
                s.str[pos] == c
            )
    
    # 不同的特定序列模式特征
    df[f"feat_siRNA_{name}_seq_pattern_1"] = s.str.startswith("AA") & s.str.endswith("UU")
    df[f"feat_siRNA_{name}_seq_pattern_2"] = s.str.startswith("GA") & s.str.endswith("UU")
    df[f"feat_siRNA_{name}_seq_pattern_3"] = s.str.startswith("CA") & s.str.endswith("UU")
    df[f"feat_siRNA_{name}_seq_pattern_4"] = s.str.startswith("UA") & s.str.endswith("UU")
    df[f"feat_siRNA_{name}_seq_pattern_5"] = s.str.startswith("UU") & s.str.endswith("AA")
    df[f"feat_siRNA_{name}_seq_pattern_6"] = s.str.startswith("UU") & s.str.endswith("GA")
    df[f"feat_siRNA_{name}_seq_pattern_7"] = s.str.startswith("UU") & s.str.endswith("CA")
    df[f"feat_siRNA_{name}_seq_pattern_8"] = s.str.startswith("UU") & s.str.endswith("UA")
    df[f"feat_siRNA_{name}_seq_pattern_9"] = s.str[1] == "A"
    df[f"feat_siRNA_{name}_seq_pattern_10"] = s.str[-2] == "A"
 
    # GC含量特征
    df[f"feat_siRNA_{name}_seq_pattern_GC_frac"] = (
        s.str.count("G") + s.str.count("C")
    ) / s.str.len()
    
    return df.iloc[:, 1:]  # 返回特征列
 
def siRNA_feat_builder3(s: pd.Series, anti: bool = False):
    name = "anti" if anti else "sense"
    df = s.to_frame()
    #长度分组
    # df[f"feat_siRNA_{name}_len21"] = (s.str .len() == 21)
    df[f"feat_siRNA {name}_len21_25"] = (s.str.len() >= 21) & (s.str.len() <= 25)
    # GC含量
    GC_frac = (s.str . count("G") + s. str . count("C" ))/s.str.len()
    df[f"feat_siRNA {name}_GC_in"] = (GC_frac >= 0.36) & (GC_frac <= 0.52)
    #局部GC含量
    GC_frac1 = (s.str[1:7].str . count("G") + s.str[1:7].str . count("C" ))/s.str[1:7].str.len()
    GC_frac2 = (s. str[7:18]. str . count("G") + s.str[7:18]. str. count("C" ))/s. str[7:18].str .len()
    df[f"feat_siRNA{name}_GC_in1"] = (GC_frac1 == 0.19)
    df[f"feat_siRNA{name}_GC in2"] = (GC_frac2 == 8.52)
    return df.iloc[:, 1:]

# 对publication_id进行独热编码
df_publication_id = pd.get_dummies(df.publication_id)
df_publication_id.columns = [f"feat_publication_id_{c}" for c in df_publication_id.columns]
 
# 对gene_target_symbol_name进行独热编码
df_gene_target_symbol_name = pd.get_dummies(df.gene_target_symbol_name)
df_gene_target_symbol_name.columns = [f"feat_gene_target_symbol_name_{c}" for c in df_gene_target_symbol_name.columns]
 
# 对gene_target_ncbi_id进行独热编码
df_gene_target_ncbi_id = pd.get_dummies(df.gene_target_ncbi_id)
df_gene_target_ncbi_id.columns = [f"feat_gene_target_ncbi_id_{c}" for c in df_gene_target_ncbi_id.columns]
 
# 对gene_target_species进行独热编码
df_gene_target_species = pd.get_dummies(df.gene_target_species)
df_gene_target_species.columns = [f"feat_gene_target_species_{c}" for c in df_gene_target_species.columns]
 
# 标准化siRNA_duplex_id
siRNA_duplex_id_values = df.siRNA_duplex_id.str[3:-2].str.strip(".").astype("int")
siRNA_duplex_id_values = (siRNA_duplex_id_values - siRNA_duplex_id_values.min()) / (
    siRNA_duplex_id_values.max() - siRNA_duplex_id_values.min()
)
df_siRNA_duplex_id = pd.DataFrame(siRNA_duplex_id_values)
 
# 对cell_line_donor进行独热编码
df_cell_line_donor = pd.get_dummies(df.cell_line_donor)
df_cell_line_donor.columns = [f"feat_cell_line_donor_{c}" for c in df_cell_line_donor.columns]
 
# 增加特定的细胞系特征
df_cell_line_donor["feat_cell_line_donor_hepatocytes"] = (
    (df.cell_line_donor.str.contains("Hepatocytes")).fillna(False).astype("int")
)
df_cell_line_donor["feat_cell_line_donor_cells"] = (
    df.cell_line_donor.str.contains("Cells").fillna(False).astype("int")
)
# 将siRNA浓度转换为DataFrame
df_siRNA_concentration = df.siRNA_concentration.to_frame()
 
# 对Transfection_method进行独热编码
df_Transfection_method = pd.get_dummies(df.Transfection_method)
df_Transfection_method.columns = [f"feat_Transfection_method_{c}" for c in df_Transfection_method.columns]
 
# 对Duration_after_transfection_h进行独热编码
df_Duration_after_transfection_h = pd.get_dummies(df.Duration_after_transfection_h)
df_Duration_after_transfection_h.columns = [
    f"feat_Duration_after_transfection_h_{c}" for c in df_Duration_after_transfection_h.columns
]
 
# 合并所有特征
feats = pd.concat(
    [
        df_publication_id,
        df_gene_target_symbol_name,
        df_gene_target_ncbi_id,
        df_gene_target_species,
        df_siRNA_duplex_id,
        df_cell_line_donor,
        df_siRNA_concentration,
        df_Transfection_method,
        df_Duration_after_transfection_h,
        siRNA_feat_builder(df.siRNA_sense_seq, False),
        siRNA_feat_builder(df.siRNA_antisense_seq, True),
        siRNA_feat_builder3(df.siRNA_sense_seq, False),
        siRNA_feat_builder3(df.siRNA_antisense_seq, True),
        df.iloc[:, -1].to_frame(),
    ],
    axis=1,
)
 
import lightgbm as lgb
from sklearn.model_selection import train_test_split
 
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    feats.iloc[:n_original, :-1],
    feats.iloc[:n_original, -1],
    test_size=0.25,
    random_state=42,
)
 
# 构建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
 
# 打印验证结果的回调函数
def print_validation_result(env):
    result = env.evaluation_result_list[-1]
    print(f"[{env.iteration}] {result[1]}'s {result[0]}: {result[2]}")
 
# 设置LightGBM参数
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "root_mean_squared_error",
    "max_depth": 7,
    "learning_rate": 0.02,
    "verbose": 0,
}
 
# 训练模型
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=20000,
    valid_sets=[test_data],
    callbacks=[print_validation_result],
)
 
# 预测结果
y_pred = gbm.predict(feats.iloc[n_original:, :-1])
 
# 保存预测结果到CSV文件
df_submit["mRNA_remaining_pct"] = y_pred
df_submit.to_csv("submission.csv", index=False)