# -*- coding: utf-8 -*-
"""
房价预测高分方案（修正版）
包含：特征工程、XGBoost + LightGBM + RandomForest 融合、交叉验证
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

# ================== 1. 读取数据 ==================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 保存ID
train_id = train['Id']
test_id = test['Id']

# 删除ID列，模型不需要
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# ================== 1.5 提前计算 Neighborhood 均值编码（需要 SalePrice） ==================
neighborhood_mean = train.groupby('Neighborhood')['SalePrice'].mean()

# 保存目标变量 y
y = train['SalePrice']
y_log = np.log1p(y)

# 删除 SalePrice 列（train 中不再需要）
train.drop('SalePrice', axis=1, inplace=True)

# 合并以便统一处理特征
all_data = pd.concat([train, test], axis=0, ignore_index=True)
print(f"合并后数据形状: {all_data.shape}")

# ================== 2. 缺失值处理 ==================
# 2.1 对于“无”表示为缺失的特征（如 Alley, BsmtQual, FireplaceQu, GarageType, PoolQC, Fence等）
none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'PoolQC', 'Fence', 'MiscFeature']
for col in none_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

# 2.2 数值型特征的缺失值（如 LotFrontage, MasVnrArea, GarageYrBlt）
num_cols = all_data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# 2.3 剩余类别特征（如有缺失）用众数填充
cat_cols = all_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# ================== 3. 特征工程 ==================
# 3.1 对数变换：对严重偏斜的数值特征取 log(1+x)
skew_cols = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
             'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'PoolArea', 'MiscVal']
for col in skew_cols:
    if col in all_data.columns:
        all_data[col] = np.log1p(all_data[col])

# 3.2 创建新的组合特征
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = (all_data['FullBath'] + 0.5*all_data['HalfBath'] +
                         all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath'])
all_data['Age'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['IsRemod'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)

# 3.3 Neighborhood 目标编码
all_data['Neighborhood_encoded'] = all_data['Neighborhood'].map(neighborhood_mean)
all_data['Neighborhood_encoded'] = all_data['Neighborhood_encoded'].fillna(all_data['Neighborhood_encoded'].median())

# 3.4 其他类别特征进行 Label Encoding
cat_cols_to_encode = [col for col in cat_cols if col != 'Neighborhood']
for col in cat_cols_to_encode:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))

# 3.5 删除原始 Neighborhood 列（已编码）
all_data.drop('Neighborhood', axis=1, inplace=True)

# 确认所有列都是数值类型
print(f"处理后特征数: {all_data.shape[1]}")

# ================== 4. 准备训练和测试数据 ==================
X = all_data.iloc[:len(train), :]
X_test = all_data.iloc[len(train):, :]

print(f"训练集特征形状: {X.shape}, 测试集特征形状: {X_test.shape}")

# ================== 5. 模型定义（交叉验证 + 融合） ==================
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

models = {
    'xgb': xgb.XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8, random_state=42),
    'lgb': lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbose=-1),
    'rf': RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
}

test_preds = {name: np.zeros(X_test.shape[0]) for name in models}
val_scores = {name: [] for name in models}

for name, model in models.items():
    print(f"\n训练模型: {name}")
    val_preds = np.zeros(X.shape[0])
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        
        pred_val = model_clone.predict(X_val)
        val_preds[val_idx] = pred_val
        
        test_pred = model_clone.predict(X_test)
        test_preds[name] += test_pred / n_folds
        
        score = rmsle(y_val, pred_val)
        val_scores[name].append(score)
        print(f"  Fold {fold+1} RMSLE: {score:.5f}")
    print(f"  Average RMSLE: {np.mean(val_scores[name]):.5f}")

# ================== 6. 模型融合 ==================
weights = {'xgb': 0.4, 'lgb': 0.4, 'rf': 0.2}
final_test_pred = np.zeros(X_test.shape[0])
for name in models:
    final_test_pred += weights[name] * test_preds[name]

# 还原预测值
final_pred = np.expm1(final_test_pred)

# ================== 7. 生成提交文件 ==================
submission = pd.DataFrame({'Id': test_id, 'SalePrice': final_pred})
submission.to_csv('submission.csv', index=False)
print("\n✅ 提交文件已生成: submission.csv")
print(f"预测值范围: {final_pred.min():.0f} ~ {final_pred.max():.0f}")
