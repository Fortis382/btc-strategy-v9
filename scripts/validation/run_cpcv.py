# scripts/validation/run_cpcv.py (신규)

from src.validation.purged_cv import combinatorial_purged_cv

df = load_ohlcv(...)
data = df.to_numpy()

folds = combinatorial_purged_cv(data, n_splits=5, n_test_groups=2)

train_wrs = []
test_wrs = []

for i, (train_idx, test_idx) in enumerate(folds):
    df_train = df[train_idx]
    df_test = df[test_idx]
    
    # 각각 백테스트
    _, result_train = simple_backtest(df_train, cfg)
    _, result_test = simple_backtest(df_test, cfg)
    
    train_wrs.append(result_train["winrate"])
    test_wrs.append(result_test["winrate"])

train_mean = np.mean(train_wrs)
test_mean = np.mean(test_wrs)
gap = (test_mean - train_mean) / train_mean

print(f"Train WR: {train_mean:.2%}")
print(f"Test WR: {test_mean:.2%}")
print(f"Gap: {gap:.1%}")
print(f"Overfitting risk: {max(0, -gap)*100:.0f}%")