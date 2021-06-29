# BaseBall-result-prediction-Probspace  
次の一投の行方を予測！ プロ野球データ分析チャレンジ_Solutions

## コンペのページ  
* https://prob.space/competitions/npb

## Result  
Public：5位
Private：7位

## Solution

以下のフォルダ構成になっています。
今回は、1exp-1notebook形式で行っています。

.
├── exp  
│ └── exp104.ipynb  
│
└── train_data.csv  
├── test_data.csv  
├── game_info.csv  
└── submission.csv    


## Solution解説  

### 方針  
![Screenshot from 2021-06-21 00-09-19](https://user-images.githubusercontent.com/46860245/122679212-f4a45c80-d224-11eb-955d-43344930a783.png)

流れは上記のような形です。
以下に、cv, lbともに大きく改善したところにpickupしたいと思います。

### train_dataの作成  
コンペ開始時から、train_dataとtest_dataの構造が違うことが気になっていました。  
（以下のトピックにも上がっています）  
https://prob.space/competitions/npb/discussions/columbia2131-Post0bae12df9ec3007c4cce  

ここは必要だなと感じていましたが、うまく機能しない状態でした。  
トピックに上がっていたデータの作り方を参考に実装したら上手くいったので、こちらを参考にしました。  
```python
def random_sampling(input_df, n_sample=10):
    dfs = []
    tr_df = input_df[input_df['y'].notnull()].copy()
    ts_df = input_df[input_df['y'].isnull()].copy()
    for i in tqdm(range(n_sample)):
        df = tr_df.groupby(['gameID', 'outCount']).apply(lambda x: x.sample(n=1, random_state=i+30)).reset_index(drop=True)
        df['subGameID'] = df['gameID'] * n_sample + i
        dfs.append(df)
    ts_df['subGameID'] = ts_df['gameID'] * n_sample
    return pd.concat(dfs,axis=0), ts_df
```

トピックに上がっているものをそのまま使用せず、random_stateを調整しました。

### Fold設定  
多クラス分類だったので、StratifiedKFoldで実施してましたが、GameIDでGroupKFoldで実施したところcvとlbの乖離が小さくなりました（train_dataとtest_dataがGameIDで分割さえていたのはわかっていたのに。。。）

```python
# GroupKFold with random shuffle with a sklearn-like structure
class RandomGroupKFold:
    def __init__(self, n_splits=4, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        unique_ids = groups.unique()
        for tr_group_idx, va_group_idx in kf.split(unique_ids):
            # split group
            tr_group, va_group = unique_ids[tr_group_idx], unique_ids[va_group_idx]
            train_idx = np.where(groups.isin(tr_group))[0]
            val_idx = np.where(groups.isin(va_group))[0]
            yield train_idx, val_idx
```

###特徴量作成  
今回の特徴量は以下です。  
```python
processes = [
        get_oe_features,
        get_ce_features,
        get_ohe_features,
        create_numeric_feature,
        get_inning,
        get_pvsb,
        get_agg_inning,
        get_agg_runner_status,
        get_full_count,
        get_match,
        get_league,
        get_count_batter,
        get_HR_batter,
        get_KK_pitcher,
        ballPositionLabel_ratio,
        ballPositionLabel_ratio_b,
        ballPositionLabel_ratio_p_speed,
        get_batting_average,
        get_next_data,
        get_prev_data,
        get_next_diff,
        get_prev_diff,
        get_skip,
        get_tfidf,
        get_pivot_NMF9_features,
        get_pivot_NMF27_features,
        get_pivot_NMF54_features,
        get_next_data_BSO,
        get_prev_data_BSO,
        get_next_diff_BSO,
        get_prev_diff_BSO,
        get_tfidf_p,
        get_pivot_NMF9_features_BSO,
        get_pivot_NMF27_features_BSO,
        get_pivot_NMF54_features_BSO,
        pca_ball_position_counts
    ]

```
* BallYを予測して特徴量として実装
* pitchtypeを予測して特徴量として実装

###CV, LB効いた特徴量や、やったこと
* BallYの予測
BallYを予測して特徴量に入れる。またはpivottableにしてpitcherごとやbatterごとに集計したものも効果がありました。  

* counts, runnerでの特徴量生成
```python
train_df["total_counts2"] = train_df[['S', 'B', 'O', 'totalPitchingCount']].apply(lambda x: '{}-{}-{}-{}'.format(x[0], x[1], x[2],x[3]),axis=1)
test_df["total_counts2"] = test_df[['S', 'B', 'O','totalPitchingCount']].apply(lambda x: '{}-{}-{}-{}'.format(x[0], x[1], x[2],x[3]),axis=1)

```
カウントの状態などを特徴量にしたのに効果がありました。  

* "class_weight":"balanced"
多クラス分類で、不均衡なデータだったので、lgbmのparam設定を変更することで改善しました。

* 前後特徴量  
トピックから拝借した特徴量も効きました。  
https://prob.space/topics/DT-SN-Post2126e8f25865e24a1cc4  

```python
def get_diff_feature(input_df, value_col, periods, in_inning=True, aggfunc=np.median):
    pivot_df = pd.pivot_table(input_df, index='subGameID', columns='outCount', values=value_col, aggfunc=aggfunc)
    if in_inning:
        dfs = []
        for inning in range(9):
            df0 = pivot_df.loc[:, [out+inning*6 for out in range(0,3)]].diff(periods, axis=1)
            df1 = pivot_df.loc[:, [out+inning*6 for out in range(3,6)]].diff(periods, axis=1)
            dfs += [df0, df1]
        pivot_df = pd.concat(dfs, axis=1).stack()
    else:
        df0 = pivot_df.loc[:, [out+inning*6 for inning in range(9) for out in range(0,3)]].diff(periods, axis=1)
        df1 = pivot_df.loc[:, [out+inning*6 for inning in range(9) for out in range(3,6)]].diff(periods, axis=1)
        pivot_df = pd.concat([df0, df1], axis=1).stack()
    return pivot_df

def get_shift_feature(input_df, value_col, periods, in_inning=True, aggfunc=np.median):
    pivot_df = pd.pivot_table(input_df, index='subGameID', columns='outCount', values=value_col, aggfunc=aggfunc)
    if in_inning:
        dfs = []
        for inning in range(9):
            df0 = pivot_df.loc[:, [out+inning*6 for out in range(0,3)]].shift(periods, axis=1)
            df1 = pivot_df.loc[:, [out+inning*6 for out in range(3,6)]].shift(periods, axis=1)
            dfs += [df0, df1]
        pivot_df = pd.concat(dfs, axis=1).stack()
    else:
        df0 = pivot_df.loc[:, [out+inning*6 for inning in range(9) for out in range(0,3)]].shift(periods, axis=1)
        df1 = pivot_df.loc[:, [out+inning*6 for inning in range(9) for out in range(3,6)]].shift(periods, axis=1)
        pivot_df = pd.concat([df0, df1], axis=1).stack()
    return pivot_df
```

### アンサンブルに関して  
今回、学習データをテストテータに近づけるために、学習データからサンプリングを行うことをしています。そのため、サンプリングの偏りによって、overfitしてしまわないか？などが気になったので、複数のサンプリングの設定で学習データを生成して、予測結果を算出。最瀕値を予測結果とすることで提出するなどを行いましたが、結果Publicにoverfitした形になっていました。