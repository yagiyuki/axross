# 「より良い」機械学習モデルの構築方法を学べるレシピ

## レシピの概要

このレシピは、「より良い」機械学習モデルの構築方法を学べるレシピです。

「より良い」機械学習モデルとは、以下の2点の条件を満たすものをさします。

* 性能がよいこと
* 処理効率がよいこと

このレシピでは、文章分類のタスクの改善に取組、
「より良い」機械学習モデルにするための改善方法を実戦的に学んでいきます。

### 前提知識

* python (基本構文であるif, for などが理解できる。)
* sklearn, pandasなどのライブラリを扱える(写経できるだけでもよい)
* 機械学習の教師あり学習の概念がわかる

### 達成できること

* 「より良い」に記載した2点の基準でモデル改善アプローチを実戦的に学べる
* 文章分類モデルの学習・評価の基礎がわかる

## 材料・道具

### 実行環境
Google Colaboratory

### 道具

* python(3.5以上)
* sudachipy
* sudachidict-core
* sortedcontainers
* pandas
* matplotlib
* seaborn

### 材料

livedoor ニュースコーパス https://www.rondhuit.com/download.html#ldcc

## 良い機械学習モデルをつくるには何をするか

レシピ概要で記載したとおり、より良い機械学習モデルとは以下の2点をさします。

* 性能がよいこと
* 処理効率がよいこと


### 性能がよいこと

モデルの正解率、再現率、適合率といった指標で高い数値が出ていることです。

### 処理効率がよいこと

処理速度が早く、なるべく低リソースで動作することです。


上記の視点で「より良い」機械学習モデルをつくるには、
「データ前処理」、「ハイパーパラメータのチューニング」を実戦的に学んでいきます。

ただ単に最終系のコードを提示するだけでなく、改善の過程も残しています。
実務に役立つノウハウを提供できれば、幸いです。

## 調理

### 道具の準備

必要な道具と材料をそろえます。

まず道具ですが、ベースの動作環境は、Google Colaboratoryにだいたいのものは揃っているので、
追加で用意する必要があるのは、形態素解析ツールであるsudachiのみになります。

* sudachipy
* sudachidict-core
* sortedcontainers

以下のコマンドでインストールしてください。

```
%%bash
pip install sudachipy==0.5.1 sudachidict-core==20201223.post1 sortedcontainers==2.1.0
```

インストール後は、Colaboratoryのメニューから"ランタイム>ランタイム" を再起動してください。
※ 実施しないとインストールした道具がロードされないので注意

### 材料を揃える

livedoor ニュースコーパスのデータセットをダウンロード&&展開します。 

```
%%bash
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar xzf ldcc-20140209.tar.gz
```

展開すると直下にディレクトリ`text`というディレクトリがあり、その下に9つのディレクトリがあります。 

```
topic-news
sports-watch
kaden-channel
smax
livedoor-homme
it-life-hack
dokujo-tsushin
peachy
movie-enter
```

個々のディレクトリにはニュース記事が入っています。 
ディレクトリがニュースのカテゴリにあたり、その直下にニュースの記事があります。 

```
# 独女通信の例
./text/
├── CHANGES.txt
├── dokujo-tsushin
│   ├── dokujo-tsushin-4778030.txt
│   ├── dokujo-tsushin-4778031.txt
│   ├── dokujo-tsushin-4782522.txt
│   ├── dokujo-tsushin-4788357.txt
│   ├── dokujo-tsushin-4788362.txt

```

### データセットの準備

livedoorのニュース記事を対応するカテゴリに分類していくタスクに取り組みます。 
そのためのデータセットを準備します。 

```python
from pprint import pprint
import os

#
# テキスト直下のディレクトリ一覧を取得(これがカテゴリになる。)
#
dirlist = os.listdir('text')
category_list = {}
i=0
for dirname in dirlist:
    if dirname[-3:] != 'txt':
        category_list[str(i)] = dirname
        i+=1

#
# データセットを作成して、ファイルに出力する。
#　　ファイルはtsv形式で、ファイル名、ラベルid、カテゴリ名、テキストを出力する。
#
with open('dataset.tsv', 'w') as f_out:
    for label, category in category_list.items():
        path = './text/{}/'.format(category)
        filelist = os.listdir(path)
        filelist.remove('LICENSE.txt')
        for filename in filelist:
            with open(path + filename, 'r') as f_in:
                # テキストはタイトルのみ取得　(本文は学習対象にしない)
                text = f_in.readlines()[2]
                # カラム生成
                out_row = [filename, label, category, text]
                f_out.write("\t".join(out_row))
```

`dataset.tsv`というファイルが生成されて、以下のようなデータがあれば、成功です。 

```
topic-news-6612237.txt	0	topic-news	神戸「サンテレビ」、プロ野球中継で放送事故
topic-news-6298663.txt	0	topic-news	フジで午後のワイドショーが復活、韓流推し反対デモの影響は「関係ない」に物議
topic-news-6625187.txt	0	topic-news	「全てのトイレを和式に」 野村ホールディングス株主の珍提案が海外で話題に
topic-news-6118456.txt	0	topic-news	女性教授が男子生徒に「なめるな」「テクニシャン」などと発言し提訴される
topic-news-6657046.txt	0	topic-news	「週刊文春」でAKB指原交際報道、衝撃内容にファン「絶対許さない」
```

データをpandasでロードします。
ロード時は必ずランダムサンプリングを実施してください。 
実施しない場合は、データに偏りが出て、正確な検証ができなくなります。 
```python
import pandas as pd
df = pd.read_table(
    'dataset.tsv',
    names=['filename', 'label', 'category', 'text']
    ).sample(frac=1, random_state=0).reset_index(drop=True)
```

最後にデータを学習・検証・テスト用=7:2:1に分割して完了です。 

```python
N = len(df)
train_df = df[:int(N * 0.7)] # 学習
dev_df = df[len(train_df): len(train_df) + int(N * 0.2)] # 検証
test_df = df[len(train_df) + len(dev_df):] # テスト
```

### ベースラインを決める

本記事の目的は、「より良い」機械学習モデルをつくるための改善方法を学ぶところにあります。 
そこで、まずはベースラインとなるモデルを作り、ベースラインのモデルに対して、
改善のアプローチをとっていくことにしていきます。

ベースラインとして設定が必要なのは、「前処理」と「文章分類ツール」です。 

今回は、以下の条件をベースラインとします。

```
# 前処理
テキストの加工処理は、sudachiによるトークン化のみとする。　
正規化やストップワード除去などは、実施しない。
文章ベクトルは、Bag-of-Word形式とする。 
```


```
# 文章分類ツール
sklearnのロジスティック回帰をツールとして使う。
ハイパーパラメータは、デフォルトの状態とする。　
```

なお、ベースラインの前処理コードは、以下を使います。

```python

```


### ベースラインの評価

ベースラインに対して、「より良い」モデルの基準に対応する評価を実施します。

「より良い」モデルの基準は、以下2点でした。 
* 性能がすぐれていること
* 処理効率がよいこと

まず性能ですが、性能は検証データに対する正解率(accuracy)評価します。 

処理効率は、以下の時間を評価対象します。 

* 学習データに対する前処理の時間
* 学習時間

本来はリソース使用量も評価対象とすべきですが、
Colaboratoryでの確認が難しいため定量評価はなしとします。 

上記の条件で、評価した結果以下のとおりとなりました。 

```python
# 学習データに対する前処理 -> 10.5s
tp = TextPreprocessing()
bag = tp.get_matrix(train_df.text)
train_X = bag.toarray()
train_y = pd.Series(train_df.label)
```

```python
* 学習 -> 66s
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(train_X, train_y)
```

```python
bag_dev = tp.get_matrix(dev_df.text, mode='dev')
dev_X = bag_dev.toarray()
dev_y = pd.Series(dev_df.label)
dev_score = clf.score(dev_X, dev_y)
print(dev_score) // 正解率 -> 0.7970128988458928
```

### 前処理改善

ようやく本題である改善タスクに入ります。 


### ハイパーパラメータのチューニング



### まとめ

