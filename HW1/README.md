# 📚 評論評分預測模型 (Bi-LSTM Model for Review Rating Prediction)

## 🎯 專案簡介
[cite_start]本專案旨在透過自然語言處理（NLP）技術，使用 **雙向長短期記憶網路 (Bi-LSTM)** 模型來預測產品評論的星級評分 (1-5星)。模型整合了評論的標題 (`title`)、內文 (`text`)，以及額外的結構化特徵如是否為驗證購買 (`verified_purchase`) 和有用票數 (`helpful_vote`) 進行多模態輸入訓練 [cite: 4, 9]。

## ⚙️ 模型架構與技術細節

### 1. 特徵選擇與預處理 (Feature Selection & Preprocessing)

[cite_start]模型選擇了 `title`、`text`、`verified_purchase` 和 `helpful_vote` 四個特徵作為輸入 [cite: 4]。

#### 文本預處理 (Text Preprocessing)
* [cite_start]**清理**：將 `title` 和 `text` 轉換為小寫並去除標點符號 [cite: 4]。
* [cite_start]**Tokenization**：透過 Keras 的 `Tokenizer` 將文本轉換為數字序列 [cite: 4, 6]。
* [cite_start]**Padding**：為了選定合適的 padding 大小，檢查了 `title` 和 `text` 的長度分佈 [cite: 6]。
    * [cite_start]`title` 最大長度 (`max_length_title`): 10 [cite: 6]
    * [cite_start]`text` 最大長度 (`max_length_text`): 50 [cite: 6]

### 2. 模型架構 (Bi-LSTM Model)

本模型採用 Bi-LSTM 架構，並結合結構化數據：
* **文本特徵處理**：`title` 和 `text` 經過 Embedding 層後，分別送入 `Bidirectional(LSTM(32))` 層提取特徵。
* **數值特徵處理**：`verified_purchase` 和 `helpful_vote` 通過各自的 Dense 層。
* **合併**：所有特徵通過 `Concatenate` 層合併。
* [cite_start]**避免過度擬合**：在 Dense 層之後使用了 **Dropout (0.5)** [cite: 9]。
* **輸出層**：最終輸出層為 `Dense(5, activation='softmax')`，預測 5 個評分類別。

### 3. 訓練配置 (Training Configuration)
* **優化器 (Optimizer)**: Adam
* **損失函數 (Loss Function)**: `sparse_categorical_crossentropy`
* [cite_start]**回調函數 (Callbacks)**: 使用了 **Early Stopping** 和 **Dropout** 來防止過度擬合 (Overfitting) [cite: 9]。

### 4. 數據準備的影響 (Impact on Different Rating Categories)
* [cite_start]簡單的 Tokenization 和 Padding 對**高評分** (4星和5星) 效果較佳，推測是因高評分評論中的用詞較豐富，模型較能捕捉複雜語境 [cite: 8]。
* [cite_start]對**低評分** (1星或2星) 的預測效果比較差，推測是因這類評論通常文字較短且直接 [cite: 8]。
* [cite_start]**備註**：在實作過程中，曾嘗試使用 **DNN** 來建立模型，但 overfitting 的情況非常嚴重，最終決定選用 **Bi-LSTM** 來實作 [cite: 9]。
