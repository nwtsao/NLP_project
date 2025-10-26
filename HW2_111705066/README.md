# 基於 BERT 的推文多標籤分類模型 (Multi-Label Classification of Tweet Concerns)

## 🎯 專案簡介

本專案實作了一個基於 **BERT (Bidirectional Encoder Representations from Transformers)** 模型，用於對推文（Tweets）進行**多標籤分類**（Multi-Label Classification）。模型的目標是識別一則推文可能同時表達的**多種**關注類別或擔憂。

---

## ⚙️ 模型與資料處理細節

### 1. 資料預處理 (Data Preprocessing)

1.  [cite_start]**標籤映射 (Label Mapping)**[cite: 3, 4, 8, 10, 12]:
    * [cite_start]建立 `label_mapping` 字典，將 12 個關注類別（例如 'ineffective', 'unnecessary', 'pharma' 等）映射為 0 到 11 的數值索引 [cite: 3, 4, 8, 10, 12]。
2.  [cite_start]**文本 Tokenization**[cite: 15, 16, 17, 19]:
    * [cite_start]使用 `BertTokenizer` 來自 `bert-base-uncased` 模型對每則推文進行分詞 [cite: 15, 16]。
    * [cite_start]應用 **Padding** 和 **Truncation**，確保所有輸入序列長度固定為 **128** 個 tokens [cite: 17, 19]。
    * [cite_start]將輸入轉換為 PyTorch 張量 [cite: 21]。
3.  [cite_start]**多標籤編碼 (Multi-label Encoding)**[cite: 22]:
    * [cite_start]標籤被轉換為**二進制向量 (One-Hot Encoded)**，以適應多標籤分類問題（例如，如果推文同時包含兩個關注點，標籤可能為 `[1, 0, 1, 0, ...]`） [cite: 22]。

### 2. 模型與超參數 (Model & Hyperparameters)

* **基礎模型**：`BertForSequenceClassification` (基於 `bert-base-uncased`)，配置為 `problem_type="multi_label_classification"`。
* [cite_start]**訓練參數** (通過 `TrainingArguments` 配置)[cite: 27, 29, 30, 31]:
    * [cite_start]**學習率 (Learning Rate)**：`4e-5` [cite: 29][cite_start]。在調參過程中，`4e-5` 比預設的 `2e-5` 提供了略好的收斂效果 [cite: 43]。
    * [cite_start]**Epochs**：8 [cite: 30]。
    * [cite_start]**Batch Size**：8 [cite: 28, 30][cite_start]。較高的批次大小（如 16）在實驗中導致訓練不穩定，且容易造成標準 GPU 記憶體溢出 [cite: 37, 38, 45]。
    * [cite_start]**權重衰減 (Weight Decay)**：`0.01` [cite: 31]。
    * [cite_start]**評估指標**：以 **F1 Score** 作為選擇最佳模型的依據 [cite: 40]。

### 3. 評估指標 (Evaluation Metric)

使用 `compute_metrics` 函數計算以下指標：
* **Accuracy (準確率)**
* **F1 Score** (使用 `average='weighted'`)
* **Precision (精確率)**
* **Recall (召回率)**

---

## 🔬 實驗結果與優化 (Experimental Results & Optimization)

### 1. 最難預測的關注類別 (Most Difficult Categories)

[cite_start]以下類別的預測表現較差，精確率 (Precision) 和召回率 (Recall) 較低 [cite: 49, 50, 52]：

| 類別 (Category) | 困難原因 | 常被誤分類為 (Misclassified As) |
| :--- | :--- | :--- |
| [cite_start]**`none`** [cite: 50, 51] | [cite_start]許多沒有明確關注點的推文，因文本特徵模糊或重疊，被錯誤地分類到其他類別 [cite: 51]。 | [cite_start]`side-effect` 或 `pharma` (因推文常包含醫學術語) [cite: 56]。 |
| [cite_start]**`conspiracy`** [cite: 52] | [cite_start]語言與其他類別（如 `mandatory` 或 `pharma`）有重疊 [cite: 52]。 | - |
| [cite_start]**`political`** [cite: 52] | [cite_start]語言與其他類別有重疊 [cite: 52]。 | [cite_start]`mandatory` 或 `conspiracy` (因圍繞法規和懷疑論的論述相似) [cite: 57, 58]。 |

### 2. 模型優化方法 (Optimization Methods)

| 方法分類 | [cite_start]成功方法 (Successful Methods) [cite: 61] | [cite_start]失敗方法 (Unsuccessful Methods) [cite: 68] |
| :--- | :--- | :--- |
| **處理數據不平衡** | [cite_start]**權重平衡 (Weight Balancing)** [cite: 62, 63, 64, 65][cite_start]: 在訓練時應用類別加權，以處理數據不平衡問題 [cite: 65]。 | [cite_start]**過度採樣 (Oversampling)** [cite: 69, 70][cite_start]: 對少數類別進行過度採樣，導致模型過度擬合 (Overfitting)，特別是對於如 `conspiracy` 等類別 [cite: 70]。 |
| **提升模型表現** | [cite_start]**數據增強 (Data Augmentation)** [cite: 66][cite_start]: 透過改寫 (paraphrasing) 推文，增加數據的多樣性以提升模型健壯性 [cite: 66]。 | [cite_start]**增加額外特徵 (Adding Features)** [cite: 72, 73][cite_start]: 加入推文的元數據（如使用者資訊）未顯著提升性能，可能是特徵冗餘或噪聲 [cite: 73]。 |
| **優化預測** | [cite_start]**決策閾值調整 (Threshold Tuning)** [cite: 67][cite_start]: 調整每個標籤的決策閾值（例如，對代表性不足的類別設定 > 0.5 的閾值）以減少假陰性 (False Negatives) [cite: 67]。 | [cite_start]**提升模型複雜度 (Model Complexity)** [cite: 74][cite_start]: 測試 `bert-large-uncased` 模型，但性能提升微乎其微，卻大幅增加了訓練時間和 GPU 記憶體需求 [cite: 74]。 |

---
