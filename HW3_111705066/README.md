# 基於 RoBERTa 的對話回應品質評估模型 (Conversational Response Quality Evaluation)

## 🎯 專案簡介

本專案旨在建立一個二元分類模型，用於評估對話回應的品質。模型採用 **RoBERTa (Robustly Optimized BERT Pretraining Approach)** 架構，通過結合使用者發言 (`u`)、情境陳述 (`s`) 和實際回應 (`r`) 作為輸入，預測回應是「品質優良 (1)」還是「品質不佳 (0)」。

---

## ⚙️ 模型架構與實作細節

### 1. 模型選擇與架構

* [cite_start]**基礎模型**：選擇 **RoBERTa ForSequence Classification** [cite: 9]。
    * [cite_start]模型可使用 `roberta-base` 或從 `textattack/roberta-base-imdb` 權重進行初始化 [cite: 9]。
* [cite_start]**模型類型**：二元分類模型 (Binary Classification)，具有兩個輸出標籤 [cite: 10]：
    * [cite_start]`0`：品質不佳 (poor response quality) [cite: 10]
    * [cite_start]`1`：品質優良 (good response quality) [cite: 11]
* [cite_start]**底層架構**：使用 12 層 Transformer 編碼器，已針對序列理解進行優化 [cite: 12]。
* [cite_start]**套件**：主要使用 **Hugging Face Transformers**、**PyTorch** 和 **Pandas** 進行實作 [cite: 5, 6, 8]。

### 2. 資料預處理與輸入格式

#### 輸入格式 (`Model Input`)
[cite_start]為了保留完整的對話和情境資訊，模型輸入將使用者發言 (`u`)、情境陳述 (`s`) 和回應 (`r`) 合併成特定的格式 [cite: 14, 15]：

[cite_start]這種結構有效地保持了**對話的上下文流動 (Conversational Context)** 和**情境相關性 (Situational Relevance)** [cite: 16, 34]。

#### Tokenization
* [cite_start]使用 Hugging Face 的 Tokenizer 進行分詞、截斷和填充 [cite: 28, 29]。
* [cite_start]**最大序列長度 (`Max sequence length`)**：`256` tokens [cite: 23, 30]。
* [cite_start]**Padding**：用於統一輸入長度 [cite: 31]。

### 3. 超參數設定 (Hyperparameters)

| 參數 | 設定值 | 說明 |
| :--- | :--- | :--- |
| **損失函數 (Loss Function)** | Cross-Entropy Loss | [cite_start]適用於二元分類任務 [cite: 18]。 |
| **學習率 (Learning Rate)** | `2e-5` | [cite_start]經調整以平衡收斂速度和穩定性 [cite: 20]。 |
| **訓練 Batch Size** | [cite_start]`16` | [cite: 21] |
| **評估 Batch Size** | [cite_start]`32` | [cite: 21] |
| **Epochs** | `7` (程式碼中為 7) / `9` (報告中為 9) | [cite_start]允許足夠學習，同時避免過度擬合 [cite: 22]。 |

---

## 🔬 實驗結果與方法比較

### 1. 情境資訊的影響 (Impact of Situations)

* [cite_start]**預測準確性**：相較於**僅**依賴使用者發言 (`u`) 進行預測，結合**情境陳述 (`s`) 和發言**作為輸入時，**準確性顯著提高 (Significantly improved)** [cite: 33]。
* [cite_start]**原因**：組合輸入捕捉了**對話流程**和**情境相關性**，從而增強了模型評估回應的能力 [cite: 34]。

### 2. 模型比較 (Model Comparison)

| 方法 (Method) | 性能 (Performance) | 影響因素 (Factor) |
| :--- | :--- | :--- |
| **RoBERTa** | [cite_start]**最高準確性 (Highest Accuracy)** [cite: 37] | [cite_start]它是 BERT 模型的增強版本 [cite: 37]。 |
| **llama** | [cite_start]第二高準確性 (Second Highest Accuracy) [cite: 37] | [cite_start]雖然是一個大型語言模型，但模型尺寸仍小於 GPT 等專有模型，且可能無法支持複雜任務 [cite: 37, 38]。 |
| **BERT** | [cite_start]第二低準確性 (Second Lowest Accuracy) [cite: 37] | [cite_start]基礎的 NLP 模型，但模型尺寸和複雜度對於多樣化任務可能偏小 [cite: 37]。 |
| **LightGBM** | [cite_start]最低準確性 (Lowest Accuracy) [cite: 37] | [cite_start]作為一個傳統的決策樹模型，可能過於簡單，不適用於此訓練案例 [cite: 37]。 |

> [cite_start]**最佳方法**：**RoBERTa** 表現最佳，因為它在 BERT 的基礎上進行了增強和優化 [cite: 37]。

---