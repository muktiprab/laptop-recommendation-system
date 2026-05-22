# 💻 Laptop Recommendation System

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

> A content-based laptop recommendation system built with TF-IDF and Cosine Similarity, deployed as an interactive web app via Streamlit.

🔗 **Live App:** [laptopia-mukti.streamlit.app](https://laptop-recommendation-mukti.streamlit.app/)

---

## 📸 Preview

![Laptopia Preview](https://raw.githubusercontent.com/Muktiprab007/laptop-recommendation-system/streamlit-app/preview.png)

---

## 📂 Repository Structure

```
├── main (this branch)
│   ├── smartprix_laptop.csv       # Raw dataset
│   ├── laptop_recommendation.ipynb  # Full notebook
│   └── README.md
│
└── streamlit-app
    ├── app.py                     # Streamlit web app
    ├── laptop_cleaned.csv         # Cleaned dataset
    ├── cosine_sim.pkl             # Cosine similarity matrix
    ├── tfidf.pkl                  # TF-IDF vectorizer
    ├── scaler.pkl                 # MinMaxScaler
    └── requirements.txt
```

---

## 📊 Dataset

**Source:** [Smartprix Laptop Specifications & Prices — Kaggle](https://www.kaggle.com/datasets/souravghosh999/laptop-specifications-and-prices-smartprix)

| Feature | Detail |
|---|---|
| Total records | 947 laptops |
| Features | 18 columns |
| Price range | ₹10,999 – ₹6,20,990 |
| Rating range | 4.0 – 4.75 |
| Brands covered | 26 brands |

---

## ⚙️ Methodology

### 1. Data Cleaning
- Removed duplicate columns (`screen_size.1`)
- Fixed corrupted OS values
- Removed rows with invalid data
- Cleaned hidden Unicode characters (`\u200e`) in laptop names
- Converted price from INR to IDR

### 2. Feature Engineering

| New Feature | Description |
|---|---|
| `ram_gb` | Extracted numeric RAM value from string |
| `storage_gb` | Extracted storage in GB (TB converted) |
| `gpu_type` | Categorized GPU: NVIDIA / AMD / Intel Arc / Apple / Integrated / etc. |
| `processor_brand` | Extracted: Intel / AMD / Apple / Qualcomm / MediaTek |
| `laptop_category` | Rule-based: Gaming / Ultrabook / MacBook / Business / Daily / etc. |
| `price_category` | Low / Entry / Premium / Flagship |
| `brand` | Extracted from laptop name |

### 3. Group-Based Imputation
Missing `cores` and `threads` values were filled using the median value of laptops with the same processor. If no reference exists, left empty.

### 4. Content-Based Filtering

```
Combined Text Feature (TF-IDF)
  brand + processor_brand + laptop_category +
  price_category + gpu_type + os + disk_type
        +
Normalized Numerical Features (MinMaxScaler)
  ram_gb + storage_gb + price_idr + rating
        ↓
Weighted Matrix (60% text + 40% numerical)
        ↓
Cosine Similarity Matrix (947 × 947)
```

---

## 🚀 App Features

- **Search** — find laptops by partial name, click any result to get similar recommendations
- **Recommend by Filter** — filter by category, brand, processor, price range, GPU, OS, and disk type

---

## 🛠️ Tech Stack

| Tool | Usage |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | TF-IDF, MinMaxScaler, Cosine Similarity |
| SciPy | Sparse matrix operations |
| Joblib | Model serialization |
| Streamlit | Web app deployment |

---

## 👤 Author

**Mukti Prabowo**
Final-year Computer Science student at Universitas Lampung

[![GitHub](https://img.shields.io/badge/GitHub-Muktiprab007-181717?style=flat-square&logo=github)](https://github.com/Muktiprab007)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mukti%20Prabowo-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/mukti-prabowo)
