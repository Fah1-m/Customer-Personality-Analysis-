# Customer Personality Analysis — Unsupervised Segmentation

An unsupervised learning project that segments customers of a retail/marketing dataset into distinct groups based on their demographics, spending habits, and campaign responsiveness — using PCA for dimensionality reduction and Agglomerative Clustering to find the segments.

## Project goal

Businesses rarely treat all customers the same, but without labeled "customer types" to train on, you need clustering rather than classification. The goal here was to take a raw marketing campaign dataset and answer: **can we find natural customer segments, and do they actually behave differently in terms of spending and promotion uptake?**

## Dataset

`marketing_campaign.csv` — a tab-separated file with per-customer records including:
- Demographics: `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`
- Enrollment info: `Dt_Customer`, `Recency`
- Spending by category: wine, fruits, meat, fish, sweets, gold products
- Campaign history: whether each of 5 past campaigns was accepted, plus complaints and final campaign response

## Pipeline

### 1. Cleaning

Missing values were dropped outright rather than imputed, since the dataset is small enough that a full-row drop didn't cost much data:

```python
data = data.dropna()
```

`Dt_Customer` gets parsed into an actual date, which lets me compute how long each customer has been enrolled relative to the most recent signup in the dataset:

```python
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
d1 = max(dates)
data['Customer_For'] = [d1 - i for i in dates]
```

### 2. Feature engineering

This is where most of the signal gets built. A handful of derived features turn raw columns into things that are actually meaningful for segmentation:

```python
data['Age'] = 2025 - data['Year_Birth']

data['Spent'] = (data['MntWines'] + data['MntFruits'] + data['MntMeatProducts']
                  + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds'])

data['Living_With'] = data['Marital_Status'].replace({
    'Married': 'Partner', 'Together': 'Partner',
    'Absurd': 'Alone', 'Widow': 'Alone', 'YOLO': 'Alone',
    'Single': 'Alone', 'Divorced': 'Alone'
})

data['Children'] = data['Kidhome'] + data['Teenhome']
data['Family_Size'] = data['Living_With'].replace({'Alone': 1, 'Partner': 2}) + data['Children']
data['Is_Parent'] = np.where(data['Children'] > 0, 1, 0)

data['Education'] = data['Education'].replace({
    'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate',
    'Graduation': 'Graduate', 'Master': 'Postgraduate', 'PhD': 'Postgraduate'
})
```

A few noisy/redundant columns (`Marital_Status`, `Dt_Customer`, `Z_CostContact`, `Z_Revenue`, `Year_Birth`, `ID`) get dropped once their information is captured elsewhere, and spending columns are renamed for readability (`MntWines` → `Wines`, etc.).

### 3. Outlier removal

A quick pairplot of `Income`, `Recency`, `Customer_For`, `Age`, `Spent`, and `Is_Parent` surfaced a couple of extreme outliers — a handful of implausible ages and one absurd income value — so a simple cap handled it:

```python
data = data[(data["Age"] < 90)]
data = data[(data["Income"] < 600000)]
```

### 4. Encoding & scaling

Categorical columns (`Education`, `Living_With`) are label-encoded, and the full feature set is standardized before any distance-based modeling:

```python
LE = LabelEncoder()
for i in object_cols:
    data[i] = LE.fit_transform(data[i])

scaler = StandardScaler()
scaled_ds = pd.DataFrame(scaler.fit_transform(ds), columns=ds.columns)
```

Campaign-acceptance and complaint columns were excluded from the clustering feature set itself (`cols_del`) — they're kept aside to evaluate the clusters afterward rather than to define them, so the segmentation is based on who the customer *is* and how much they *spend*, not on how they already responded to marketing.

### 5. Dimensionality reduction (PCA)

With scaled features in hand, PCA compresses everything down to 3 components for both computational ease and visualization:

```python
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=["col1", "col2", "col3"])
```

A 3D scatter of these three components gives a first visual gut-check that the data actually has some structure to cluster.

### 6. Finding the right number of clusters

Rather than guessing `k`, the elbow method (via `yellowbrick`) is used on the PCA-reduced data to find the inflection point in inertia:

```python
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()
```

This pointed to **4 clusters** as a reasonable choice.

### 7. Clustering

Agglomerative (hierarchical) clustering is used instead of K-Means for the final assignment:

```python
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(PCA_ds)
data["Clusters"] = yhat_AC
```

## Results & cluster profiling

Once customers were assigned to one of 4 clusters, several plots were used to characterize what actually distinguishes them:

- **3D cluster plot** on the PCA components — confirms the clusters are reasonably well separated in reduced space.
- **Cluster size distribution** (`countplot`) — shows whether segments are balanced or skewed.
- **Income vs. Spending scatter, colored by cluster** — the clearest business-relevant view: it separates high-income/high-spend customers from low-income/low-spend ones, with the middle segments filling in between.
- **Spend distribution per cluster** (swarm + boxen plot) — confirms the segments differ meaningfully in total spend, not just superficially.
- **Total promotions accepted per cluster** and **deals purchased per cluster** — connects the *unsupervised* segments back to *marketing behavior* they weren't trained on, which is really the payoff: do naturally-occurring segments respond differently to promotions? The plots suggest yes — some clusters are much more promotion-responsive than others.
- **Joint KDE plots** of `Spent` against personal features (age, family size, parenthood, education, etc.) per cluster — used to sketch a qualitative "persona" for each segment (e.g. a segment of older, higher-income, non-parent big spenders vs. a segment of younger, budget-conscious parents).

## Tech stack

- `pandas`, `numpy` — data wrangling & feature engineering
- `scikit-learn` — `LabelEncoder`, `StandardScaler`, `PCA`, `AgglomerativeClustering`, `KMeans`
- `yellowbrick` — elbow-method visualization for choosing `k`
- `matplotlib`, `seaborn` — EDA and cluster visualization (including custom color palettes for consistency across plots)

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick
```

Place `marketing_campaign.csv` in the working directory (tab-separated), then run the notebook top to bottom. Cluster labels end up in the `Clusters` column of the final `data` DataFrame.

## Possible next steps

- Compare Agglomerative Clustering against K-Means and DBSCAN outputs on the same PCA space to check cluster stability
- Use silhouette score or Davies-Bouldin index to quantify cluster separation instead of relying on visual inspection alone
- Turn the qualitative personas into a labeled summary table (name each cluster, e.g. "High-Value Loyalists", "Budget Families") for easier stakeholder communication
- Test whether the clusters predict `Response` (acceptance of the final campaign) better than a simple income/spend cutoff would — this would validate that the segmentation adds value over naive rules
