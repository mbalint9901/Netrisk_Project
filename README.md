# Netrisk Lakásbiztosítási Elemzés

> Prediktív modellezés és ügyfélszegmentáció a Netrisk lakásbiztosítási kampányához

## Projekt Struktúra

```
Netrisk_Project/
??? data/                          # Adatfájlok
?   ??? raw/                      # Nyers adatok
?   ?   ??? netrisk_dataset.csv   # Fõ ügyfél adatbázis
?   ?   ??? county_prices/        # KSH ingatlanárak megyénként
?   ??? processed/                # Feldolgozott adatok
??? notebooks/                    # Jupyter notebookok
?   ??? 01_netrisk_eda_ml.ipynb  # Fõ elemzési notebook
??? src/                          # Forráskód modulok
?   ??? utils.py                  # Segédfüggvények
??? outputs/                      # Kimenetek
?   ??? figures/                  # Generált ábrák
?   ??? models/                   # Mentett modellek
?   ??? reports/                  # Elemzési jelentések
??? requirements.txt              # Python függõségek
??? setup.py                     # Projekt konfiguráció
??? README.md                    # Ez a fájl
```

## ??? Telepítés és Futtatás

### Elõfeltételek

- Python 3.11+
- Git

### 1. Repository klónozása

```bash
git clone https://github.com/mbalint9901/Netrisk_Project.git
cd Netrisk_Project
```

### 2. Virtuális környezet létrehozása

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Függõségek telepítése

```bash
pip install -r requirements.txt
```

### 4. Jupyter Lab indítása

```bash
jupyter lab
```

### 5. Elemzés futtatása

Nyisd meg a `notebooks/01_netrisk_eda_ml.ipynb` fájlt és futtasd végig a cellákat.

## Adatforrások

- **Netrisk ügyfél adatbázis**: 250,000+ ügyfél adatai
- **KSH Ingatlanadattár 2024**: Megyei átlagárak ingatlan típusonként

## Használt Technológiák

### Data Science Stack
- **Pandas & Polars**: Adatmanipuláció
- **NumPy & SciPy**: Numerikus számítások
- **Scikit-learn**: Machine learning alapok
- **XGBoost**: Gradient boosting
- **Optuna**: Hyperparameter optimization

### Vizualizáció
- **Matplotlib**: Alapvetõ ábrák
- **Seaborn**: Statisztikai vizualizáció
- **Plotly**: Interaktív ábrák
- **SHAP**: Model explanability

## Kimenet Fájlok

### Ábrák (`outputs/figures/`)
- `01_log_odds_comparison.png` - Log-odds elemzés
- `02_optimized_feature_importance.png` - Feature fontosság
- `03_shap_*.png` - SHAP elemzések
- `05_top_prospects_segmentation.png` - Ügyfélszegmentek
- `07_optuna_optimization_history.html` - Hyperparameter optimalizálás

### Modellek (`outputs/models/`)
- `xgb_home_insurance.pkl` - Mentett Optuna study

### Jelentések (`outputs/reports/`)
- `02_netrisk_analysis_report.md` - Részletes elemzési jelentés