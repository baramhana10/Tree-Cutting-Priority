# 🌲 Tree Cutting Priority Analysis - Fire Creek

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-1.0+-green.svg)](https://geopandas.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Spatial Data Analysis - Homework 2**  
> Multi-criteria decision analysis (MCDA) for wildfire risk mitigation through strategic tree cutting prioritization.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Output Files](#-output-files)
- [Visualization](#-visualization)
- [Authors](#-authors)

---

## 🎯 Overview

This project identifies and prioritizes tree cutting zones for wildfire risk mitigation in the **Fire Creek** area using Python-based spatial analysis. The analysis evaluates **80 grid zones** across **5 weighted criteria** to produce normalized priority scores (0-100) classified into actionable priority classes.

### Key Objectives

1. Calculate tree cutting priority for each zone in the cutting grid
2. Apply multi-criteria decision analysis (MCDA) with weighted factors
3. Normalize all scores to a 0-100 scale for consistent comparison
4. Classify zones into priority classes: **Low**, **Medium**, **High**, **Critical**
5. Generate visualization charts for decision-making support

---

## ✨ Features

- **Multi-Factor Analysis**: Evaluates 5 different risk factors
- **Weighted Scoring**: Customizable weights for each factor
- **Normalized Scores**: All factors scaled to 0-100 for fair comparison
- **Priority Classification**: Automatic classification into 4 priority levels
- **Multiple Output Formats**: Shapefile, GeoPackage, and CSV
- **Visualization Charts**: Bar charts and pie charts for analysis
- **QGIS Compatible**: Direct integration with QGIS for mapping

---

## 📁 Project Structure

```
Raster/
├── data/                           # Input shapefiles
│   ├── CuttingGrids.shp           # Main analysis grid (80 zones)
│   ├── SBNFMortalityt.shp         # Tree mortality data
│   ├── Communityfeatures.shp      # Community infrastructure
│   ├── EgressRoutes.shp           # Evacuation routes
│   ├── PopulatedAreast.shp        # Populated areas
│   ├── Transmission.shp           # High-voltage lines
│   ├── SubTransmission.shp        # Sub-transmission lines
│   ├── DistCircuits.shp           # Distribution circuits
│   ├── Substations.shp            # Power substations
│   ├── PoleTopSubs.shp            # Pole-top substations
│   └── TownBoundary.shp           # Town boundary reference
│
├── output/                         # Generated outputs
│   ├── TreeCuttingPriority_*.shp  # Priority shapefile
│   ├── TreeCuttingPriority_*.gpkg # GeoPackage output
│   ├── TreeCuttingPriority.csv    # CSV tabular data
│   └── TreeCuttingPriority_Charts.png  # Visualization charts
│
├── tree_cutting_priority.py        # Main analysis script
└── README.md                       # This file
```

---

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/tree-cutting-priority.git
cd tree-cutting-priority
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install geopandas pandas numpy matplotlib
```

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| geopandas | ≥1.0 | Spatial data manipulation |
| pandas | ≥2.0 | Data processing |
| numpy | ≥1.24 | Numerical calculations |
| matplotlib | ≥3.7 | Chart visualization |

---

## 🚀 Usage

### Running the Analysis

```bash
python tree_cutting_priority.py
```

### Expected Output

```
============================================================
Tree Cutting Priority Analysis - Fire Creek
============================================================

[STEP 1] Loading shapefiles...
[STEP 2] Checking and aligning coordinate reference systems...
[STEP 3] Initializing priority calculation...
[STEP 4] Calculating Tree Mortality Score...
[STEP 5] Calculating Community Features Score...
[STEP 6] Calculating Egress Routes Score...
[STEP 7] Calculating Populated Areas Score...
[STEP 8] Calculating Electric Utilities Score...
[STEP 9] Calculating Overall Tree Cutting Priority...
[STEP 10] Saving results...
[STEP 11] Display Summary...
[STEP 12] Generating visualization charts...

============================================================
Analysis Complete!
============================================================
```

---

## 📊 Methodology

### Analysis Workflow

```
CuttingGrids (80 Zones)
        ↓
① Load & Align CRS → EPSG:26711
        ↓
② Calculate Factor Scores → 5 factors
        ↓
③ Normalize Scores → 0-100 scale
        ↓
④ Apply Weighted MCDA → Priority Score
        ↓
⑤ Classify Priority → Low/Medium/High/Critical
        ↓
⑥ Generate Charts → Visualization
```

### Factor Weights

| Factor | Weight | Description |
|--------|:------:|-------------|
| **Tree Mortality** | 30% | Dead/dying trees indicating fire fuel load |
| **Egress Routes** | 20% | Emergency evacuation route protection |
| **Populated Areas** | 20% | Residential zone life safety |
| **Community Features** | 15% | Critical infrastructure protection |
| **Electric Utilities** | 15% | Power infrastructure protection |

### Priority Classification

| Class | Score Range | Action Required |
|-------|:-----------:|-----------------|
| 🔴 **Critical** | 75 - 100 | Immediate action required |
| 🟠 **High** | 50 - 74 | Near-term action needed |
| 🟡 **Medium** | 25 - 49 | Scheduled maintenance |
| 🟢 **Low** | 0 - 24 | Routine monitoring |

### Normalization Formula

```python
Normalized_Score = (Raw_Score / Max_Raw_Score) × 100
```

### Weighted Priority Score

```python
priority_score = (
    mortality_score × 0.30 +
    community_score × 0.15 +
    egress_score × 0.20 +
    populated_score × 0.20 +
    utility_score × 0.15
)
```

---

## 📈 Results

### Priority Distribution

| Priority Class | Count | Percentage |
|----------------|:-----:|:----------:|
| 🔴 Critical | 0 | 0.0% |
| 🟠 High | 5 | 6.3% |
| 🟡 Medium | 43 | 53.8% |
| 🟢 Low | 32 | 40.0% |

### Score Statistics

| Metric | Value |
|--------|:-----:|
| Mean Priority Score | 29.37 |
| Maximum Score | 60.73 |
| Minimum Score | 8.16 |
| Standard Deviation | 13.71 |

### Top 5 Priority Zones

| Rank | Grid ID | Score | Class |
|:----:|:-------:|:-----:|:-----:|
| 1 | 163 | 60.73 | 🟠 High |
| 2 | 158 | 55.51 | 🟠 High |
| 3 | 130 | 54.48 | 🟠 High |
| 4 | 157 | 51.84 | 🟠 High |
| 5 | 162 | 50.74 | 🟠 High |

---

## 📂 Output Files

### Generated Files

| File | Format | Description |
|------|--------|-------------|
| `TreeCuttingPriority_*.shp` | Shapefile | Spatial output for GIS |
| `TreeCuttingPriority_*.gpkg` | GeoPackage | Single-file spatial format |
| `TreeCuttingPriority.csv` | CSV | Tabular data for analysis |
| `TreeCuttingPriority_Charts.png` | PNG | Visualization charts (300 DPI) |

### Output Schema

| Field | Description | Range |
|-------|-------------|:-----:|
| `mort_score` | Normalized mortality score | 0-100 |
| `comm_score` | Normalized community score | 0-100 |
| `egrs_score` | Normalized egress score | 0-100 |
| `pop_score` | Normalized populated score | 0-100 |
| `util_score` | Normalized utility score | 0-100 |
| `priority_s` | Final weighted priority score | 0-100 |
| `prior_cls` | Priority classification | Low/Medium/High/Critical |
| `prior_rank` | Priority rank (1 = highest) | 1-80 |

---

## 🗺 Visualization

### Charts Generated

The script automatically generates three charts:

1. **Bar Chart**: Number of grid zones per priority class
2. **Pie Chart**: Percentage distribution of priority classes
3. **Horizontal Bar Chart**: Top 10 highest priority zones

### QGIS Styling

To visualize in QGIS with priority colors:

1. Load `TreeCuttingPriority_*.shp`
2. Right-click → Properties → Symbology
3. Select **Categorized**
4. Set Value to `prior_cls`
5. Click **Classify**
6. Apply colors:
   - Critical: `#E53935` (Red)
   - High: `#FB8C00` (Orange)
   - Medium: `#FDD835` (Yellow)
   - Low: `#43A047` (Green)

---

## 🔧 Configuration

### Adjusting Weights

Modify the `WEIGHTS` dictionary in `tree_cutting_priority.py`:

```python
WEIGHTS = {
    'mortality': 0.30,      # 30% - Tree mortality
    'community': 0.15,      # 15% - Community features
    'egress': 0.20,         # 20% - Egress routes
    'populated': 0.20,      # 20% - Populated areas
    'utility': 0.15         # 15% - Electric utilities
}
```

### Adjusting Classification Thresholds

Modify the `classify_priority` function:

```python
def classify_priority(score):
    if score >= 75:
        return 'Critical'
    elif score >= 50:
        return 'High'
    elif score >= 25:
        return 'Medium'
    else:
        return 'Low'
```

---

## 👥 Authors

- **Ameer Saleh**
- **Bara Mhana**

**Course:** Spatial Data Analysis  
**Assignment:** Homework 2 - Tree Cutting Priority Analysis  
**Date:** December 2025

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- San Bernardino National Forest (SBNF) for mortality data
- Fire Creek community for infrastructure data
- Course instructors for guidance and support

---

<p align="center">
  Made with ❤️ for Spatial Data Analysis
</p>

