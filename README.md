# Climate Policy Search Index (CPSI)

**A comprehensive analysis of climate salience in oil-exporting nations using Google Trends data and economic indicators**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Research](https://img.shields.io/badge/Research-Geophysics-green.svg)](#)

## 📋 Project Overview

This project develops a novel **Google Trends Search Index (GTSI)** to quantify climate policy focus in major oil-exporting nations and assess its economic impacts. The research combines digital behavioral data with traditional economic indicators to understand the relationship between climate salience and economic performance in fossil fuel-dependent economies.

### Key Findings
- **10-point increase in climate salience reduces GDP growth by 0.5–0.7%** in oil-dependent nations
- Developed a robust methodology using **PCA to weight climate-related keywords** (explaining ~75% variance)
- Built an **end-to-end Python pipeline** for processing 2004–2024 data across 10 oil-exporting countries
- Validated index against major climate events and policy announcements

## 🎯 Research Objectives

1. **Index Construction**: Develop GTSI using Google Trends data for climate-related keywords
2. **Temporal Analysis**: Track climate engagement trends from 2004-2024
3. **Economic Impact**: Quantify relationship between climate salience and GDP growth
4. **Cross-Country Comparison**: Identify patterns between oil-dependent and non-oil-dependent nations
5. **Policy Insights**: Provide evidence-based recommendations for climate policy frameworks

## 📊 Methodology

### Data Collection
- **Google Trends Keywords**: "Climate Change", "Global Warming", "Renewable Energy", "Net Zero", "Green Energy"
- **Economic Data**: GDP growth rates (World Bank), oil production data (OPEC/EIA)
- **Time Period**: 2004-2024 with monthly aggregation to annual

### Index Construction
1. **Keyword Selection**: Expert-validated climate terms with correlation threshold |r| > 0.3
2. **PCA Weighting**: Country-specific principal component analysis for optimal keyword weights
3. **Normalization**: Min-max scaling (0-100) for cross-country comparability
4. **Validation**: Temporal consistency checks against major climate events

### Statistical Analysis
- Regression analysis: GTSI vs GDP growth and oil production
- Cross-country comparison with control groups
- Sensitivity testing across different weighting schemes

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
Git
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/climate-policy-search-index.git
cd climate-policy-search-index
```

2. **Create virtual environment**
```bash
python -m venv cpsi_env
source cpsi_env/bin/activate  # On Windows: cpsi_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

```python
# Example usage
from src.gtsi_calculator import GTSICalculator
from src.data_processor import DataProcessor

# Initialize the calculator
calculator = GTSICalculator()

# Load and process data
data_processor = DataProcessor()
trends_data = data_processor.load_google_trends_data('data/google_trends/')
economic_data = data_processor.load_economic_data('data/economic/')

# Calculate GTSI
gtsi_scores = calculator.calculate_gtsi(trends_data, method='pca')

# Analyze economic impact
economic_impact = calculator.analyze_economic_correlation(gtsi_scores, economic_data)
```

## 📁 Project Structure

```
climate-policy-search-index/
├── data/
│   ├── google_trends/          # Google Trends CSV files
│   ├── economic/               # GDP and oil production data
│   └── processed/              # Cleaned and processed datasets
├── src/
│   ├── data_processor.py       # Data cleaning and preprocessing
│   ├── gtsi_calculator.py      # Core GTSI calculation methods
│   ├── visualization.py       # Plotting and visualization
│   └── utils.py               # Utility functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_gtsi_development.ipynb
│   ├── 03_economic_analysis.ipynb
│   └── 04_validation_results.ipynb
├── reports/
│   ├── methodology.md
│   ├── results_summary.md
│   └── policy_recommendations.md
├── tests/
│   ├── test_data_processor.py
│   ├── test_gtsi_calculator.py
│   └── test_validation.py
├── requirements.txt
├── README.md
└── LICENSE
```

## 📈 Key Results

### Economic Impact Findings
- **Significant negative correlation** between GTSI and GDP growth in oil-dependent economies
- **R² values ranging 0.13-0.46** across different countries (China showing strongest relationship)
- **3-6 month lead time** for GTSI predicting economic trends

### Regional Variations
- **GCC countries**: Lower baseline climate interest, stronger economic sensitivity
- **North American producers**: Higher climate awareness, more diverse economies
- **Developing oil producers**: Limited digital engagement, policy gaps

### Validation Results
- **Temporal validity**: GTSI peaks align with COP summits, major climate events
- **Cross-validation**: Consistent results across different weighting schemes
- **Robustness**: Stable performance across 2004-2013 and 2014-2024 periods

## 🛠️ Technical Implementation

### Core Technologies
- **Python 3.8+**: Primary development language
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: PCA implementation and statistical modeling
- **Matplotlib/Seaborn**: Data visualization
- **Statsmodels**: Time series analysis and econometric modeling

### Data Pipeline Features
- **Automated data collection** from Google Trends API
- **Robust data cleaning** with outlier detection and missing value handling
- **Scalable processing** for multiple countries and time periods
- **Comprehensive validation** framework with quality checks

## 📚 Research Applications

### Academic Research
- Climate policy effectiveness measurement
- Digital behavior analysis in environmental contexts
- Economic impact assessment of environmental awareness

### Policy Applications
- Real-time monitoring of public climate engagement
- Early warning system for policy resistance
- International climate cooperation frameworks

### Industry Applications
- Investment risk assessment for fossil fuel companies
- Market sentiment analysis for renewable energy sectors
- Corporate sustainability strategy development

## 🎓 Skills Demonstrated

- **Data Science**: Large-scale data processing, statistical modeling, PCA implementation
- **Economic Analysis**: GDP correlation analysis, time series forecasting, econometric modeling
- **Research Methodology**: Hypothesis testing, validation frameworks, reproducible research
- **Programming**: Python development, API integration, automated data pipelines
- **Visualization**: Professional charts, interactive dashboards, academic publication graphics


## 🤝 Contributing

This is an academic research project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📧 Contact

**Aditya Jaideep Khare**
- 🎓 Final Year Masters in Geophysics @ IIT Kharagpur
- 📧 Email: [adityakhare.iitkgp@gmail.com]


## 🙏 Acknowledgments

- **Data Sources**: Google Trends, World Bank, OPEC, U.S. EIA
- **Research Community**: Climate policy researchers and data scientists

## 📊 Citation

If you use this research or methodology in your work, please cite:

```bibtex
```

---

⭐ **If you find this research valuable, please consider starring the repository to support ongoing development!**
