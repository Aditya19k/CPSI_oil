"""
Google Trends Search Index (GTSI) Calculator

This module contains the core functionality for calculating the Google Trends Search Index
for climate policy focus in oil-exporting nations.

Author: Om Narhari Gawande
Date: June 2024
Project: Climate Policy Search Index (CPSI)
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class GTSICalculator:
    """
    A class to calculate Google Trends Search Index (GTSI) for climate policy analysis.
    
    This class implements the methodology described in the research paper:
    "Building the Google Trends Search Index (GTSI) for Oil-Rich Countries: 
    Quantifying Climate Focus Amid Economic Development"
    """
    
    def __init__(self, keywords=None, correlation_threshold=0.3):
        """
        Initialize the GTSI Calculator.
        
        Parameters:
        -----------
        keywords : list, optional
            List of climate-related keywords to analyze.
            Default: ["Climate Change", "Global Warming", "Renewable Energy", "Net Zero", "Green Energy"]
        correlation_threshold : float, optional
            Minimum correlation threshold for keyword inclusion (default: 0.3)
        """
        if keywords is None:
            self.keywords = [
                "Climate Change",
                "Global Warming", 
                "Renewable Energy",
                "Net Zero",
                "Green Energy"
            ]
        else:
            self.keywords = keywords
            
        self.correlation_threshold = correlation_threshold
        self.pca_weights = {}
        self.gtsi_scores = {}
        
    def load_google_trends_data(self, data_path, country):
        """
        Load and preprocess Google Trends data for a specific country.
        
        Parameters:
        -----------
        data_path : str
            Path to the Google Trends CSV file
        country : str
            Country name
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed Google Trends data
        """
        try:
            # Load the data
            df = pd.read_csv(data_path)
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter for required keywords
            keyword_columns = [col for col in df.columns if any(keyword.lower() in col.lower() 
                                                              for keyword in self.keywords)]
            
            # Create the processed dataframe
            processed_df = df[['date'] + keyword_columns].copy()
            
            # Aggregate from monthly to annual
            processed_df['year'] = processed_df['date'].dt.year
            annual_df = processed_df.groupby('year')[keyword_columns].mean().reset_index()
            
            # Remove years with insufficient data (less than 10 months)
            monthly_counts = processed_df.groupby('year').size()
            valid_years = monthly_counts[monthly_counts >= 10].index
            annual_df = annual_df[annual_df['year'].isin(valid_years)]
            
            return annual_df
            
        except Exception as e:
            print(f"Error loading Google Trends data for {country}: {str(e)}")
            return None
    
    def filter_keywords_by_correlation(self, trends_data, economic_data, gdp_column='gdp_growth'):
        """
        Filter keywords based on correlation with GDP growth.
        
        Parameters:
        -----------
        trends_data : pandas.DataFrame
            Google Trends data with keyword columns
        economic_data : pandas.DataFrame
            Economic data with GDP growth column
        gdp_column : str
            Name of the GDP growth column
            
        Returns:
        --------
        list
            List of keywords that meet the correlation threshold
        """
        # Merge datasets on year
        merged_data = pd.merge(trends_data, economic_data, on='year', how='inner')
        
        valid_keywords = []
        correlations = {}
        
        keyword_columns = [col for col in trends_data.columns if col != 'year']
        
        for keyword_col in keyword_columns:
            try:
                # Calculate Pearson correlation
                corr, p_value = pearsonr(merged_data[keyword_col], merged_data[gdp_column])
                correlations[keyword_col] = {'correlation': corr, 'p_value': p_value}
                
                # Check if meets threshold
                if abs(corr) >= self.correlation_threshold:
                    valid_keywords.append(keyword_col)
                    
            except Exception as e:
                print(f"Error calculating correlation for {keyword_col}: {str(e)}")
                continue
        
        print(f"Keywords meeting correlation threshold (|r| >= {self.correlation_threshold}): {len(valid_keywords)}")
        return valid_keywords, correlations
    
    def calculate_pca_weights(self, trends_data, country):
        """
        Calculate PCA-based weights for keywords for a specific country.
        
        Parameters:
        -----------
        trends_data : pandas.DataFrame
            Google Trends data for the country
        country : str
            Country name
            
        Returns:
        --------
        dict
            Dictionary of keyword weights
        """
        try:
            # Extract keyword columns (exclude 'year')
            keyword_columns = [col for col in trends_data.columns if col != 'year']
            keyword_data = trends_data[keyword_columns]
            
            # Standardize the data
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(keyword_data)
            
            # Apply PCA
            pca = PCA()
            pca.fit(standardized_data)
            
            # Get first principal component loadings (explains 65-75% variance typically)
            pc1_loadings = pca.components_[0]
            explained_variance = pca.explained_variance_ratio_[0]
            
            print(f"First PC explains {explained_variance:.3f} of variance for {country}")
            
            # Square the loadings to emphasize influence
            squared_loadings = pc1_loadings ** 2
            
            # Normalize to sum to 1
            weights = squared_loadings / np.sum(squared_loadings)
            
            # Create weights dictionary
            weight_dict = dict(zip(keyword_columns, weights))
            
            # Store for the country
            self.pca_weights[country] = weight_dict
            
            return weight_dict
            
        except Exception as e:
            print(f"Error calculating PCA weights for {country}: {str(e)}")
            return None
    
    def calculate_gtsi(self, trends_data, country, weights=None):
        """
        Calculate the Google Trends Search Index (GTSI) for a country.
        
        Parameters:
        -----------
        trends_data : pandas.DataFrame
            Google Trends data for the country
        country : str
            Country name
        weights : dict, optional
            Pre-calculated weights. If None, will calculate PCA weights.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with year and GTSI scores
        """
        try:
            # Get or calculate weights
            if weights is None:
                weights = self.calculate_pca_weights(trends_data, country)
                if weights is None:
                    return None
            
            # Extract keyword columns
            keyword_columns = [col for col in trends_data.columns if col != 'year']
            
            # Calculate weighted contributions
            gtsi_raw = np.zeros(len(trends_data))
            
            for i, keyword_col in enumerate(keyword_columns):
                if keyword_col in weights:
                    contribution = trends_data[keyword_col] * weights[keyword_col]
                    gtsi_raw += contribution
            
            # Min-max normalization to 0-100 scale
            gtsi_min = np.min(gtsi_raw)
            gtsi_max = np.max(gtsi_raw)
            
            if gtsi_max > gtsi_min:
                gtsi_normalized = 100 * (gtsi_raw - gtsi_min) / (gtsi_max - gtsi_min)
            else:
                gtsi_normalized = np.full_like(gtsi_raw, 50)  # Default to midpoint if no variation
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'year': trends_data['year'],
                'gtsi_raw': gtsi_raw,
                'gtsi': gtsi_normalized
            })
            
            # Store results
            self.gtsi_scores[country] = results_df
            
            return results_df
            
        except Exception as e:
            print(f"Error calculating GTSI for {country}: {str(e)}")
            return None
    
    def analyze_economic_correlation(self, gtsi_data, economic_data, country):
        """
        Analyze correlation between GTSI and economic indicators.
        
        Parameters:
        -----------
        gtsi_data : pandas.DataFrame
            GTSI scores for the country
        economic_data : pandas.DataFrame
            Economic data (GDP growth, oil production, etc.)
        country : str
            Country name
            
        Returns:
        --------
        dict
            Dictionary with correlation analysis results
        """
        try:
            # Merge datasets
            merged_data = pd.merge(gtsi_data, economic_data, on='year', how='inner')
            
            if len(merged_data) < 3:
                print(f"Insufficient data for correlation analysis for {country}")
                return None
            
            results = {'country': country, 'n_observations': len(merged_data)}
            
            # Economic indicators to analyze
            economic_indicators = ['gdp_growth', 'oil_production', 'oil_revenue']
            
            for indicator in economic_indicators:
                if indicator in merged_data.columns:
                    try:
                        corr, p_value = pearsonr(merged_data['gtsi'], merged_data[indicator])
                        results[f'{indicator}_correlation'] = corr
                        results[f'{indicator}_pvalue'] = p_value
                        results[f'{indicator}_significant'] = p_value < 0.05
                    except:
                        results[f'{indicator}_correlation'] = np.nan
                        results[f'{indicator}_pvalue'] = np.nan
                        results[f'{indicator}_significant'] = False
            
            return results
            
        except Exception as e:
            print(f"Error in economic correlation analysis for {country}: {str(e)}")
            return None
    
    def validate_temporal_alignment(self, gtsi_data, event_dates, country, window_months=3):
        """
        Validate GTSI temporal alignment with major climate events.
        
        Parameters:
        -----------
        gtsi_data : pandas.DataFrame
            GTSI scores with date information
        event_dates : list
            List of major climate event dates
        country : str
            Country name
        window_months : int
            Window for detecting GTSI peaks around events
            
        Returns:
        --------
        dict
            Validation results
        """
        try:
            # Convert event dates to years for annual data
            event_years = [pd.to_datetime(date).year for date in event_dates]
            
            # Find GTSI peaks (above 75th percentile)
            gtsi_threshold = np.percentile(gtsi_data['gtsi'], 75)
            peak_years = gtsi_data[gtsi_data['gtsi'] > gtsi_threshold]['year'].tolist()
            
            # Check alignment
            aligned_events = 0
            for event_year in event_years:
                if any(abs(event_year - peak_year) <= 1 for peak_year in peak_years):
                    aligned_events += 1
            
            alignment_ratio = aligned_events / len(event_years) if event_years else 0
            
            results = {
                'country': country,
                'total_events': len(event_years),
                'aligned_events': aligned_events,
                'alignment_ratio': alignment_ratio,
                'peak_years': peak_years,
                'event_years': event_years
            }
            
            return results
            
        except Exception as e:
            print(f"Error in temporal validation for {country}: {str(e)}")
            return None
    
    def generate_summary_report(self, countries):
        """
        Generate a summary report of GTSI analysis for all countries.
        
        Parameters:
        -----------
        countries : list
            List of countries analyzed
            
        Returns:
        --------
        dict
            Summary statistics and findings
        """
        summary = {
            'total_countries': len(countries),
            'countries_analyzed': [],
            'average_variance_explained': [],
            'gtsi_statistics': {}
        }
        
        for country in countries:
            if country in self.gtsi_scores:
                summary['countries_analyzed'].append(country)
                
                # GTSI statistics
                gtsi_data = self.gtsi_scores[country]['gtsi']
                summary['gtsi_statistics'][country] = {
                    'mean': np.mean(gtsi_data),
                    'std': np.std(gtsi_data),
                    'min': np.min(gtsi_data),
                    'max': np.max(gtsi_data),
                    'years_covered': len(gtsi_data)
                }
        
        return summary


# Example usage and testing functions
def example_usage():
    """
    Example of how to use the GTSICalculator class.
    """
    # Initialize calculator
    calculator = GTSICalculator()
    
    # Example workflow (would need actual data files)
    """
    # Load data for a country
    trends_data = calculator.load_google_trends_data('data/saudi_arabia_trends.csv', 'Saudi Arabia')
    economic_data = pd.read_csv('data/saudi_arabia_economic.csv')
    
    # Filter keywords by correlation
    valid_keywords, correlations = calculator.filter_keywords_by_correlation(
        trends_data, economic_data
    )
    
    # Calculate GTSI
    gtsi_results = calculator.calculate_gtsi(trends_data, 'Saudi Arabia')
    
    # Analyze economic correlations
    economic_analysis = calculator.analyze_economic_correlation(
        gtsi_results, economic_data, 'Saudi Arabia'
    )
    
    print(f"GTSI calculated for Saudi Arabia: {len(gtsi_results)} years of data")
    print(f"Economic correlation results: {economic_analysis}")
    """
    
    print("GTSICalculator initialized successfully!")
    print(f"Default keywords: {calculator.keywords}")
    print(f"Correlation threshold: {calculator.correlation_threshold}")


if __name__ == "__main__":
    example_usage()