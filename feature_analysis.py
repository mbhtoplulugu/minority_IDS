"""
Advanced Feature Analysis for Attack Detection
Analyzes discriminative power of new features
"""
import numpy as np
import pandas as pd

# Matplotlib backend'ini non-interactive yap (tkinter hatalarn nler)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class FeatureAnalyzer:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.results = {}
        
    def load_data_with_features(self):
        """Load data and create advanced features"""
        # Load original data
        df = joblib.load(self.cache_dir / "raw_df.joblib")
        y = df['attack_cat'].values
        
        # Create advanced features
        from advanced_attack_features import create_all_advanced_features
        
        # Generate new features
        all_new_features, all_new_names = create_all_advanced_features(df)
        
        return df, all_new_features, all_new_names, y
    
    def analyze_feature_discrimination(self, features, feature_names, y, top_k=10):
        """Analyze discriminative power of features"""
        print(f"\n=== FEATURE DISCRIMINATION ANALYSIS ===")
        
        # Calculate mutual information scores
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        mi_scores = []
        for i, feature_name in enumerate(feature_names):
            feature_data = features[:, i]
            # Handle NaN values
            mask = ~np.isnan(feature_data)
            if mask.sum() > 100:  # Minimum samples for reliable MI
                try:
                    mi_score = mutual_info_score(y_encoded[mask], feature_data[mask])
                    mi_scores.append((feature_name, mi_score))
                except:
                    mi_scores.append((feature_name, 0.0))
            else:
                mi_scores.append((feature_name, 0.0))
        
        # Sort by MI score
        mi_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_k} Most Discriminative Features:")
        print("-" * 50)
        for i, (name, score) in enumerate(mi_scores[:top_k]):
            print(f"{i+1:2d}. {name:<30} MI: {score:.4f}")
        
        self.results['mi_scores'] = mi_scores
        return mi_scores
    
    def analyze_class_separation(self, features, feature_names, y, focus_classes=None):
        """Analyze how well features separate attack classes from normal"""
        print(f"\n=== CLASS SEPARATION ANALYSIS ===")
        
        if focus_classes is None:
            focus_classes = ['analysis', 'backdoor', 'dos', 'worms']
        
        separation_results = {}
        
        for class_name in focus_classes:
            if class_name not in y:
                continue
                
            print(f"\nAnalyzing {class_name.upper()} vs Normal:")
            print("-" * 40)
            
            # Create binary labels (class vs normal)
            normal_mask = y == 'normal'
            class_mask = y == class_name
            
            if class_mask.sum() < 10:  # Skip if too few samples
                print(f"Insufficient samples for {class_name}")
                continue
            
            class_results = []
            
            for i, feature_name in enumerate(feature_names):
                feature_data = features[:, i]
                
                # Skip if too many NaN values
                if np.isnan(feature_data).sum() > len(feature_data) * 0.5:
                    continue
                
                # Get feature values for normal and attack class
                normal_values = feature_data[normal_mask]
                class_values = feature_data[class_mask]
                
                # Remove NaN values
                normal_values = normal_values[~np.isnan(normal_values)]
                class_values = class_values[~np.isnan(class_values)]
                
                if len(normal_values) < 10 or len(class_values) < 10:
                    continue
                
                # Statistical tests
                try:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.ks_2samp(normal_values, class_values)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(normal_values)-1)*np.var(normal_values) + 
                                        (len(class_values)-1)*np.var(class_values)) / 
                                       (len(normal_values) + len(class_values) - 2))
                    cohens_d = abs(np.mean(class_values) - np.mean(normal_values)) / (pooled_std + 1e-10)
                    
                    # Mutual information
                    combined_data = np.concatenate([normal_values, class_values])
                    combined_labels = np.concatenate([np.zeros(len(normal_values)), 
                                                    np.ones(len(class_values))])
                    mi_score = mutual_info_score(combined_labels, combined_data)
                    
                    class_results.append({
                        'feature': feature_name,
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'cohens_d': cohens_d,
                        'mi_score': mi_score,
                        'normal_mean': np.mean(normal_values),
                        'class_mean': np.mean(class_values),
                        'normal_std': np.std(normal_values),
                        'class_std': np.std(class_values)
                    })
                    
                except Exception as e:
                    continue
            
            # Sort by discriminative power (combination of KS statistic and Cohen's d)
            class_results.sort(key=lambda x: x['ks_statistic'] * x['cohens_d'], reverse=True)
            
            # Display top features for this class
            print(f"Top 5 discriminative features for {class_name}:")
            for i, result in enumerate(class_results[:5]):
                print(f"{i+1}. {result['feature']:<25} "
                      f"KS: {result['ks_statistic']:.3f} "
                      f"Cohen's d: {result['cohens_d']:.3f} "
                      f"MI: {result['mi_score']:.3f}")
            
            separation_results[class_name] = class_results
        
        self.results['separation'] = separation_results
        return separation_results
    
    def create_feature_distribution_plots(self, features, feature_names, y, 
                                        focus_features=None, focus_classes=None):
        """Create distribution plots for top discriminative features"""
        print(f"\n=== CREATING DISTRIBUTION PLOTS ===")
        
        if focus_classes is None:
            focus_classes = ['normal', 'analysis', 'backdoor', 'dos', 'worms']
        
        if focus_features is None:
            # Use top 6 features from MI analysis
            if 'mi_scores' in self.results:
                focus_features = [name for name, _ in self.results['mi_scores'][:6]]
            else:
                focus_features = feature_names[:6]
        
        # Create subplots
        n_features = min(len(focus_features), 6)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(n_features):
            feature_name = focus_features[i]
            
            # Find feature index
            feature_idx = feature_names.index(feature_name) if feature_name in feature_names else i
            feature_data = features[:, feature_idx]
            
            ax = axes[i]
            
            # Plot distributions for each class
            for class_name in focus_classes:
                if class_name in y:
                    class_mask = y == class_name
                    class_data = feature_data[class_mask]
                    class_data = class_data[~np.isnan(class_data)]
                    
                    if len(class_data) > 10:
                        ax.hist(class_data, bins=30, alpha=0.6, label=class_name, density=True)
            
            ax.set_title(f'{feature_name}', fontsize=10)
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.cache_dir / 'advanced_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution plots saved to: {self.cache_dir / 'advanced_feature_distributions.png'}")
    
    def create_pca_analysis(self, features, feature_names, y, n_components=2):
        """PCA analysis of new features"""
        print(f"\n=== PCA ANALYSIS ===")
        
        # Handle NaN values
        feature_mask = ~np.isnan(features).any(axis=1)
        clean_features = features[feature_mask]
        clean_y = y[feature_mask]
        
        if clean_features.shape[1] == 0:
            print("No valid features for PCA analysis")
            return None, None
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clean_features)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_features)
        
        # Plot PCA results
        plt.figure(figsize=(12, 8))
        
        focus_classes = ['normal', 'analysis', 'backdoor', 'dos', 'worms']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, class_name in enumerate(focus_classes):
            if class_name in clean_y:
                class_mask = clean_y == class_name
                if class_mask.sum() > 0:
                    plt.scatter(pca_features[class_mask, 0], pca_features[class_mask, 1], 
                              c=colors[i], label=class_name, alpha=0.6, s=20)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Analysis of Advanced Attack Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.cache_dir / 'advanced_features_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PCA plot saved to: {self.cache_dir / 'advanced_features_pca.png'}")
        print(f"PC1 explains {pca.explained_variance_ratio_[0]:.2%} of variance")
        print(f"PC2 explains {pca.explained_variance_ratio_[1]:.2%} of variance")
        
        # Feature importance in PCA
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'pc1_loading': abs(pca.components_[0]),
            'pc2_loading': abs(pca.components_[1])
        })
        
        print(f"\nTop 5 features contributing to PC1:")
        top_pc1 = feature_importance.nlargest(5, 'pc1_loading')
        for _, row in top_pc1.iterrows():
            print(f"  {row['feature']:<25} Loading: {row['pc1_loading']:.3f}")
        
        self.results['pca'] = {
            'explained_variance': pca.explained_variance_ratio_,
            'feature_importance': feature_importance
        }
        
        return pca_features, pca
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print(f"\n" + "="*60)
        print(f"ADVANCED FEATURES ANALYSIS SUMMARY REPORT")
        print(f"="*60)
        
        if 'mi_scores' in self.results:
            print(f"\n1. OVERALL FEATURE RANKING (by Mutual Information):")
            print("-" * 50)
            for i, (name, score) in enumerate(self.results['mi_scores'][:10]):
                print(f"{i+1:2d}. {name:<30} MI: {score:.4f}")
        
        if 'separation' in self.results:
            print(f"\n2. CLASS-SPECIFIC DISCRIMINATION:")
            print("-" * 50)
            for class_name, results in self.results['separation'].items():
                if results:
                    best_feature = results[0]
                    print(f"{class_name.upper():<12} Best: {best_feature['feature']:<25} "
                          f"KS: {best_feature['ks_statistic']:.3f}")
        
        if 'pca' in self.results:
            print(f"\n3. PCA ANALYSIS:")
            print("-" * 50)
            variance = self.results['pca']['explained_variance']
            print(f"First 2 components explain {sum(variance[:2]):.2%} of variance")
        
        # Save detailed results
        results_file = self.cache_dir / 'advanced_features_analysis.joblib'
        joblib.dump(self.results, results_file)
        print(f"\nDetailed results saved to: {results_file}")

def run_advanced_feature_analysis(cache_dir):
    """Main function to run comprehensive feature analysis"""
    analyzer = FeatureAnalyzer(cache_dir)
    
    try:
        # Load data and create features
        df, features, feature_names, y = analyzer.load_data_with_features()
        
        print(f"Created {len(feature_names)} advanced features for {len(df)} samples")
        print(f"Feature names: {feature_names}")
        
        if len(feature_names) == 0:
            print("No advanced features could be created")
            return {}
        
        # Run analyses
        mi_scores = analyzer.analyze_feature_discrimination(features, feature_names, y)
        separation_results = analyzer.analyze_class_separation(features, feature_names, y)
        analyzer.create_feature_distribution_plots(features, feature_names, y)
        pca_features, pca = analyzer.create_pca_analysis(features, feature_names, y)
        
        # Generate summary
        analyzer.generate_summary_report()
        
        return analyzer.results
        
    except Exception as e:
        print(f"Error in advanced feature analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}