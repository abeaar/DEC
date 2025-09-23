import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pickle
import pandas as pd

class DECEvaluator:
    """Comprehensive DEC Model Evaluator"""
    
    def __init__(self, results_path='dec_seismic_results.pkl'):
        """Load DEC results"""
        print(" Loading DEC Results...")
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)
        
        self.labels = self.results['cluster_labels']
        self.encoded_data = self.results['encoded_features']
        self.n_clusters = len(np.unique(self.labels))
        
        print(f"    Data shape: {self.encoded_data.shape}")
        print(f"    Active clusters: {self.n_clusters}")
        print(f"    Total windows: {len(self.labels)}")
    
    def clustering_metrics(self):
        """Calculate clustering quality metrics"""
        print("\n Clustering Quality Metrics")
        print("=" * 40)
        
        # Silhouette Score (-1 to 1, higher better)
        sil_score = silhouette_score(self.encoded_data, self.labels)
        
        # Calinski-Harabasz Score (higher better)
        ch_score = calinski_harabasz_score(self.encoded_data, self.labels)
        
        # Davies-Bouldin Score (lower better)
        db_score = davies_bouldin_score(self.encoded_data, self.labels)
        
        print(f"Silhouette Score: {sil_score:.4f}")
        print(f"   ├── Range: [-1, 1]")
        print(f"   └── Interpretation: {'Excellent' if sil_score > 0.7 else 'Good' if sil_score > 0.5 else 'Fair' if sil_score > 0.25 else 'Poor'}")
        
        print(f"\nCalinski-Harabasz Score: {ch_score:.2f}")
        print(f"   ├── Range: [0, ∞]")
        print(f"   └── Interpretation: {'Excellent' if ch_score > 1000 else 'Good' if ch_score > 500 else 'Fair'}")
        
        print(f"\nDavies-Bouldin Score: {db_score:.4f}")
        print(f"   ├── Range: [0, ∞]")
        print(f"   └── Interpretation: {'Excellent' if db_score < 0.5 else 'Good' if db_score < 1.0 else 'Fair'}")
        
        return {
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score
        }
    
    def optimal_clusters_analysis(self, max_clusters=10):
        """Find optimal number of clusters"""
        print(f"\n Optimal Cluster Analysis (2-{max_clusters} clusters)")
        print("=" * 50)
        
        cluster_range = range(2, max_clusters + 1)
        silhouette_scores = []
        ch_scores = []
        db_scores = []
        
        for n in cluster_range:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            temp_labels = kmeans.fit_predict(self.encoded_data)
            
            sil = silhouette_score(self.encoded_data, temp_labels)
            ch = calinski_harabasz_score(self.encoded_data, temp_labels)
            db = davies_bouldin_score(self.encoded_data, temp_labels)
            
            silhouette_scores.append(sil)
            ch_scores.append(ch)
            db_scores.append(db)
            
            print(f"{n} clusters: Silhouette={sil:.3f}, CH={ch:.1f}, DB={db:.3f}")
        
        # Find optimal
        optimal_sil = cluster_range[np.argmax(silhouette_scores)]
        optimal_ch = cluster_range[np.argmax(ch_scores)]
        optimal_db = cluster_range[np.argmin(db_scores)]
        
        print(f"\n Optimal Clusters:")
        print(f"   ├── By Silhouette: {optimal_sil} clusters")
        print(f"   ├── By Calinski-Harabasz: {optimal_ch} clusters")
        print(f"   └── By Davies-Bouldin: {optimal_db} clusters")
        
        # Plot results
        self._plot_cluster_analysis(cluster_range, silhouette_scores, ch_scores, db_scores)
        
        return {
            'range': list(cluster_range),
            'silhouette_scores': silhouette_scores,
            'optimal_silhouette': optimal_sil,
            'optimal_ch': optimal_ch,
            'optimal_db': optimal_db
        }
    
    def cluster_distribution_analysis(self):
        """Analyze cluster distribution"""
        print("\n Cluster Distribution Analysis")
        print("=" * 40)
        
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        
        print("Cluster Distribution:")
        for cluster, count in zip(unique, counts):
            percentage = (count / total) * 100
            print(f"   ├── Cluster {cluster}: {count:,} windows ({percentage:.1f}%)")
        
        # Check balance
        max_pct = max(counts) / total * 100
        min_pct = min(counts) / total * 100
        balance_ratio = max_pct / min_pct
        
        print(f"\nBalance Analysis:")
        print(f"   ├── Largest cluster: {max_pct:.1f}%")
        print(f"   ├── Smallest cluster: {min_pct:.1f}%")
        print(f"   └── Balance ratio: {balance_ratio:.1f}x")
        
        if balance_ratio > 5:
            print("   ️  Warning: Highly imbalanced clusters!")
        elif balance_ratio > 3:
            print("   ️  Moderate imbalance detected")
        else:
            print("    Well-balanced clusters")
        
        return {
            'clusters': unique,
            'counts': counts,
            'percentages': counts / total * 100,
            'balance_ratio': balance_ratio
        }
    
    def stability_test(self, n_runs=5):
        """Test clustering stability"""
        print(f"\n Stability Test ({n_runs} runs)")
        print("=" * 30)
        
        stability_scores = []
        
        for run in range(n_runs):
            # K-means with different random states
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=run, n_init=10)
            temp_labels = kmeans.fit_predict(self.encoded_data)
            
            # Calculate ARI with original labels
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(self.labels, temp_labels)
            stability_scores.append(ari)
            
            print(f"   Run {run+1}: ARI = {ari:.3f}")
        
        avg_stability = np.mean(stability_scores)
        std_stability = np.std(stability_scores)
        
        print(f"\nStability Summary:")
        print(f"   ├── Average ARI: {avg_stability:.3f} ± {std_stability:.3f}")
        
        if avg_stability > 0.8:
            print("   └──  Highly stable clustering")
        elif avg_stability > 0.6:
            print("   └──  Moderately stable clustering")
        else:
            print("   └── ️  Low stability - consider parameter tuning")
        
        return {
            'stability_scores': stability_scores,
            'average': avg_stability,
            'std': std_stability
        }
    
    def training_analysis(self):
        """Analyze training process"""
        print("\n Training Process Analysis")
        print("=" * 35)
        
        ae_losses = self.results['losses_ae']
        dec_losses = self.results['losses_dec']
        
        # Convergence analysis
        ae_final_loss = ae_losses[-1]
        ae_improvement = (ae_losses[0] - ae_losses[-1]) / ae_losses[0] * 100
        
        dec_final_loss = dec_losses[-1]
        dec_improvement = (dec_losses[0] - dec_losses[-1]) / dec_losses[0] * 100
        
        print("Autoencoder Training:")
        print(f"   ├── Initial loss: {ae_losses[0]:.6f}")
        print(f"   ├── Final loss: {ae_final_loss:.6f}")
        print(f"   └── Improvement: {ae_improvement:.1f}%")
        
        print("\nDEC Training:")
        print(f"   ├── Initial KL loss: {dec_losses[0]:.6f}")
        print(f"   ├── Final KL loss: {dec_final_loss:.6f}")
        print(f"   └── Improvement: {dec_improvement:.1f}%")
        
        # Check convergence
        if ae_improvement > 10 and dec_improvement > 80:
            print("\n Training converged successfully!")
        else:
            print("\n️  Training may need more epochs")
        
        return {
            'ae_improvement': ae_improvement,
            'dec_improvement': dec_improvement,
            'ae_final': ae_final_loss,
            'dec_final': dec_final_loss
        }
    
    def _plot_cluster_analysis(self, cluster_range, sil_scores, ch_scores, db_scores):
        """Plot cluster analysis results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Silhouette Score
        axes[0].plot(cluster_range, sil_scores, 'bo-', linewidth=2, markersize=6)
        axes[0].set_title('Silhouette Score')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].grid(True, alpha=0.3)
        
        # Calinski-Harabasz Score
        axes[1].plot(cluster_range, ch_scores, 'ro-', linewidth=2, markersize=6)
        axes[1].set_title('Calinski-Harabasz Score')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('CH Score')
        axes[1].grid(True, alpha=0.3)
        
        # Davies-Bouldin Score
        axes[2].plot(cluster_range, db_scores, 'go-', linewidth=2, markersize=6)
        axes[2].set_title('Davies-Bouldin Score')
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('DB Score')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=200, bbox_inches='tight')
        plt.show()
        print("    Saved: cluster_analysis.png")
    
    def comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print(" COMPREHENSIVE DEC EVALUATION REPORT")
        print("="*60)
        
        # Run all evaluations
        metrics = self.clustering_metrics()
        optimal = self.optimal_clusters_analysis()
        distribution = self.cluster_distribution_analysis()
        stability = self.stability_test()
        training = self.training_analysis()
        
        # Overall assessment
        print(f"\n OVERALL ASSESSMENT")
        print("="*25)
        
        score = 0
        total_checks = 5
        
        # Check 1: Silhouette score
        if metrics['silhouette'] > 0.5:
            print(" Good cluster separation")
            score += 1
        else:
            print("️  Cluster separation could be improved")
        
        # Check 2: Balance
        if distribution['balance_ratio'] < 3:
            print(" Well-balanced clusters")
            score += 1
        else:
            print("️  Cluster imbalance detected")
        
        # Check 3: Stability
        if stability['average'] > 0.6:
            print(" Stable clustering results")
            score += 1
        else:
            print("️  Low clustering stability")
        
        # Check 4: Training convergence
        if training['dec_improvement'] > 80:
            print(" Model training converged well")
            score += 1
        else:
            print("️  Training convergence issues")
        
        # Check 5: Reasonable cluster count
        current_clusters = len(np.unique(self.labels))
        if 3 <= current_clusters <= 7:
            print(" Reasonable number of clusters")
            score += 1
        else:
            print("️  Unusual number of clusters")
        
        print(f"\n Overall Score: {score}/{total_checks} ({score/total_checks*100:.0f}%)")
        
        if score >= 4:
            print(" Excellent DEC results!")
        elif score >= 3:
            print(" Good DEC results with minor improvements needed")
        else:
            print(" DEC results need significant improvements")
        
        return {
            'metrics': metrics,
            'optimal': optimal,
            'distribution': distribution,
            'stability': stability,
            'training': training,
            'overall_score': score,
            'total_checks': total_checks
        }

def main():
    """Run comprehensive DEC evaluation"""
    print(" DEC Model Evaluation Suite")
    print("="*40)
    
    try:
        evaluator = DECEvaluator('dec_seismic_results.pkl')
        report = evaluator.comprehensive_report()
        
        print(f"\n Evaluation complete! Check the generated plots.")
        
    except FileNotFoundError:
        print(" Error: dec_seismic_results.pkl not found!")
        print("   Make sure you've run the DEC training first.")
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()
