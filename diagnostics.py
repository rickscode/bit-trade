#!/usr/bin/env python3
"""
Diagnostics System for Bit-Trade v3.0
Validates system functionality, Q-learning performance, and generates reports
"""

import json
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitTradeDiagnostics:
    """Comprehensive diagnostics system for trading bot validation"""
    
    def __init__(self):
        self.reports_dir = Path("reports")
        self.outputs_dir = Path("outputs")
        self.logs_dir = Path("logs")
        
        # Create directories if they don't exist
        self.reports_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.diagnostics_results = {}
        
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run complete diagnostics suite"""
        logger.info("🔍 Starting comprehensive system diagnostics...")
        
        # Core system checks
        self.diagnostics_results['q_learning'] = self._check_q_learning_performance()
        self.diagnostics_results['model_performance'] = self._analyze_model_performance()
        self.diagnostics_results['agent_memory'] = self._validate_agent_memory()
        self.diagnostics_results['timeout_analysis'] = self._analyze_timeouts()
        self.diagnostics_results['system_stability'] = self._check_system_stability()
        
        # Generate comprehensive report
        report_path = self._generate_report()
        
        logger.info(f"✅ Diagnostics complete. Report saved to: {report_path}")
        return self.diagnostics_results
        
    def _check_q_learning_performance(self) -> Dict[str, Any]:
        """Validate Q-learning model selection effectiveness"""
        logger.info("📊 Analyzing Q-learning performance...")
        
        try:
            # Check for Q-learning data
            q_data_path = self.outputs_dir / "q_learning_data.json"
            if not q_data_path.exists():
                return {"status": "warning", "message": "No Q-learning data found"}
            
            with open(q_data_path, 'r') as f:
                q_data = json.load(f)
            
            # Analyze Q-values convergence
            q_values_history = q_data.get('q_values_history', [])
            if len(q_values_history) < 10:
                return {"status": "warning", "message": "Insufficient Q-learning history"}
            
            # Check for learning progress
            recent_values = q_values_history[-10:]
            early_values = q_values_history[:10]
            
            avg_recent = np.mean([np.mean(list(q_vals.values())) for q_vals in recent_values])
            avg_early = np.mean([np.mean(list(q_vals.values())) for q_vals in early_values])
            
            learning_progress = ((avg_recent - avg_early) / abs(avg_early)) * 100 if avg_early != 0 else 0
            
            # Model selection efficiency
            model_selections = q_data.get('model_selections', [])
            selection_diversity = len(set(model_selections)) / len(model_selections) if model_selections else 0
            
            return {
                "status": "healthy" if learning_progress > 5 else "warning",
                "learning_progress_pct": learning_progress,
                "selection_diversity": selection_diversity,
                "total_episodes": len(q_values_history),
                "convergence": "improving" if learning_progress > 0 else "stable"
            }
            
        except Exception as e:
            logger.error(f"Q-learning analysis error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """Analyze LLM model performance metrics"""
        logger.info("🤖 Analyzing model performance...")
        
        try:
            perf_path = self.outputs_dir / "model_performance.json"
            if not perf_path.exists():
                return {"status": "warning", "message": "No model performance data found"}
            
            with open(perf_path, 'r') as f:
                perf_data = json.load(f)
            
            # Analyze each model's performance
            model_analysis = {}
            best_model = None
            best_success_rate = 0
            
            for model, stats in perf_data.items():
                success_rate = stats.get('success_rate', 0)
                strategies_gen = stats.get('strategies_generated', 0)
                avg_return = stats.get('avg_return', 0)
                
                model_analysis[model] = {
                    "success_rate": success_rate,
                    "strategies_generated": strategies_gen,
                    "avg_return": avg_return,
                    "performance_rating": "excellent" if success_rate > 0.7 else "good" if success_rate > 0.5 else "needs_improvement"
                }
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_model = model
            
            return {
                "status": "healthy",
                "models": model_analysis,
                "best_model": best_model,
                "best_success_rate": best_success_rate,
                "total_models": len(model_analysis)
            }
            
        except Exception as e:
            logger.error(f"Model performance analysis error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _validate_agent_memory(self) -> Dict[str, Any]:
        """Validate agent memory growth and quality"""
        logger.info("🧠 Validating agent memory...")
        
        try:
            memory_path = self.outputs_dir / "agent_memory.pkl"
            if not memory_path.exists():
                return {"status": "warning", "message": "No agent memory file found"}
            
            with open(memory_path, 'rb') as f:
                memory_data = pickle.load(f)
            
            memory_size = len(memory_data) if isinstance(memory_data, (list, dict)) else 0
            
            # Check memory structure
            if isinstance(memory_data, list) and memory_size > 0:
                sample_entry = memory_data[0]
                has_proper_structure = all(key in sample_entry for key in ['state', 'action', 'reward']) if isinstance(sample_entry, dict) else False
            else:
                has_proper_structure = False
            
            return {
                "status": "healthy" if memory_size > 10 and has_proper_structure else "warning",
                "memory_size": memory_size,
                "proper_structure": has_proper_structure,
                "growth_indicator": "growing" if memory_size > 50 else "starting"
            }
            
        except Exception as e:
            logger.error(f"Agent memory validation error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_timeouts(self) -> Dict[str, Any]:
        """Analyze timeout patterns and frequency"""
        logger.info("⏱️ Analyzing timeout patterns...")
        
        try:
            timeout_log_path = self.logs_dir / "timeouts.log"
            if not timeout_log_path.exists():
                return {"status": "healthy", "message": "No timeout log found - system running smoothly"}
            
            # Read and analyze timeout log
            with open(timeout_log_path, 'r') as f:
                timeout_lines = f.readlines()
            
            total_timeouts = len(timeout_lines)
            
            # Analyze timeout patterns
            timeout_models = {}
            for line in timeout_lines:
                if "model:" in line:
                    model = line.split("model:")[1].split(",")[0].strip()
                    timeout_models[model] = timeout_models.get(model, 0) + 1
            
            most_problematic = max(timeout_models.items(), key=lambda x: x[1]) if timeout_models else None
            
            return {
                "status": "warning" if total_timeouts > 10 else "healthy",
                "total_timeouts": total_timeouts,
                "timeout_by_model": timeout_models,
                "most_problematic_model": most_problematic[0] if most_problematic else None,
                "recommendation": "Consider increasing timeout limits" if total_timeouts > 20 else "Timeout handling working well"
            }
            
        except Exception as e:
            logger.error(f"Timeout analysis error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _check_system_stability(self) -> Dict[str, Any]:
        """Check overall system stability indicators"""
        logger.info("🏥 Checking system stability...")
        
        try:
            # Check for recent successful runs
            demo_results_path = self.outputs_dir / "demo_trading_results.json"
            recent_runs = 0
            last_run_time = None
            
            if demo_results_path.exists():
                with open(demo_results_path, 'r') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        recent_runs = len([r for r in results[-10:] if r.get('status') == 'completed'])
                        if results:
                            last_run_time = results[-1].get('timestamp')
            
            # Check error logs
            error_count = 0
            if (self.logs_dir / "errors.log").exists():
                with open(self.logs_dir / "errors.log", 'r') as f:
                    error_count = len(f.readlines())
            
            stability_score = max(0, 100 - (error_count * 2) - (10 if recent_runs == 0 else 0))
            
            return {
                "status": "healthy" if stability_score > 80 else "warning" if stability_score > 50 else "critical",
                "stability_score": stability_score,
                "recent_successful_runs": recent_runs,
                "last_run_time": last_run_time,
                "total_errors": error_count,
                "recommendation": "System stable" if stability_score > 80 else "Monitor system closely"
            }
            
        except Exception as e:
            logger.error(f"System stability check error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _generate_report(self) -> str:
        """Generate comprehensive markdown report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"diagnostics_{timestamp}.md"
        
        # Generate visualizations
        self._create_performance_plots()
        
        # Create markdown report
        report_content = f"""# 🔍 Bit-Trade System Diagnostics Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 🎯 Executive Summary

"""
        
        # Overall health assessment
        health_scores = []
        for category, results in self.diagnostics_results.items():
            if results.get('status') == 'healthy':
                health_scores.append(100)
            elif results.get('status') == 'warning':
                health_scores.append(70)
            elif results.get('status') == 'error':
                health_scores.append(30)
            else:
                health_scores.append(50)
        
        overall_health = np.mean(health_scores) if health_scores else 50
        
        report_content += f"""
**System Health Score**: {overall_health:.1f}/100

**Status**: {'🟢 Healthy' if overall_health > 80 else '🟡 Needs Attention' if overall_health > 50 else '🔴 Critical'}

---

## 📊 Detailed Analysis

### 🧠 Q-Learning Performance
"""
        
        q_results = self.diagnostics_results.get('q_learning', {})
        report_content += f"""
- **Status**: {q_results.get('status', 'unknown').upper()}
- **Learning Progress**: {q_results.get('learning_progress_pct', 0):.1f}%
- **Selection Diversity**: {q_results.get('selection_diversity', 0):.2f}
- **Total Episodes**: {q_results.get('total_episodes', 0)}
- **Convergence**: {q_results.get('convergence', 'unknown')}

"""
        
        # Model Performance
        model_results = self.diagnostics_results.get('model_performance', {})
        report_content += f"""### 🤖 Model Performance

**Best Model**: {model_results.get('best_model', 'N/A')} (Success Rate: {model_results.get('best_success_rate', 0):.1%})

"""
        
        if 'models' in model_results:
            for model, stats in model_results['models'].items():
                report_content += f"""- **{model}**: {stats['success_rate']:.1%} success, {stats['strategies_generated']} strategies, {stats['avg_return']:.2f}% avg return
"""
        
        # Agent Memory
        memory_results = self.diagnostics_results.get('agent_memory', {})
        report_content += f"""
### 🧠 Agent Memory Status
- **Status**: {memory_results.get('status', 'unknown').upper()}
- **Memory Size**: {memory_results.get('memory_size', 0)} entries
- **Structure Valid**: {'✅' if memory_results.get('proper_structure') else '❌'}
- **Growth**: {memory_results.get('growth_indicator', 'unknown')}

"""
        
        # Timeout Analysis
        timeout_results = self.diagnostics_results.get('timeout_analysis', {})
        report_content += f"""### ⏱️ Timeout Analysis
- **Total Timeouts**: {timeout_results.get('total_timeouts', 0)}
- **Most Problematic Model**: {timeout_results.get('most_problematic_model', 'None')}
- **Recommendation**: {timeout_results.get('recommendation', 'N/A')}

"""
        
        # System Stability
        stability_results = self.diagnostics_results.get('system_stability', {})
        report_content += f"""### 🏥 System Stability
- **Stability Score**: {stability_results.get('stability_score', 0)}/100
- **Recent Successful Runs**: {stability_results.get('recent_successful_runs', 0)}
- **Total Errors**: {stability_results.get('total_errors', 0)}
- **Last Run**: {stability_results.get('last_run_time', 'N/A')}

---

## 🎯 Recommendations

"""
        
        # Generate recommendations
        recommendations = []
        
        if q_results.get('learning_progress_pct', 0) < 5:
            recommendations.append("🧠 Q-learning showing minimal progress - consider tuning hyperparameters")
        
        if model_results.get('best_success_rate', 0) < 0.5:
            recommendations.append("🤖 Model success rates low - review strategy generation prompts")
        
        if timeout_results.get('total_timeouts', 0) > 10:
            recommendations.append("⏱️ High timeout frequency - consider increasing timeout limits")
        
        if stability_results.get('stability_score', 0) < 70:
            recommendations.append("🏥 System stability concerns - review error logs")
        
        if not recommendations:
            recommendations.append("✅ System performing well - continue monitoring")
        
        for rec in recommendations:
            report_content += f"- {rec}\n"
        
        report_content += f"""
---

**🤖 Generated with Bit-Trade Diagnostics System**

**Report Location**: `{report_path}`
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def _create_performance_plots(self):
        """Create performance visualization plots"""
        try:
            # Create plots directory
            plots_dir = self.reports_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Q-learning performance plot
            self._plot_q_learning_progress(plots_dir)
            
            # Model performance comparison
            self._plot_model_comparison(plots_dir)
            
        except Exception as e:
            logger.error(f"Plot generation error: {e}")
    
    def _plot_q_learning_progress(self, plots_dir: Path):
        """Plot Q-learning convergence"""
        try:
            q_data_path = self.outputs_dir / "q_learning_data.json"
            if not q_data_path.exists():
                return
                
            with open(q_data_path, 'r') as f:
                q_data = json.load(f)
            
            q_values_history = q_data.get('q_values_history', [])
            if len(q_values_history) < 5:
                return
            
            # Calculate average Q-values over time
            avg_q_values = []
            for q_vals in q_values_history:
                if q_vals:
                    avg_q_values.append(np.mean(list(q_vals.values())))
                else:
                    avg_q_values.append(0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(avg_q_values, linewidth=2)
            plt.title('Q-Learning Convergence Over Time')
            plt.xlabel('Episode')
            plt.ylabel('Average Q-Value')
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / 'q_learning_progress.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Q-learning plot error: {e}")
    
    def _plot_model_comparison(self, plots_dir: Path):
        """Plot model performance comparison"""
        try:
            perf_path = self.outputs_dir / "model_performance.json"
            if not perf_path.exists():
                return
                
            with open(perf_path, 'r') as f:
                perf_data = json.load(f)
            
            models = list(perf_data.keys())
            success_rates = [perf_data[m].get('success_rate', 0) for m in models]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, success_rates)
            plt.title('Model Performance Comparison')
            plt.xlabel('Model')
            plt.ylabel('Success Rate')
            plt.xticks(rotation=45)
            
            # Color bars based on performance
            for i, bar in enumerate(bars):
                if success_rates[i] > 0.7:
                    bar.set_color('green')
                elif success_rates[i] > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Model comparison plot error: {e}")


def main():
    """Run diagnostics from command line"""
    diagnostics = BitTradeDiagnostics()
    results = diagnostics.run_full_diagnostics()
    
    # Print summary
    print("\n🔍 DIAGNOSTICS SUMMARY:")
    print("=" * 50)
    
    for category, result in results.items():
        status_icon = {
            'healthy': '🟢',
            'warning': '🟡', 
            'error': '🔴',
            'critical': '🔴'
        }.get(result.get('status'), '⚪')
        
        print(f"{status_icon} {category.upper()}: {result.get('status', 'unknown').upper()}")
    
    print("\n📊 Full report generated in reports/ directory")


if __name__ == "__main__":
    main()