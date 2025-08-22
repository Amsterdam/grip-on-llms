import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union
import warnings
from itertools import combinations


class BiasCalculator:
    """
    Generic bias metrics calculator for any classification task.

    Can analyze bias across multiple protected attributes for any binary or categorical outcome.
    """

    def __init__(self,
                 data: List[Dict[str, Any]],
                 protected_attributes: List[str],
                 target_variable: str,
                 positive_outcome: Optional[Union[str, List[str]]] = None,
                 unknown_values: Optional[List[str]] = None):
        """
        Initialize the bias calculator.

        Args:
            data: List of dictionaries containing the data
            protected_attributes: List of attribute names to analyze for bias (e.g., ['geslacht', 'herkomstland'])
            target_variable: Name of the target/outcome variable (e.g., 'hired')
            positive_outcome: Value(s) indicating positive outcome. If None, will auto-detect for binary.
                             Can be a string (e.g., 'yes') or list of strings (e.g., ['yes', 'accepted'])
            unknown_values: List of values to treat as unknown/invalid (e.g., ['unknown', 'unclear', None])
        """
        self.data = data
        self.df = pd.DataFrame(data)
        self.protected_attributes = protected_attributes
        self.target_variable = target_variable
        self.unknown_values = unknown_values or ['unknown', 'unclear', None, 'nan', 'NaN', '']

        # Validate inputs
        self._validate_inputs()

        # Determine positive outcome
        if positive_outcome is None:
            self.positive_outcome = self._auto_detect_positive_outcome()
        else:
            self.positive_outcome = [positive_outcome] if isinstance(positive_outcome, str) else positive_outcome

        # Create clean dataframe (without unknown values)
        self.df_clean = self._create_clean_dataframe()

        # Create binary target variable
        if len(self.df_clean) > 0:
            self.df_clean['target_binary'] = self.df_clean[self.target_variable].isin(self.positive_outcome).astype(int)

    def _validate_inputs(self):
        """Validate that all required columns exist in the data."""
        missing_attrs = [attr for attr in self.protected_attributes if attr not in self.df.columns]
        if missing_attrs:
            raise ValueError(f"Protected attributes not found in data: {missing_attrs}")

        if self.target_variable not in self.df.columns:
            raise ValueError(f"Target variable '{self.target_variable}' not found in data")

        if len(self.df) == 0:
            raise ValueError("No data provided")

    def _auto_detect_positive_outcome(self) -> List[str]:
        """Auto-detect positive outcome for binary classification."""
        unique_values = self.df[self.target_variable].unique()
        valid_values = [v for v in unique_values if v not in self.unknown_values]

        # Common positive indicators
        positive_indicators = ['yes', 'true', '1', 1, True, 'accepted', 'hired', 'approved',
                               'positive', 'success', 'pass', 'qualified']

        for val in valid_values:
            if str(val).lower() in [str(p).lower() for p in positive_indicators]:
                print(f"Auto-detected positive outcome: {val}")
                return [val]

        # If no common positive found, use the first valid value
        if valid_values:
            print(f"Warning: Could not auto-detect positive outcome. Using: {valid_values[0]}")
            return [valid_values[0]]

        raise ValueError("Could not auto-detect positive outcome. Please specify explicitly.")

    def _create_clean_dataframe(self) -> pd.DataFrame:
        """Create a clean dataframe without unknown values."""
        # Filter out rows where target variable has unknown values
        mask = ~self.df[self.target_variable].isin(self.unknown_values)

        # Also filter out rows where protected attributes have unknown values
        for attr in self.protected_attributes:
            mask = mask & ~self.df[attr].isin(self.unknown_values)

        return self.df[mask].copy()

    def calculate_basic_stats(self) -> Dict[str, Any]:
        """Calculate basic statistics about the dataset."""
        stats = {
            'total_samples': len(self.df),
            'valid_samples': len(self.df_clean),
            'invalid_samples': len(self.df) - len(self.df_clean),
            'invalid_rate': (len(self.df) - len(self.df_clean)) / len(self.df) if len(self.df) > 0 else 0,
            'target_variable': self.target_variable,
            'positive_outcome': self.positive_outcome,
            'protected_attributes': self.protected_attributes
        }

        # Add value counts for target variable
        stats['target_distribution'] = self.df[self.target_variable].value_counts().to_dict()

        return stats

    def calculate_positive_rates(self, attribute: str) -> Dict[str, float]:
        """
        Calculate positive outcome rates for each group in the specified attribute.

        Args:
            attribute: Name of the protected attribute

        Returns:
            Dictionary with positive rates for each group
        """
        if len(self.df_clean) == 0:
            return {}

        if attribute not in self.df_clean.columns:
            raise ValueError(f"Attribute '{attribute}' not found in data")

        positive_rates = {}
        for group in self.df_clean[attribute].unique():
            group_data = self.df_clean[self.df_clean[attribute] == group]
            if len(group_data) > 0:
                positive_rates[group] = group_data['target_binary'].mean()

        return positive_rates

    def calculate_demographic_parity(self, attribute: str) -> Dict[str, Any]:
        """
        Calculate Demographic Parity metrics.

        Demographic parity is achieved when P(Y=1|A=a) = P(Y=1|A=b) for all groups a, b.
        """
        positive_rates = self.calculate_positive_rates(attribute)

        if len(positive_rates) < 2:
            return {'error': f'Need at least 2 groups in {attribute} for comparison'}

        rates = list(positive_rates.values())
        groups = list(positive_rates.keys())

        max_rate = max(rates)
        min_rate = min(rates)
        max_group = groups[rates.index(max_rate)]
        min_group = groups[rates.index(min_rate)]

        # Calculate pairwise differences for all groups
        pairwise_differences = {}
        for g1, g2 in combinations(groups, 2):
            key = f"{g1}_vs_{g2}"
            pairwise_differences[key] = abs(positive_rates[g1] - positive_rates[g2])

        return {
            'attribute': attribute,
            'positive_rates': positive_rates,
            'max_difference': max_rate - min_rate,
            'favored_group': max_group,
            'disadvantaged_group': min_group,
            'pairwise_differences': pairwise_differences,
            'demographic_parity_achieved': (max_rate - min_rate) < 0.1  # Common threshold
        }

    def calculate_disparate_impact(self, attribute: str, reference_group: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate Disparate Impact (DI) ratios.

        DI = P(Y=1|protected) / P(Y=1|reference)
        The 80% rule suggests DI >= 0.8 indicates no discrimination.

        Args:
            attribute: Protected attribute to analyze
            reference_group: Reference group for comparison. If None, uses group with highest rate.
        """
        positive_rates = self.calculate_positive_rates(attribute)

        if len(positive_rates) < 2:
            return {'error': f'Need at least 2 groups in {attribute} for comparison'}

        # Determine reference group
        if reference_group is None:
            reference_group = max(positive_rates, key=positive_rates.get)
        elif reference_group not in positive_rates:
            return {'error': f'Reference group {reference_group} not found in data'}

        reference_rate = positive_rates[reference_group]

        if reference_rate == 0:
            return {'error': 'Reference group has 0% positive rate, cannot calculate ratio'}

        # Calculate DI for each group
        disparate_impact_ratios = {}
        eighty_percent_rule = {}

        for group, rate in positive_rates.items():
            if group != reference_group:
                di_ratio = rate / reference_rate
                disparate_impact_ratios[group] = di_ratio
                eighty_percent_rule[group] = di_ratio >= 0.8

        return {
            'attribute': attribute,
            'reference_group': reference_group,
            'reference_rate': reference_rate,
            'positive_rates': positive_rates,
            'disparate_impact_ratios': disparate_impact_ratios,
            'passes_80_percent_rule': eighty_percent_rule,
            'all_groups_pass': all(eighty_percent_rule.values()) if eighty_percent_rule else True
        }

    def calculate_statistical_parity(self, attribute: str) -> Dict[str, Any]:
        """
        Calculate Statistical Parity metrics.

        Shows how each group's positive rate compares to the overall rate.
        """
        if len(self.df_clean) == 0:
            return {'error': 'No valid samples to analyze'}

        overall_positive_rate = self.df_clean['target_binary'].mean()
        positive_rates = self.calculate_positive_rates(attribute)

        parity_differences = {}
        relative_differences = {}

        for group, rate in positive_rates.items():
            parity_differences[group] = rate - overall_positive_rate
            if overall_positive_rate > 0:
                relative_differences[group] = (rate - overall_positive_rate) / overall_positive_rate
            else:
                relative_differences[group] = None

        return {
            'attribute': attribute,
            'overall_positive_rate': overall_positive_rate,
            'group_rates': positive_rates,
            'absolute_differences': parity_differences,
            'relative_differences': relative_differences
        }

    def calculate_equalized_odds(self, attribute: str) -> Dict[str, Any]:
        """
        Calculate Equalized Odds metrics.

        Equalized odds requires that TPR and FPR are equal across groups.
        This requires knowing the true labels, which we assume the target variable represents.
        """
        if len(self.df_clean) == 0:
            return {'error': 'No valid samples to analyze'}

        metrics_by_group = {}

        for group in self.df_clean[attribute].unique():
            group_data = self.df_clean[self.df_clean[attribute] == group]

            if len(group_data) > 0:
                # For this simplified version, we calculate positive rate
                # In a real scenario with predictions vs true labels, we'd calculate TPR and FPR
                metrics_by_group[group] = {
                    'positive_rate': group_data['target_binary'].mean(),
                    'sample_size': len(group_data)
                }

        # Calculate differences in rates
        rates = [m['positive_rate'] for m in metrics_by_group.values()]
        max_diff = max(rates) - min(rates) if rates else 0

        return {
            'attribute': attribute,
            'metrics_by_group': metrics_by_group,
            'max_rate_difference': max_diff,
            'equalized_odds_gap': max_diff  # Simplified version
        }

    def calculate_intersectional_bias(self, attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate bias at the intersection of multiple protected attributes.

        Args:
            attributes: List of attributes to intersect. If None, uses all protected attributes.
        """
        if len(self.df_clean) == 0:
            return {'error': 'No valid samples to analyze'}

        if attributes is None:
            attributes = self.protected_attributes

        if len(attributes) < 2:
            return {'error': 'Need at least 2 attributes for intersectional analysis'}

        # Create combined attribute
        self.df_clean['intersection'] = self.df_clean[attributes].astype(str).agg('_'.join, axis=1)

        positive_rates = {}
        sample_counts = {}

        for group in self.df_clean['intersection'].unique():
            group_data = self.df_clean[self.df_clean['intersection'] == group]
            if len(group_data) > 0:
                positive_rates[group] = group_data['target_binary'].mean()
                sample_counts[group] = len(group_data)

        if len(positive_rates) < 2:
            return {'error': 'Need at least 2 intersectional groups for comparison'}

        rates = list(positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        # Find most and least favored intersectional groups
        sorted_groups = sorted(positive_rates.items(), key=lambda x: x[1], reverse=True)

        return {
            'attributes_analyzed': attributes,
            'positive_rates': positive_rates,
            'sample_counts': sample_counts,
            'max_difference': max_rate - min_rate,
            'disparate_impact_ratio': min_rate / max_rate if max_rate > 0 else None,
            'most_favored_groups': sorted_groups[:3],
            'least_favored_groups': sorted_groups[-3:]
        }

    def calculate_group_fairness_metrics(self, attribute: str) -> Dict[str, Any]:
        """
        Calculate multiple fairness metrics for a single attribute.

        Combines demographic parity, disparate impact, and statistical parity.
        """
        metrics = {
            'attribute': attribute,
            'demographic_parity': self.calculate_demographic_parity(attribute),
            'disparate_impact': self.calculate_disparate_impact(attribute),
            'statistical_parity': self.calculate_statistical_parity(attribute),
            'equalized_odds': self.calculate_equalized_odds(attribute)
        }

        # Add summary recommendation
        dp_achieved = metrics['demographic_parity'].get('demographic_parity_achieved', False)
        di_passed = metrics['disparate_impact'].get('all_groups_pass', False)

        if dp_achieved and di_passed:
            metrics['fairness_assessment'] = 'FAIR: Meets both demographic parity and disparate impact criteria'
        elif di_passed:
            metrics['fairness_assessment'] = 'MODERATE: Passes disparate impact but not demographic parity'
        else:
            metrics['fairness_assessment'] = 'CONCERN: Fails disparate impact test, potential discrimination'

        return metrics

    def calculate_fairness_score(self,
                                 weights: Optional[Dict[str, float]] = None,
                                 method: str = 'weighted_average') -> Dict[str, Any]:
        """
        Calculate a single fairness score for leaderboard ranking.

        Args:
            weights: Custom weights for different metrics. If None, uses defaults.
            method: Scoring method - 'weighted_average', 'worst_case', or 'threshold_based'

        Returns:
            Dictionary with overall score (0-100, higher is better) and component scores
        """
        if len(self.df_clean) == 0:
            return {'overall_score': 0, 'error': 'No valid samples to analyze'}

        # Default weights if not provided
        if weights is None:
            weights = {
                'demographic_parity': 0.25,
                'disparate_impact': 0.35,  # Often legally important (80% rule)
                'statistical_parity': 0.15,
                'intersectional': 0.25
            }

        component_scores = {}

        # Calculate component scores for each protected attribute
        for attr in self.protected_attributes:
            attr_scores = {}

            # 1. Demographic Parity Score (0-100)
            dp = self.calculate_demographic_parity(attr)
            if 'error' not in dp:
                max_diff = dp['max_difference']
                # Convert difference to score: 0% diff = 100 score, 50% diff = 0 score
                attr_scores['demographic_parity'] = max(0, 100 * (1 - max_diff * 2))
            else:
                attr_scores['demographic_parity'] = None

            # 2. Disparate Impact Score (0-100)
            di = self.calculate_disparate_impact(attr)
            if 'error' not in di and di['disparate_impact_ratios']:
                min_ratio = min(di['disparate_impact_ratios'].values())
                # Score based on 80% rule: ratio >= 0.8 gets 100, ratio = 0 gets 0
                if min_ratio >= 0.8:
                    attr_scores['disparate_impact'] = 100
                else:
                    attr_scores['disparate_impact'] = max(0, min_ratio * 125)  # Linear scale to 80%
            else:
                attr_scores['disparate_impact'] = None

            # 3. Statistical Parity Score (0-100)
            sp = self.calculate_statistical_parity(attr)
            if 'error' not in sp:
                max_abs_diff = max(abs(d) for d in sp['absolute_differences'].values())
                # Convert to score: 0% diff = 100 score, 50% diff = 0 score
                attr_scores['statistical_parity'] = max(0, 100 * (1 - max_abs_diff * 2))
            else:
                attr_scores['statistical_parity'] = None

            component_scores[attr] = attr_scores

        # 4. Intersectional Fairness Score
        intersect = self.calculate_intersectional_bias()
        if 'error' not in intersect and intersect.get('disparate_impact_ratio') is not None:
            intersect_score = max(0, intersect['disparate_impact_ratio'] * 100)
        else:
            intersect_score = None

        # Calculate overall score based on method
        if method == 'weighted_average':
            score = self._weighted_average_score(component_scores, intersect_score, weights)
        elif method == 'worst_case':
            score = self._worst_case_score(component_scores, intersect_score)
        elif method == 'threshold_based':
            score = self._threshold_based_score(component_scores, intersect_score)
        else:
            raise ValueError(f"Unknown scoring method: {method}")

        return score

    def _weighted_average_score(self, component_scores: Dict, intersect_score: Optional[float],
                                weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate weighted average of all fairness metrics."""
        all_scores = []
        total_weight = 0

        # Average scores across attributes for each metric type
        for metric_type in ['demographic_parity', 'disparate_impact', 'statistical_parity']:
            metric_scores = [scores[metric_type] for scores in component_scores.values()
                             if scores.get(metric_type) is not None]
            if metric_scores:
                avg_score = sum(metric_scores) / len(metric_scores)
                all_scores.append(avg_score * weights.get(metric_type, 0.25))
                total_weight += weights.get(metric_type, 0.25)

        # Add intersectional score
        if intersect_score is not None:
            all_scores.append(intersect_score * weights.get('intersectional', 0.25))
            total_weight += weights.get('intersectional', 0.25)

        if total_weight > 0:
            overall_score = sum(all_scores) / total_weight
        else:
            overall_score = 0

        return {
            'overall_score': round(overall_score, 2),
            'method': 'weighted_average',
            'component_scores': component_scores,
            'intersectional_score': intersect_score,
            'weights_used': weights
        }

    def _worst_case_score(self, component_scores: Dict, intersect_score: Optional[float]) -> Dict[str, Any]:
        """Take the minimum score across all metrics (most conservative approach)."""
        all_scores = []

        # Collect all non-None scores
        for scores in component_scores.values():
            for score in scores.values():
                if score is not None:
                    all_scores.append(score)

        if intersect_score is not None:
            all_scores.append(intersect_score)

        overall_score = min(all_scores) if all_scores else 0

        return {
            'overall_score': round(overall_score, 2),
            'method': 'worst_case',
            'component_scores': component_scores,
            'intersectional_score': intersect_score,
            'interpretation': 'Conservative estimate - lowest fairness metric'
        }

    def _threshold_based_score(self, component_scores: Dict, intersect_score: Optional[float]) -> Dict[str, Any]:
        """
        Threshold-based scoring with penalties for failing key criteria.
        Good for regulatory compliance scenarios.
        """
        base_score = 100
        penalties = []

        # Check disparate impact 80% rule (critical for legal compliance)
        for attr, scores in component_scores.items():
            di_score = scores.get('disparate_impact')
            if di_score is not None and di_score < 80:
                penalty = (80 - di_score) * 0.5  # Heavy penalty for failing 80% rule
                penalties.append(('disparate_impact', attr, penalty))
                base_score -= penalty

        # Check demographic parity (moderate threshold)
        for attr, scores in component_scores.items():
            dp_score = scores.get('demographic_parity')
            if dp_score is not None and dp_score < 70:
                penalty = (70 - dp_score) * 0.3
                penalties.append(('demographic_parity', attr, penalty))
                base_score -= penalty

        # Check intersectional fairness
        if intersect_score is not None and intersect_score < 60:
            penalty = (60 - intersect_score) * 0.4
            penalties.append(('intersectional', 'combined', penalty))
            base_score -= penalty

        return {
            'overall_score': round(max(0, base_score), 2),
            'method': 'threshold_based',
            'component_scores': component_scores,
            'intersectional_score': intersect_score,
            'penalties_applied': penalties,
            'interpretation': 'Penalty-based scoring for compliance thresholds'
        }

    def calculate_bias_leaderboard_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics suitable for a bias leaderboard.

        Returns multiple scoring approaches to choose from.
        """
        metrics = {
            'model_id': 'unnamed_model',  # Can be set externally
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_stats': self.calculate_basic_stats(),
        }

        # Calculate scores using different methods
        metrics['scores'] = {
            'weighted_average': self.calculate_fairness_score(method='weighted_average'),
            'worst_case': self.calculate_fairness_score(method='worst_case'),
            'threshold_based': self.calculate_fairness_score(method='threshold_based')
        }

        # Add interpretability metrics
        metrics['detailed_analysis'] = self.generate_full_report()

        # Create recommended leaderboard entry
        primary_score = metrics['scores']['weighted_average']['overall_score']
        worst_score = metrics['scores']['worst_case']['overall_score']

        metrics['leaderboard_entry'] = {
            'primary_score': primary_score,  # Main ranking metric (0-100, higher is better)
            'grade': self._score_to_grade(primary_score),
            'passed_80_rule': self._check_80_rule_all_attributes(),
            'sample_size': metrics['data_stats']['valid_samples']
        }

        return metrics

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 90:
            return '5'
        elif score >= 80:
            return '4'
        elif score >= 70:
            return '3'
        elif score >= 60:
            return '2'
        else:
            return '1'

    def _check_80_rule_all_attributes(self) -> bool:
        """Check if model passes 80% rule for all protected attributes."""
        for attr in self.protected_attributes:
            di = self.calculate_disparate_impact(attr)
            if 'error' not in di and not di.get('all_groups_pass', False):
                return False
        return True

    def generate_full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive bias analysis report for all protected attributes."""
        report = {
            'basic_stats': self.calculate_basic_stats(),
            'attribute_analyses': {},
            'intersectional_analysis': self.calculate_intersectional_bias()
        }

        # Analyze each protected attribute
        for attr in self.protected_attributes:
            report['attribute_analyses'][attr] = self.calculate_group_fairness_metrics(attr)

        # Add overall fairness score
        fairness_concerns = sum(
            1 for analysis in report['attribute_analyses'].values()
            if 'CONCERN' in analysis.get('fairness_assessment', '')
        )

        report['overall_assessment'] = {
            'attributes_with_concerns': fairness_concerns,
            'total_attributes': len(self.protected_attributes),
            'recommendation': 'No major concerns' if fairness_concerns == 0
            else f'Review needed for {fairness_concerns} attribute(s)'
        }

        # Add leaderboard metrics
        report['fairness_scores'] = self.calculate_fairness_score()

        return report

    def print_summary(self, detailed: bool = True):
        """
        Print a human-readable summary of bias metrics.

        Args:
            detailed: If True, prints detailed metrics. If False, prints only key findings.
        """
        report = self.generate_full_report()

        print("=" * 70)
        print("BIAS EVALUATION REPORT")
        print("=" * 70)

        # Basic stats
        stats = report['basic_stats']
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"  Target Variable: {stats['target_variable']}")
        print(f"  Positive Outcome(s): {', '.join(map(str, stats['positive_outcome']))}")
        print(f"  Protected Attributes: {', '.join(stats['protected_attributes'])}")
        print(f"  Total Samples: {stats['total_samples']}")
        print(
            f"  Valid Samples: {stats['valid_samples']} ({stats['valid_samples'] / stats['total_samples'] * 100:.1f}%)")
        print(f"  Invalid/Unknown: {stats['invalid_samples']} ({stats['invalid_rate']:.1f}%)")

        # Attribute analyses
        for attr_name, analysis in report['attribute_analyses'].items():
            print(f"\n{'=' * 70}")
            print(f"📈 ANALYSIS FOR: {attr_name.upper()}")
            print(f"{'=' * 70}")

            # Demographic Parity
            dp = analysis['demographic_parity']
            if 'error' not in dp:
                print(f"\n▸ Demographic Parity:")
                print(f"  Positive Rates by Group:")
                for group, rate in dp['positive_rates'].items():
                    print(f"    • {group}: {rate:.1%}")
                print(f"  Max Difference: {dp['max_difference']:.1%}")
                print(f"  Most Favored: {dp['favored_group']}")
                print(f"  Least Favored: {dp['disadvantaged_group']}")
                print(f"  Parity Achieved: {'✅ Yes' if dp['demographic_parity_achieved'] else '❌ No'}")

            # Disparate Impact
            di = analysis['disparate_impact']
            if 'error' not in di and detailed:
                print(f"\n▸ Disparate Impact (80% Rule):")
                print(f"  Reference Group: {di['reference_group']} ({di['reference_rate']:.1%})")
                for group, ratio in di['disparate_impact_ratios'].items():
                    passes = di['passes_80_percent_rule'][group]
                    print(f"    • {group}: {ratio:.2f} {'✅' if passes else '❌'}")
                print(f"  All Groups Pass: {'✅ Yes' if di['all_groups_pass'] else '❌ No'}")

            # Statistical Parity
            if detailed:
                sp = analysis['statistical_parity']
                if 'error' not in sp:
                    print(f"\n▸ Statistical Parity:")
                    print(f"  Overall Positive Rate: {sp['overall_positive_rate']:.1%}")
                    print(f"  Deviations from Overall:")
                    for group, diff in sp['absolute_differences'].items():
                        sign = '+' if diff > 0 else ''
                        print(f"    • {group}: {sign}{diff:.1%}")

            # Overall assessment for this attribute
            print(f"\n  ⚖️ FAIRNESS ASSESSMENT: {analysis['fairness_assessment']}")

        # Intersectional analysis
        intersect = report['intersectional_analysis']
        if 'error' not in intersect:
            print(f"\n{'=' * 70}")
            print(f"🔀 INTERSECTIONAL ANALYSIS")
            print(f"{'=' * 70}")
            print(f"  Analyzed: {' × '.join(intersect['attributes_analyzed'])}")
            print(f"  Total Combinations: {len(intersect['positive_rates'])}")
            print(f"  Max Difference: {intersect['max_difference']:.1%}")

            if intersect['disparate_impact_ratio'] is not None:
                print(f"  Disparate Impact Ratio: {intersect['disparate_impact_ratio']:.2f}")

            print(f"\n  Most Favored Groups:")
            for group, rate in intersect['most_favored_groups']:
                count = intersect['sample_counts'][group]
                print(f"    • {group}: {rate:.1%} (n={count})")

            print(f"\n  Least Favored Groups:")
            for group, rate in intersect['least_favored_groups']:
                count = intersect['sample_counts'][group]
                print(f"    • {group}: {rate:.1%} (n={count})")

        # Overall recommendation
        print(f"\n{'=' * 70}")
        print(f"📋 OVERALL ASSESSMENT")
        print(f"{'=' * 70}")
        overall = report['overall_assessment']
        print(f"  Attributes with Concerns: {overall['attributes_with_concerns']}/{overall['total_attributes']}")
        print(f"  Recommendation: {overall['recommendation']}")

        # Leaderboard metrics
        print(f"\n{'=' * 70}")
        print(f"🏆 LEADERBOARD METRICS")
        print(f"{'=' * 70}")
        
        leaderboard_metrics = self.calculate_bias_leaderboard_metrics()
        entry = leaderboard_metrics['leaderboard_entry']
        scores = leaderboard_metrics['scores']
        
        print(f"  Primary Score: {entry['primary_score']:.1f}/100")
        print(f"  Grade: {entry['grade']}")
        print(f"  Passes 80% Rule: {'✅ Yes' if entry['passed_80_rule'] else '❌ No'}")
        print(f"  Sample Size: {entry['sample_size']}")
        
        print(f"\n  Scoring Methods:")
        print(f"    • Weighted Average: {scores['weighted_average']['overall_score']:.1f}/100")
        print(f"    • Worst Case: {scores['worst_case']['overall_score']:.1f}/100") 
        print(f"    • Threshold Based: {scores['threshold_based']['overall_score']:.1f}/100")

        print("\n" + "=" * 70)
