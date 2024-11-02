def analyze_clusters_raw(feature_data):
    feature_comparisons = {}

    for feature_name, cluster_values in feature_data.items():
        # Calculate statistics for each cluster
        cluster_stats = {}
        raw_values = []
        labels = []

        for cluster_id, values in cluster_values.items():
            cluster_stats[cluster_id] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'n': len(values)
            }
            raw_values.append(values)
            labels.extend([cluster_id] * len(values))

        # Perform one-way ANOVA using raw values
        if len(raw_values) > 1:
            f_statistic, p_value = stats.f_oneway(*raw_values)

            feature_comparisons[feature_name] = {
                'cluster_stats': cluster_stats,
                'anova': {
                    'f_statistic': f_statistic,
                    'p_value': p_value
                },
                'significant_difference': bool(p_value < 0.05)
            }

            if p_value < 0.05:
                all_values = np.concatenate(raw_values)
                tukey_result = statsmodels.stats.multicomp.pairwise_tukeyhsd(endog=all_values,
                                                                             groups=labels, alpha=0.05)

                write_tukey_hsd_to_file(feature_name, tukey_result)

    analysis_results = {
        'num_clusters': len(next(iter(feature_data.values()))),
        'feature_comparisons': feature_comparisons
    }

    return analysis_results

