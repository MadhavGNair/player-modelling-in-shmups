def display_pca(data, cluster_dict):
    filenames = list(data.keys())
    features = list(data[filenames[0]].keys())

    # convert to numpy array
    X = np.array([[data[fname][feat] for feat in features] for fname in filenames])

    # standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    principal_df = pd.DataFrame(principal_components, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principal_df, pd.DataFrame(filenames)], axis=1)
    finalDf.rename(columns={0: 'filename'}, inplace=True)

    finalDf['cluster'] = [cluster_dict[name] for name in filenames]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA')
    cluster_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
                      '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
    for c in set(finalDf['cluster']):
        cluster_data = finalDf.loc[finalDf['cluster'] == c]
        ax.scatter(cluster_data['principal component 1'], cluster_data['principal component 2'], c=cluster_colors[c],
                   label=str(c))

    return plt, finalDf