import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import logging
from anndata import AnnData

logger = logging.getLogger('spatialflow.analysis.ligrec')

def run_ligrec_analysis(adata, cluster_key=None, n_perms=1000, use_raw=False, 
                      database='CellPhoneDB', species='human', threshold=None, **kwargs):
    """
    Run ligand-receptor interaction analysis
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str, optional
        Key in adata.obs for cluster assignments
    n_perms : int, optional
        Number of permutations for significance testing
    use_raw : bool, optional
        Whether to use raw data for analysis
    database : str, optional
        Database for ligand-receptor pairs. Options: 'CellPhoneDB', 'CellChat', 'custom'
    species : str, optional
        Species for ligand-receptor database. Options: 'human', 'mouse'
    threshold : float, optional
        P-value threshold for significance
    **kwargs
        Additional parameters for squidpy.gr.ligrec
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added ligand-receptor results
    """
    logger.info(f"Running ligand-receptor analysis using {database} database")
    if cluster_key is None:
        cluster_keys = [key for key in adata.obs.columns if 
                      any(key.startswith(prefix) for prefix in 
                         ['leiden', 'louvain', 'kmeans', 'spatial_domains', 'spatial_clusters'])]
        
        if not cluster_keys:
            logger.error("No cluster assignments found in AnnData object")
            raise ValueError("No cluster assignments found in AnnData object")
        
        cluster_key = cluster_keys[0]
        logger.info(f"Using {cluster_key} for ligand-receptor analysis")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    lr_pairs = _get_lr_database(database, species)
    valid_genes = adata.var_names.tolist()
    filtered_pairs = []
    
    for ligand, receptor in lr_pairs:
        if ligand in valid_genes and receptor in valid_genes:
            filtered_pairs.append((ligand, receptor))
    
    if not filtered_pairs:
        logger.error("No valid ligand-receptor pairs found in the dataset")
        raise ValueError("No valid ligand-receptor pairs found in the dataset")
    
    logger.info(f"Found {len(filtered_pairs)} valid ligand-receptor pairs")
    try:
        import pandas as pd
        lr_df = pd.DataFrame([
            {'source': ligand, 'target': receptor, 'interaction': f"{ligand}_{receptor}"}
            for ligand, receptor in filtered_pairs
        ])
        sq.gr.ligrec(
            adata,
            cluster_key=cluster_key,
            n_perms=n_perms,
            use_raw=use_raw,
            interactions=lr_df,  # Use DataFrame with required columns
            **kwargs
        )
        if threshold is not None and f"{cluster_key}_ligrec" in adata.uns:
            logger.info(f"Applying significance threshold of {threshold}")
            try:
                _filter_significant_interactions(adata, cluster_key, threshold)
            except Exception as e:
                logger.warning(f"Error applying threshold: {str(e)}")
        if f"{cluster_key}_ligrec" in adata.uns:
            logger.info("Ligand-receptor analysis completed successfully")
            if 'means' in adata.uns[f"{cluster_key}_ligrec"]:
                means = adata.uns[f"{cluster_key}_ligrec"]['means']
                count = 0
                for _, df in means.items():
                    if hasattr(df, 'size'):
                        count += df.size
                logger.info(f"Found approximately {count} potential interactions")
        
    except Exception as e:
        logger.error(f"Error running ligand-receptor analysis: {str(e)}")
        raise
    
    return adata

def _log_ligrec_summary(adata, cluster_key):
    """
    Log summary of ligand-receptor analysis results
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str
        Key in adata.obs for cluster assignments
    
    Returns
    -------
    None
    """
    uns_key = f"{cluster_key}_ligrec"
    
    if uns_key not in adata.uns:
        logger.warning(f"Ligand-receptor results not found in adata.uns['{uns_key}']")
        return
    clusters = adata.obs[cluster_key].cat.categories.tolist()
    means = adata.uns[uns_key]['means']
    n_total = 0
    n_pairs = 0
    
    for source_target, mean_df in means.items():
        try:
            if isinstance(source_target, tuple) and len(source_target) == 2:
                source_idx, target_idx = source_target
                source_cluster = clusters[source_idx] if isinstance(source_idx, int) else str(source_idx)
                target_cluster = clusters[target_idx] if isinstance(target_idx, int) else str(target_idx)
            else:
                source_cluster = str(source_target[0]) if hasattr(source_target, '__getitem__') else "unknown"
                target_cluster = str(source_target[1]) if hasattr(source_target, '__getitem__') else "unknown"
            non_zero = mean_df.notna().sum().sum()
            n_total += non_zero
            n_pairs = max(n_pairs, mean_df.shape[0])
            
            if non_zero > 0:
                logger.info(f"Cluster {source_cluster} -> {target_cluster}: {non_zero} interactions")
                
        except Exception as e:
            logger.warning(f"Error processing interaction: {str(e)}")
            continue
    
    logger.info(f"Total interactions: {n_total}")
    logger.info(f"Total ligand-receptor pairs: {n_pairs}")

def _get_lr_database(database='CellPhoneDB', species='human'):
    """
    Get ligand-receptor pairs from a database
    
    Parameters
    ----------
    database : str, optional
        Database for ligand-receptor pairs. Options: 'CellPhoneDB', 'CellChat', 'custom'
    species : str, optional
        Species for ligand-receptor database. Options: 'human', 'mouse'
    
    Returns
    -------
    list
        List of (ligand, receptor) tuples
    """
    cellphonedb_human = [
        ('TGFB1', 'TGFBR1'), ('TGFB1', 'TGFBR2'),
        ('IL1A', 'IL1R1'), ('IL1B', 'IL1R1'),
        ('CSF1', 'CSF1R'), ('CSF2', 'CSF2RA'),
        ('CXCL12', 'CXCR4'), ('CXCL13', 'CXCR5'),
        ('CCL19', 'CCR7'), ('CCL21', 'CCR7'),
        ('CD40LG', 'CD40'), ('CD28', 'CD80'),
        ('CD28', 'CD86'), ('CTLA4', 'CD80'),
        ('CTLA4', 'CD86'), ('PDCD1', 'CD274'),
        ('TNF', 'TNFRSF1A'), ('TNF', 'TNFRSF1B'),
        ('IFNG', 'IFNGR1'), ('IFNG', 'IFNGR2'),
        ('IL2', 'IL2RA'), ('IL2', 'IL2RB'),
        ('IL4', 'IL4R'), ('IL6', 'IL6R'),
        ('IL7', 'IL7R'), ('IL10', 'IL10RA'),
        ('IL15', 'IL15RA'), ('IL18', 'IL18R1'),
        ('VEGFA', 'FLT1'), ('VEGFA', 'KDR'),
        ('JAG1', 'NOTCH1'), ('JAG2', 'NOTCH1'),
        ('DLL1', 'NOTCH1'), ('DLL4', 'NOTCH1'),
        ('ICAM1', 'ITGAL'), ('ICAM1', 'ITGB2'),
        ('VCAM1', 'ITGA4'), ('VCAM1', 'ITGB1'),
        ('CDH1', 'ITGAE'), ('SELE', 'SELPLG'),
        ('SELP', 'SELPLG'), ('CD40LG', 'CD40'),
        ('CD70', 'CD27'), ('TNFSF13B', 'TNFRSF13B'),
        ('TNFSF13B', 'TNFRSF17'), ('FAS', 'FASLG')
    ]
    cellphonedb_mouse = [
        ('Tgfb1', 'Tgfbr1'), ('Tgfb1', 'Tgfbr2'),
        ('Il1a', 'Il1r1'), ('Il1b', 'Il1r1'),
        ('Csf1', 'Csf1r'), ('Csf2', 'Csf2ra'),
        ('Cxcl12', 'Cxcr4'), ('Cxcl13', 'Cxcr5'),
        ('Ccl19', 'Ccr7'), ('Ccl21', 'Ccr7'),
        ('Cd40lg', 'Cd40'), ('Cd28', 'Cd80'),
        ('Cd28', 'Cd86'), ('Ctla4', 'Cd80'),
        ('Ctla4', 'Cd86'), ('Pdcd1', 'Cd274'),
        ('Tnf', 'Tnfrsf1a'), ('Tnf', 'Tnfrsf1b'),
        ('Ifng', 'Ifngr1'), ('Ifng', 'Ifngr2'),
        ('Il2', 'Il2ra'), ('Il2', 'Il2rb'),
        ('Il4', 'Il4r'), ('Il6', 'Il6r'),
        ('Il7', 'Il7r'), ('Il10', 'Il10ra'),
        ('Il15', 'Il15ra'), ('Il18', 'Il18r1'),
        ('Vegfa', 'Flt1'), ('Vegfa', 'Kdr'),
        ('Jag1', 'Notch1'), ('Jag2', 'Notch1'),
        ('Dll1', 'Notch1'), ('Dll4', 'Notch1'),
        ('Icam1', 'Itgal'), ('Icam1', 'Itgb2'),
        ('Vcam1', 'Itga4'), ('Vcam1', 'Itgb1'),
        ('Cdh1', 'Itgae'), ('Sele', 'Selplg'),
        ('Selp', 'Selplg'), ('Cd40lg', 'Cd40'),
        ('Cd70', 'Cd27'), ('Tnfsf13b', 'Tnfrsf13b'),
        ('Tnfsf13b', 'Tnfrsf17'), ('Fas', 'Faslg')
    ]
    cellchat_human = [
        ('MICA', 'KLRK1'), ('MICB', 'KLRK1'),
        ('ULBP1', 'KLRK1'), ('ULBP2', 'KLRK1'),
        ('ULBP3', 'KLRK1'), ('RAET1E', 'KLRK1'),
        ('RAET1G', 'KLRK1'), ('RAET1L', 'KLRK1'),
        ('WNT1', 'FZD1'), ('WNT1', 'FZD2'),
        ('WNT1', 'FZD3'), ('WNT1', 'FZD4'),
        ('WNT1', 'FZD5'), ('WNT1', 'FZD6'),
        ('WNT1', 'FZD7'), ('WNT1', 'FZD8'),
        ('WNT1', 'FZD9'), ('WNT1', 'FZD10'),
        ('WNT2', 'FZD1'), ('WNT2', 'FZD2'),
        ('WNT2', 'FZD3'), ('WNT2', 'FZD4'),
        ('WNT2', 'FZD5'), ('WNT2', 'FZD6'),
        ('WNT2', 'FZD7'), ('WNT2', 'FZD8'),
        ('WNT2', 'FZD9'), ('WNT2', 'FZD10'),
        ('EFNA1', 'EPHA1'), ('EFNA1', 'EPHA2'),
        ('EFNA1', 'EPHA3'), ('EFNA1', 'EPHA4'),
        ('EFNA1', 'EPHA5'), ('EFNA1', 'EPHA6'),
        ('EFNA1', 'EPHA7'), ('EFNA1', 'EPHA8'),
        ('EFNA1', 'EPHA10'), ('EFNA2', 'EPHA1'),
        ('NECTIN1', 'CD226'), ('NECTIN2', 'CD226'),
        ('NECTIN3', 'CD226'), ('NECTIN4', 'CD226')
    ]
    cellchat_mouse = [
        ('H60a', 'Klrk1'), ('H60b', 'Klrk1'),
        ('H60c', 'Klrk1'), ('Rae1', 'Klrk1'),
        ('Wnt1', 'Fzd1'), ('Wnt1', 'Fzd2'),
        ('Wnt1', 'Fzd3'), ('Wnt1', 'Fzd4'),
        ('Wnt1', 'Fzd5'), ('Wnt1', 'Fzd6'),
        ('Wnt1', 'Fzd7'), ('Wnt1', 'Fzd8'),
        ('Wnt1', 'Fzd9'), ('Wnt1', 'Fzd10'),
        ('Wnt2', 'Fzd1'), ('Wnt2', 'Fzd2'),
        ('Wnt2', 'Fzd3'), ('Wnt2', 'Fzd4'),
        ('Wnt2', 'Fzd5'), ('Wnt2', 'Fzd6'),
        ('Wnt2', 'Fzd7'), ('Wnt2', 'Fzd8'),
        ('Wnt2', 'Fzd9'), ('Wnt2', 'Fzd10'),
        ('Efna1', 'Epha1'), ('Efna1', 'Epha2'),
        ('Efna1', 'Epha3'), ('Efna1', 'Epha4'),
        ('Efna1', 'Epha5'), ('Efna1', 'Epha6'),
        ('Efna1', 'Epha7'), ('Efna1', 'Epha8'),
        ('Efna1', 'Epha10'), ('Efna2', 'Epha1'),
        ('Nectin1', 'Cd226'), ('Nectin2', 'Cd226'),
        ('Nectin3', 'Cd226'), ('Nectin4', 'Cd226')
    ]
    
    if database == 'CellPhoneDB':
        if species == 'human':
            return cellphonedb_human
        elif species == 'mouse':
            return cellphonedb_mouse
    elif database == 'CellChat':
        if species == 'human':
            return cellchat_human
        elif species == 'mouse':
            return cellchat_mouse
    elif database == 'custom':
        logger.warning("Custom database not provided, using CellPhoneDB human database")
        return cellphonedb_human
    
    logger.warning(f"Unknown database {database} or species {species}, using CellPhoneDB human database")
    return cellphonedb_human

def _filter_significant_interactions(adata, cluster_key, threshold=0.05):
    """
    Filter ligand-receptor interactions by significance
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str
        Key in adata.obs for cluster assignments
    threshold : float, optional
        P-value threshold for significance
    
    Returns
    -------
    None
        Modifies adata.uns in place
    """
    uns_key = f"{cluster_key}_ligrec"
    
    if uns_key not in adata.uns:
        logger.error(f"Ligand-receptor results not found in adata.uns['{uns_key}']")
        return
    pval_key = None
    if 'pvalues' in adata.uns[uns_key]:
        pval_key = 'pvalues'
    elif 'pvals' in adata.uns[uns_key]:
        pval_key = 'pvals'
    
    if pval_key is None:
        logger.warning("P-value data not found in ligand-receptor results")
        return
    means = adata.uns[uns_key]['means']
    pvals = adata.uns[uns_key][pval_key]
    significant = {}
    for source_target, pval_df in pvals.items():
        significant[source_target] = pval_df < threshold
    filtered_means = {}
    for source_target, mean_df in means.items():
        filtered = mean_df.copy()
        mask = significant.get(source_target, None)
        if mask is not None:
            filtered = filtered.where(mask)
        filtered_means[source_target] = filtered
    adata.uns[f"{cluster_key}_ligrec_significant"] = {
        'means': filtered_means,
        'pvalues': pvals,
        'threshold': threshold
    }
    n_significant = 0
    for _, df in filtered_means.items():
        n_significant += df.count().sum()
    
    logger.info(f"Found {n_significant} significant interactions (p < {threshold})")

def _log_ligrec_summary(adata, cluster_key):
    """
    Log summary of ligand-receptor analysis results
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str
        Key in adata.obs for cluster assignments
    
    Returns
    -------
    None
    """
    uns_key = f"{cluster_key}_ligrec"
    
    if uns_key not in adata.uns:
        logger.warning(f"Ligand-receptor results not found in adata.uns['{uns_key}']")
        return
    clusters = adata.obs[cluster_key].cat.categories.tolist()
    means = adata.uns[uns_key]['means']
    n_total = 0
    n_pairs = 0
    
    for source_target, mean_df in means.items():
        source, target = source_target
        source_cluster = clusters[source]
        target_cluster = clusters[target]
        
        non_zero = mean_df.notna().sum().sum()
        n_total += non_zero
        n_pairs = max(n_pairs, mean_df.shape[0])
        
        if non_zero > 0:
            logger.info(f"Cluster {source_cluster} -> {target_cluster}: {non_zero} interactions")
    
    logger.info(f"Total interactions: {n_total}")
    logger.info(f"Total ligand-receptor pairs: {n_pairs}")

def compute_interaction_scores(adata, source_groups, target_groups, lr_pairs=None, 
                            method='product', cluster_key=None):
    """
    Compute custom interaction scores between cell clusters
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    source_groups : list
        List of source groups for interactions
    target_groups : list
        List of target groups for interactions
    lr_pairs : list, optional
        List of (ligand, receptor) tuples. If None, uses built-in database.
    method : str, optional
        Method for computing interaction scores. Options: 'product', 'min', 'max', 'sum'
    cluster_key : str, optional
        Key in adata.obs for cluster assignments
    
    Returns
    -------
    pd.DataFrame
        DataFrame with interaction scores
    """
    logger.info(f"Computing interaction scores using {method} method")
    if cluster_key is None:
        cluster_keys = [key for key in adata.obs.columns if 
                      any(key.startswith(prefix) for prefix in 
                         ['leiden', 'louvain', 'kmeans', 'spatial_domains'])]
        
        if not cluster_keys:
            logger.error("No cluster assignments found in AnnData object")
            raise ValueError("No cluster assignments found in AnnData object")
        
        cluster_key = cluster_keys[0]
        logger.info(f"Using {cluster_key} for interaction scores")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    if lr_pairs is None:
        lr_pairs = _get_lr_database()
    valid_genes = adata.var_names.tolist()
    filtered_pairs = []
    
    for ligand, receptor in lr_pairs:
        if ligand in valid_genes and receptor in valid_genes:
            filtered_pairs.append((ligand, receptor))
    
    if not filtered_pairs:
        logger.error("No valid ligand-receptor pairs found in the dataset")
        raise ValueError("No valid ligand-receptor pairs found in the dataset")
    
    logger.info(f"Using {len(filtered_pairs)} valid ligand-receptor pairs")
    cluster_categories = adata.obs[cluster_key].cat.categories.tolist()
    
    source_groups = [group for group in source_groups if group in cluster_categories]
    target_groups = [group for group in target_groups if group in cluster_categories]
    
    if not source_groups or not target_groups:
        logger.error("No valid source or target groups found")
        raise ValueError("No valid source or target groups found")
    cluster_expr = {}
    
    for cluster in cluster_categories:
        mask = adata.obs[cluster_key] == cluster
        
        if sum(mask) == 0:
            continue
        if hasattr(adata.X, 'toarray'):
            expr = adata.X[mask].toarray().mean(axis=0)
        else:
            expr = adata.X[mask].mean(axis=0)
        
        cluster_expr[cluster] = expr
    interaction_scores = {}
    
    for ligand, receptor in filtered_pairs:
        ligand_idx = adata.var_names.get_loc(ligand)
        receptor_idx = adata.var_names.get_loc(receptor)
        
        for source in source_groups:
            for target in target_groups:
                if source == target:
                    continue
                ligand_expr = cluster_expr[source][ligand_idx]
                receptor_expr = cluster_expr[target][receptor_idx]
                if method == 'product':
                    score = ligand_expr * receptor_expr
                elif method == 'min':
                    score = min(ligand_expr, receptor_expr)
                elif method == 'max':
                    score = max(ligand_expr, receptor_expr)
                elif method == 'sum':
                    score = ligand_expr + receptor_expr
                else:
                    logger.warning(f"Unknown method {method}, using 'product'")
                    score = ligand_expr * receptor_expr
                key = (source, target, f"{ligand}-{receptor}")
                interaction_scores[key] = score
    rows = []
    
    for (source, target, pair), score in interaction_scores.items():
        rows.append({
            'source': source,
            'target': target,
            'pair': pair,
            'score': score
        })
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values('score', ascending=False)
        df[['ligand', 'receptor']] = df['pair'].str.split('-', expand=True)
        
        return df
    else:
        return pd.DataFrame(columns=['source', 'target', 'pair', 'score', 'ligand', 'receptor'])

def calculate_pathway_specific_interactions(adata, pathway_genes, cluster_key=None, 
                                         database='CellPhoneDB', species='human'):
    """
    Calculate interactions specific to a biological pathway
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    pathway_genes : list or dict
        List of genes in the pathway or dict mapping pathway names to gene lists
    cluster_key : str, optional
        Key in adata.obs for cluster assignments
    database : str, optional
        Database for ligand-receptor pairs. Options: 'CellPhoneDB', 'CellChat', 'custom'
    species : str, optional
        Species for ligand-receptor database. Options: 'human', 'mouse'
    
    Returns
    -------
    dict
        Dictionary mapping pathway names to interaction DataFrames
    """
    logger.info("Calculating pathway-specific interactions")
    if isinstance(pathway_genes, dict):
        pathways = pathway_genes
    else:
        pathways = {'Pathway': pathway_genes}
    lr_pairs = _get_lr_database(database, species)
    if cluster_key is None:
        cluster_keys = [key for key in adata.obs.columns if 
                      any(key.startswith(prefix) for prefix in 
                         ['leiden', 'louvain', 'kmeans', 'spatial_domains'])]
        
        if not cluster_keys:
            logger.error("No cluster assignments found in AnnData object")
            raise ValueError("No cluster assignments found in AnnData object")
        
        cluster_key = cluster_keys[0]
        logger.info(f"Using {cluster_key} for pathway-specific interactions")
    results = {}
    
    for pathway_name, genes in pathways.items():
        valid_genes = [gene for gene in genes if gene in adata.var_names]
        
        if not valid_genes:
            logger.warning(f"No valid genes found for pathway {pathway_name}")
            continue
        
        logger.info(f"Analyzing pathway {pathway_name} with {len(valid_genes)} genes")
        pathway_lr_pairs = []
        
        for ligand, receptor in lr_pairs:
            if ligand in valid_genes and receptor in valid_genes:
                pathway_lr_pairs.append((ligand, receptor))
        
        if not pathway_lr_pairs:
            logger.warning(f"No valid ligand-receptor pairs found for pathway {pathway_name}")
            continue
        cluster_categories = adata.obs[cluster_key].cat.categories.tolist()
        logger.info(f"Computing interaction scores for {len(pathway_lr_pairs)} ligand-receptor pairs")
        scores = compute_interaction_scores(
            adata, 
            source_groups=cluster_categories,
            target_groups=cluster_categories,
            lr_pairs=pathway_lr_pairs,
            cluster_key=cluster_key
        )
        results[pathway_name] = scores
    
    return results

def compute_spatial_lr_interactions(adata, cluster_key=None, n_rings=3, 
                                 max_distance=None, context_method='exp_decay'):
    """
    Compute spatially-aware ligand-receptor interactions
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str, optional
        Key in adata.obs for cluster assignments
    n_rings : int, optional
        Number of concentric rings to consider
    max_distance : float, optional
        Maximum distance to consider for interactions
    context_method : str, optional
        Method for computing spatial context. Options: 'linear', 'exp_decay'
    
    Returns
    -------
    dict
        Dictionary with spatially-aware interaction results
    """
    logger.info("Computing spatially-aware ligand-receptor interactions")
    if cluster_key is None:
        cluster_keys = [key for key in adata.obs.columns if 
                      any(key.startswith(prefix) for prefix in 
                         ['leiden', 'louvain', 'kmeans', 'spatial_domains'])]
        
        if not cluster_keys:
            logger.error("No cluster assignments found in AnnData object")
            raise ValueError("No cluster assignments found in AnnData object")
        
        cluster_key = cluster_keys[0]
        logger.info(f"Using {cluster_key} for spatial interactions")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    clusters = adata.obs[cluster_key].values
    unique_clusters = adata.obs[cluster_key].cat.categories.tolist()
    coords = adata.obsm['spatial']
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(coords))
    if max_distance is None:
        max_distance = np.percentile(dist_matrix, 90)  # 90th percentile
        logger.info(f"Using max distance: {max_distance:.2f}")
    bins = np.linspace(0, max_distance, n_rings + 1)
    lr_pairs = _get_lr_database()
    valid_genes = adata.var_names.tolist()
    filtered_pairs = []
    
    for ligand, receptor in lr_pairs:
        if ligand in valid_genes and receptor in valid_genes:
            filtered_pairs.append((ligand, receptor))
    
    if not filtered_pairs:
        logger.error("No valid ligand-receptor pairs found in the dataset")
        raise ValueError("No valid ligand-receptor pairs found in the dataset")
    spatial_interactions = {}
    
    for ligand, receptor in filtered_pairs:
        ligand_idx = adata.var_names.get_loc(ligand)
        receptor_idx = adata.var_names.get_loc(receptor)
        if hasattr(adata.X, 'toarray'):
            ligand_expr = adata.X[:, ligand_idx].toarray().flatten()
            receptor_expr = adata.X[:, receptor_idx].toarray().flatten()
        else:
            ligand_expr = adata.X[:, ligand_idx]
            receptor_expr = adata.X[:, receptor_idx]
        for i in range(adata.n_obs):
            source_cluster = clusters[i]
            source_ligand = ligand_expr[i]
            if source_ligand <= 0:
                continue
            for j in range(adata.n_obs):
                if i == j:
                    continue
                target_cluster = clusters[j]
                target_receptor = receptor_expr[j]
                if target_receptor <= 0:
                    continue
                distance = dist_matrix[i, j]
                if distance > max_distance:
                    continue
                bin_idx = np.digitize(distance, bins) - 1
                if context_method == 'linear':
                    weight = 1 - distance / max_distance
                elif context_method == 'exp_decay':
                    weight = np.exp(-distance / (max_distance / 3))
                else:
                    logger.warning(f"Unknown context method {context_method}, using 'exp_decay'")
                    weight = np.exp(-distance / (max_distance / 3))
                interaction_score = source_ligand * target_receptor * weight
                key = (source_cluster, target_cluster, f"{ligand}-{receptor}", bin_idx)
                
                if key in spatial_interactions:
                    spatial_interactions[key] += interaction_score
                else:
                    spatial_interactions[key] = interaction_score
    rows = []
    
    for (source, target, pair, bin_idx), score in spatial_interactions.items():
        rows.append({
            'source_cluster': source,
            'target_cluster': target,
            'pair': pair,
            'distance_bin': bin_idx,
            'min_distance': bins[bin_idx],
            'max_distance': bins[bin_idx + 1],
            'score': score
        })
    if rows:
        df = pd.DataFrame(rows)
        df[['ligand', 'receptor']] = df['pair'].str.split('-', expand=True)
        bin_labels = [f"Ring {i+1}" for i in range(n_rings)]
        df['distance_ring'] = pd.Categorical(
            [bin_labels[idx] for idx in df['distance_bin']],
            categories=bin_labels
        )
        df = df.sort_values('score', ascending=False)
        agg_df = df.groupby(['source_cluster', 'target_cluster', 'distance_ring']).agg({
            'score': 'sum'
        }).reset_index()
        
        result = {
            'interactions': df,
            'aggregated': agg_df,
            'bins': bins,
            'bin_labels': bin_labels
        }
        
        return result
    else:
        empty_df = pd.DataFrame(columns=['source_cluster', 'target_cluster', 'pair', 
                                        'distance_bin', 'min_distance', 'max_distance', 
                                        'score', 'ligand', 'receptor', 'distance_ring'])
        result = {
            'interactions': empty_df,
            'aggregated': empty_df,
            'bins': bins,
            'bin_labels': [f"Ring {i+1}" for i in range(n_rings)]
        }
        
        return result
