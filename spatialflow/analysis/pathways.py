import scanpy as sc
import numpy as np
import pandas as pd
import logging
from anndata import AnnData

logger = logging.getLogger('spatialflow.analysis.pathways')

def run_pathway_analysis(adata, custom_gene_lists=None, key_added='pathway', group_key=None, **kwargs):
    """
    Run pathway analysis on the data
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    custom_gene_lists : dict, optional
        Dictionary of custom gene lists for pathway analysis
    key_added : str, optional
        Key to add to adata.uns for pathway results
    group_key : str, optional
        Key in adata.obs for group assignments
    **kwargs
        Additional arguments for pathway analysis
    
    Returns
    -------
    adata : AnnData
        Annotated data matrix with pathway analysis results
    """
    logger.info(f"Running pathway analysis using {kwargs.get('method', 'gsea')} method")
    if group_key is None:
        possible_keys = [key for key in adata.obs.columns if 
                       any(key.startswith(prefix) for prefix in 
                          ['leiden', 'louvain', 'kmeans', 'spatial_clusters', 'spatial_domains'])]
        
        if possible_keys:
            group_key = possible_keys[0]
            logger.info(f"Using {group_key} for pathway analysis grouping")
        else:
            logger.error("No group key found for GSEA. Please provide a group_key.")
            raise ValueError("No group key found for GSEA. Please provide a group_key.")
    _run_gsea_analysis(adata, custom_gene_lists, key_added, group_key=group_key, **kwargs)
    
    return adata

def _run_gsea_analysis(adata, custom_gene_lists=None, key_added='gsea', group_key=None, 
                     max_comparisons=5, min_group_size=10, **kwargs):
    """
    Run Gene Set Enrichment Analysis (GSEA)
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    custom_gene_lists : dict, optional
        Dictionary of custom gene lists for pathway analysis
    key_added : str, optional
        Key to add to adata.uns for GSEA results
    group_key : str, optional
        Key in adata.obs for group assignments
    max_comparisons : int, optional
        Maximum number of group comparisons to run
    min_group_size : int, optional
        Minimum size for a group to be included
    **kwargs
        Additional arguments for GSEA
        
    Returns
    -------
    adata : AnnData
        Annotated data matrix with GSEA results
    """
    try:
        import gseapy
        from gseapy import prerank, gsea
    except ImportError:
        logger.error("GSEApy package not installed. Please install it with 'pip install gseapy'")
        raise ImportError("GSEApy package required for GSEA analysis")
    
    logger.info("Running Gene Set Enrichment Analysis (GSEA)")
    if group_key is None or group_key not in adata.obs:
        logger.error("No group key found for GSEA. Please provide a group_key.")
        raise ValueError("No group key found for GSEA. Please provide a group_key.")
    if custom_gene_lists is not None and len(custom_gene_lists) > 0:
        gene_sets = custom_gene_lists
    else:
        gene_sets = "KEGG_2021_Human"  # Default GSEAPY gene set
    group_counts = adata.obs[group_key].value_counts()
    valid_groups = group_counts[group_counts >= min_group_size].index.tolist()
    
    if len(valid_groups) < 2:
        logger.warning(f"Not enough groups with at least {min_group_size} members. Skipping GSEA.")
        return adata
    valid_groups = sorted(valid_groups, key=lambda g: group_counts[g], reverse=True)
    reference_group = valid_groups[0]  # Use largest group as reference
    comparison_groups = valid_groups[1:min(len(valid_groups), max_comparisons+1)]
    
    logger.info(f"Using {reference_group} as reference group for GSEA")
    logger.info(f"Will compare against {len(comparison_groups)} other groups")
    import scanpy as sc
    try:
        if 'highly_variable' not in adata.var:
            sc.pp.highly_variable_genes(adata, min_mean=0.01, max_mean=8, min_disp=0.5)
        var_genes = adata.var[adata.var['highly_variable']].index.tolist()
        adata_filt = adata[:, var_genes].copy()
        logger.info(f"Using {len(var_genes)} highly variable genes for GSEA")
    except Exception as e:
        logger.warning(f"Could not filter for highly variable genes: {str(e)}")
        adata_filt = adata.copy()
    sc.tl.rank_genes_groups(adata_filt, groupby=group_key, groups=comparison_groups, 
                          reference=reference_group, method='t-test_overestim_var',
                          pts=True)  # t-test gives better resolution than wilcoxon
    gsea_results = {}
    
    for group in comparison_groups:
        logger.info(f"Running GSEA for {reference_group} vs {group}")
        
        try:
            gene_names = adata_filt.uns['rank_genes_groups']['names'][group]
            gene_scores = adata_filt.uns['rank_genes_groups']['scores'][group]
            non_zero_mask = gene_scores != 0
            gene_names = gene_names[non_zero_mask]
            gene_scores = gene_scores[non_zero_mask]
            import pandas as pd
            gene_list = pd.DataFrame({
                'Gene': gene_names,
                'Score': gene_scores
            })
            gene_list = gene_list.drop_duplicates('Score', keep='first')
            result = prerank(rnk=gene_list, 
                          gene_sets=gene_sets,
                          threads=1,  # Reduce thread count to save memory
                          permutation_num=100,  # Lower from default 1000
                          seed=42,
                          **kwargs)
            result_dict = {}
            if hasattr(result, 'res2d'):
                for key in ['nes', 'pval', 'fdr', 'geneset_size']:
                    if key in result.res2d:
                        result_dict[key] = result.res2d[key].to_dict()
            else:
                result_dict['summary'] = str(result)
            
            gsea_results[f"{reference_group}_vs_{group}"] = result_dict
            
        except Exception as e:
            logger.warning(f"GSEA failed for {reference_group} vs {group}: {str(e)}")
    adata.uns[key_added] = gsea_results
    
    logger.info("GSEA analysis completed")
    return adata

def _run_ora_analysis(adata, custom_gene_lists=None, n_genes=2000, 
                    key_added='pathway_ora', **kwargs):
    """
    Run Over-Representation Analysis (ORA)
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    custom_gene_lists : dict, optional
        Dictionary mapping pathway names to gene lists
    n_genes : int, optional
        Number of top genes to use for ORA analysis
    key_added : str, optional
        Key to add to adata.uns for ORA results
    **kwargs
        Additional parameters for ORA
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added ORA results
    """
    logger.info("Running Over-Representation Analysis (ORA)")
    gene_sets = _get_gene_sets(custom_gene_lists)
    if 'group_key' in kwargs:
        group_key = kwargs.pop('group_key')
    else:
        cluster_keys = [key for key in adata.obs.columns if 
                      any(key.startswith(prefix) for prefix in 
                         ['leiden', 'louvain', 'kmeans', 'spatial_domains'])]
        
        if cluster_keys:
            group_key = cluster_keys[0]
            logger.info(f"Using {group_key} for group comparison")
        else:
            logger.error("No group key found for ORA. Please provide a group_key.")
            raise ValueError("No group key found for ORA. Please provide a group_key.")
    if group_key not in adata.obs:
        logger.error(f"Group key {group_key} not found in adata.obs")
        raise ValueError(f"Group key {group_key} not found in adata.obs")
    groups = adata.obs[group_key].cat.categories.tolist()
    ora_results = {}
    
    for group in groups:
        logger.info(f"Running ORA for group {group}")
        sc.tl.rank_genes_groups(adata, groupby=group_key, groups=[group], method='wilcoxon')
        top_genes = adata.uns['rank_genes_groups']['names'][group][:n_genes]
        pathway_results = []
        background_genes = adata.var_names.tolist()
        
        for pathway, genes in gene_sets.items():
            pathway_genes = [gene for gene in genes if gene in background_genes]
            
            if not pathway_genes:
                continue
            overlap_genes = [gene for gene in top_genes if gene in pathway_genes]
            n_overlap = len(overlap_genes)
            from scipy.stats import hypergeom
            M = len(background_genes)  # Total number of genes
            n = len(pathway_genes)     # Number of genes in pathway
            N = len(top_genes)         # Number of selected genes
            k = n_overlap              # Number of selected genes in pathway
            p_value = hypergeom.sf(k-1, M, n, N)
            expected = (n / M) * N
            fold_enrichment = k / expected if expected > 0 else float('inf')
            pathway_results.append({
                'pathway': pathway,
                'overlap_genes': overlap_genes,
                'n_overlap': n_overlap,
                'n_pathway': n,
                'n_background': M,
                'n_selected': N,
                'p_value': p_value,
                'fold_enrichment': fold_enrichment
            })
        if pathway_results:
            results_df = pd.DataFrame(pathway_results)
            from statsmodels.stats.multitest import multipletests
            results_df['p_adjusted'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
            results_df = results_df.sort_values('p_adjusted')
            ora_results[group] = results_df
    adata.uns[key_added] = {
        'ora_results': ora_results,
        'parameters': {
            'group_key': group_key,
            'groups': groups,
            'n_genes': n_genes,
            'gene_sets': list(gene_sets.keys())
        }
    }
    _log_ora_summary(ora_results)
    
    return adata

def _run_ssgsea_analysis(adata, custom_gene_lists=None, key_added='pathway_scores', **kwargs):
    """
    Run single-sample Gene Set Enrichment Analysis (ssGSEA)
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    custom_gene_lists : dict, optional
        Dictionary mapping pathway names to gene lists
    key_added : str, optional
        Prefix for pathway scores in adata.obs
    **kwargs
        Additional parameters for ssGSEA
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added ssGSEA scores
    """
    logger.info("Running single-sample Gene Set Enrichment Analysis (ssGSEA)")
    gene_sets = _get_gene_sets(custom_gene_lists)
    for pathway, genes in gene_sets.items():
        valid_genes = [gene for gene in genes if gene in adata.var_names]
        
        if not valid_genes:
            logger.warning(f"No valid genes found for pathway {pathway}")
            continue
        score_name = f"{key_added}_{pathway}"
        
        _calculate_ssgsea_score(adata, valid_genes, score_name, **kwargs)
        
        logger.info(f"Calculated ssGSEA score for {pathway} using {len(valid_genes)} genes")
    adata.uns[key_added] = {
        'pathways': list(gene_sets.keys()),
        'parameters': {
            'method': 'ssGSEA',
            'gene_sets': {pathway: genes for pathway, genes in gene_sets.items() 
                       if any(gene in adata.var_names for gene in genes)}
        }
    }
    
    return adata

def _calculate_ssgsea_score(adata, genes, score_name, exponential=True, scale=True, **kwargs):
    """
    Calculate ssGSEA score for a gene set
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes in the gene set
    score_name : str
        Name of the score to add to adata.obs
    exponential : bool, optional
        Whether to use exponential weighting
    scale : bool, optional
        Whether to scale the score between 0 and 1
    **kwargs
        Additional parameters
    
    Returns
    -------
    np.ndarray
        Array of ssGSEA scores
    """
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()
    gene_indices = [adata.var_names.get_loc(gene) for gene in genes if gene in adata.var_names]
    
    if not gene_indices:
        logger.warning(f"No valid genes found for {score_name}")
        return np.zeros(adata.n_obs)
    gene_expr = X[:, gene_indices]
    ranks = np.zeros_like(X)
    for i in range(adata.n_obs):
        ranks[i] = np.argsort(np.argsort(X[i]))
    gene_ranks = ranks[:, gene_indices]
    n_genes = len(gene_indices)
    n_total_genes = X.shape[1]
    scores = np.zeros(adata.n_obs)
    
    for i in range(adata.n_obs):
        sorted_indices = np.argsort(gene_ranks[i])
        sorted_ranks = gene_ranks[i, sorted_indices]
        running_sum = np.zeros(n_genes)
        
        for j in range(n_genes):
            if exponential:
                numerator = np.sum(np.power(sorted_ranks[:j+1], kwargs.get('exp_weight', 1.0)))
            else:
                numerator = np.sum(sorted_ranks[:j+1])
            denominator = np.sum(sorted_ranks)
            fraction_in_set = (j + 1) / n_genes
            running_sum[j] = (numerator / denominator) - fraction_in_set
        pos_max = np.max(running_sum) if np.max(running_sum) > 0 else 0
        neg_min = np.min(running_sum) if np.min(running_sum) < 0 else 0
        
        if abs(pos_max) > abs(neg_min):
            scores[i] = pos_max
        else:
            scores[i] = neg_min
    if scale:
        if np.max(scores) > np.min(scores):
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    adata.obs[score_name] = scores
    
    return scores

def _calculate_average_pathway_scores(adata, custom_gene_lists=None, 
                                    key_added='pathway_avg', **kwargs):
    """
    Calculate average expression for each pathway
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    custom_gene_lists : dict, optional
        Dictionary mapping pathway names to gene lists
    key_added : str, optional
        Prefix for pathway scores in adata.obs
    **kwargs
        Additional parameters
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added pathway scores
    """
    logger.info("Calculating average pathway scores")
    gene_sets = _get_gene_sets(custom_gene_lists)
    for pathway, genes in gene_sets.items():
        valid_genes = [gene for gene in genes if gene in adata.var_names]
        
        if not valid_genes:
            logger.warning(f"No valid genes found for pathway {pathway}")
            continue
        if len(valid_genes) >= 3:
            score_name = f"{key_added}_{pathway}"
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=score_name)
            logger.info(f"Calculated average score for {pathway} using {len(valid_genes)} genes")
    adata.uns[key_added] = {
        'pathways': list(gene_sets.keys()),
        'parameters': {
            'method': 'average',
            'gene_sets': {pathway: genes for pathway, genes in gene_sets.items() 
                       if any(gene in adata.var_names for gene in genes)}
        }
    }
    
    return adata

def _get_gene_sets(custom_gene_lists=None):
    """
    Get gene sets for pathway analysis
    
    Parameters
    ----------
    custom_gene_lists : dict, optional
        Dictionary mapping pathway names to gene lists
    
    Returns
    -------
    dict
        Dictionary mapping pathway names to gene lists
    """
    if custom_gene_lists is not None:
        return custom_gene_lists
    logger.info("Using built-in gene sets")
    gene_sets = {
        'T_cell_activation': [
            'CD3D', 'CD3E', 'CD3G', 'CD28', 'ICOS', 'CTLA4', 'PDCD1', 'CD27',
            'IL2RA', 'IL2RB', 'IL2RG', 'LAG3', 'TIGIT', 'HAVCR2', 'CD40LG',
            'CD69', 'LCK', 'ZAP70', 'ITK', 'GZMA', 'GZMB', 'PRF1', 'IFNG'
        ],
        'B_cell_function': [
            'CD19', 'CD79A', 'CD79B', 'MS4A1', 'CR2', 'FCRL1', 'BANK1', 'BLK',
            'CD22', 'CD27', 'CD38', 'IGHM', 'IGHD', 'IGHA1', 'IGHG1',
            'VPREB3', 'PAX5', 'EBF1', 'JCHAIN', 'IRF4', 'PRDM1', 'XBP1'
        ],
        'Myeloid_function': [
            'CD14', 'CD163', 'CD68', 'CSF1R', 'FCGR1A', 'FCGR2A', 'FCGR3A',
            'ITGAM', 'ITGAX', 'MARCO', 'MERTK', 'MSR1', 'TLR2', 'TLR4',
            'TLR7', 'TLR8', 'LYZ', 'VSIG4', 'ADGRE1', 'CD1C', 'FCER1A'
        ],
        'Inflammation': [
            'IL1A', 'IL1B', 'IL6', 'IL8', 'TNF', 'IL12A', 'IL12B', 'IL18',
            'CCL2', 'CCL3', 'CCL4', 'CCL5', 'CXCL9', 'CXCL10', 'CXCL11',
            'NLRP3', 'CASP1', 'PYCARD', 'TLR2', 'TLR4', 'NFKB1', 'RELA'
        ],
        'Cytokine_signaling': [
            'IL1A', 'IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL12A', 'IL12B',
            'IL15', 'IL18', 'IFNG', 'IFNB1', 'IFNA1', 'TGFB1', 'TGFB2',
            'TGFBR1', 'TGFBR2', 'IFNGR1', 'IFNGR2', 'IL1R1', 'IL2RA', 'IL2RB',
            'IL4R', 'IL6R', 'IL10RA', 'IL12RB1', 'IL15RA', 'IL18R1', 'IFNAR1'
        ],
        'Antigen_presentation': [
            'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DRA', 'HLA-DRB1', 'HLA-DQA1',
            'HLA-DQB1', 'HLA-DPA1', 'HLA-DPB1', 'CD74', 'CIITA', 'B2M',
            'TAP1', 'TAP2', 'TAPBP', 'PSMB8', 'PSMB9', 'PSMB10'
        ],
        'Cell_adhesion': [
            'ICAM1', 'VCAM1', 'ITGA4', 'ITGB1', 'ITGB2', 'ITGAL', 'ITGAM',
            'ITGAX', 'SELP', 'SELE', 'SELL', 'SELPLG', 'CXCR3', 'CXCR4',
            'CCR2', 'CCR5', 'CCR7', 'CX3CR1', 'PECAM1', 'CDH5', 'NCAM1'
        ],
        'Cell_proliferation': [
            'MKI67', 'PCNA', 'TOP2A', 'MCM2', 'MCM5', 'CCNA2', 'CCNB1',
            'CCNB2', 'CCND1', 'CCNE1', 'CDK1', 'CDK2', 'CDK4', 'CDK6',
            'CDKN1A', 'CDKN1B', 'E2F1', 'RB1', 'CDKN2A', 'MYC', 'BIRC5'
        ],
        'Apoptosis': [
            'BAX', 'BCL2', 'BCL2L1', 'MCL1', 'BID', 'BAK1', 'BAD', 'BBC3',
            'PMAIP1', 'CASP3', 'CASP8', 'CASP9', 'CYCS', 'DIABLO', 'FASLG',
            'FAS', 'TNFRSF10A', 'TNFRSF10B', 'XIAP', 'BIRC2', 'BIRC3'
        ],
    }
    
    return gene_sets

def _log_gsea_summary(gsea_results):
    """
    Log summary of GSEA results
    
    Parameters
    ----------
    gsea_results : dict
        Dictionary with GSEA results
    
    Returns
    -------
    None
    """
    logger.info("GSEA results summary:")
    
    for comparison, result in gsea_results.items():
        if hasattr(result, 'res2d') and result.res2d is not None:
            df = result.res2d
            sig_pos = df[(df['es'] > 0) & (df['fdr'] < 0.25)].shape[0]
            sig_neg = df[(df['es'] < 0) & (df['fdr'] < 0.25)].shape[0]
            
            logger.info(f"  {comparison}: {sig_pos} enriched, {sig_neg} depleted pathways (FDR < 0.25)")
            top_pos = df[df['es'] > 0].sort_values('fdr').head(3)
            for _, row in top_pos.iterrows():
                logger.info(f"    Enriched: {row['term']} (NES={row['nes']:.2f}, FDR={row['fdr']:.3f})")
            top_neg = df[df['es'] < 0].sort_values('fdr').head(3)
            for _, row in top_neg.iterrows():
                logger.info(f"    Depleted: {row['term']} (NES={row['nes']:.2f}, FDR={row['fdr']:.3f})")
        else:
            logger.warning(f"  {comparison}: No results available")

def _log_ora_summary(ora_results):
    """
    Log summary of ORA results
    
    Parameters
    ----------
    ora_results : dict
        Dictionary with ORA results
    
    Returns
    -------
    None
    """
    logger.info("ORA results summary:")
    
    for group, results_df in ora_results.items():
        sig_pathways = results_df[results_df['p_adjusted'] < 0.05]
        n_sig = sig_pathways.shape[0]
        
        logger.info(f"  {group}: {n_sig} significant pathways (FDR < 0.05)")
        top_pathways = sig_pathways.head(3)
        for _, row in top_pathways.iterrows():
            logger.info(f"    {row['pathway']} (FE={row['fold_enrichment']:.2f}, FDR={row['p_adjusted']:.3e})")

def calculate_pathway_enrichment(adata, cluster_key, score_key_prefix='pathway_avg',
                               threshold=0.05, **kwargs):
    """
    Calculate pathway enrichment for each cluster
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str
        Key in adata.obs for cluster assignments
    score_key_prefix : str, optional
        Prefix for pathway scores in adata.obs
    threshold : float, optional
        P-value threshold for significance
    **kwargs
        Additional parameters
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with pathway enrichment results
    """
    logger.info(f"Calculating pathway enrichment for {cluster_key}")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    pathway_columns = [col for col in adata.obs.columns if col.startswith(score_key_prefix)]
    
    if not pathway_columns:
        logger.error(f"No pathway scores found with prefix {score_key_prefix}")
        raise ValueError(f"No pathway scores found with prefix {score_key_prefix}")
    pathway_names = [col[len(score_key_prefix)+1:] for col in pathway_columns]
    clusters = adata.obs[cluster_key].cat.categories.tolist()
    results = []
    
    for cluster in clusters:
        mask = adata.obs[cluster_key] == cluster
        
        for pathway_col, pathway_name in zip(pathway_columns, pathway_names):
            scores = adata.obs[pathway_col].values
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(scores[mask], scores[~mask], equal_var=False)
            mean_cluster = np.mean(scores[mask])
            mean_others = np.mean(scores[~mask])
            if mean_others > 0:
                fold_change = mean_cluster / mean_others
            else:
                fold_change = float('inf') if mean_cluster > 0 else 1.0
            results.append({
                'cluster': cluster,
                'pathway': pathway_name,
                'mean_cluster': mean_cluster,
                'mean_others': mean_others,
                'fold_change': fold_change,
                't_statistic': t_stat,
                'p_value': p_val,
                'enriched': fold_change > 1
            })
    if results:
        results_df = pd.DataFrame(results)
        from statsmodels.stats.multitest import multipletests
        results_df['p_adjusted'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df['significant'] = results_df['p_adjusted'] < threshold
        adata.uns[f"{cluster_key}_pathway_enrichment"] = results_df
        _log_enrichment_summary(results_df, threshold)
        
        return results_df
    else:
        return pd.DataFrame()

def _log_enrichment_summary(results_df, threshold=0.05):
    """
    Log summary of pathway enrichment results
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with pathway enrichment results
    threshold : float, optional
        P-value threshold for significance
    
    Returns
    -------
    None
    """
    logger.info("Pathway enrichment summary:")
    clusters = results_df['cluster'].unique()
    
    for cluster in clusters:
        cluster_results = results_df[results_df['cluster'] == cluster]
        sig_up = cluster_results[(cluster_results['significant']) & 
                               (cluster_results['fold_change'] > 1)].shape[0]
        sig_down = cluster_results[(cluster_results['significant']) & 
                                 (cluster_results['fold_change'] < 1)].shape[0]
        
        logger.info(f"  {cluster}: {sig_up} up-regulated, {sig_down} down-regulated pathways (FDR < {threshold})")
        top_up = cluster_results[cluster_results['fold_change'] > 1].sort_values('p_adjusted').head(3)
        for _, row in top_up.iterrows():
            logger.info(f"    Up: {row['pathway']} (FC={row['fold_change']:.2f}, FDR={row['p_adjusted']:.3e})")
        top_down = cluster_results[cluster_results['fold_change'] < 1].sort_values('p_adjusted').head(3)
        for _, row in top_down.iterrows():
            logger.info(f"    Down: {row['pathway']} (FC={row['fold_change']:.2f}, FDR={row['p_adjusted']:.3e})")

def calculate_pathway_spatial_autocorrelation(adata, pathway_scores=None, n_perms=1000, 
                                           use_highly_variable=True, **kwargs):
    """
    Calculate spatial autocorrelation for pathway scores
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    pathway_scores : list, optional
        List of pathway score columns in adata.obs
    n_perms : int, optional
        Number of permutations for significance testing
    use_highly_variable : bool, optional
        Whether to use highly variable genes for analysis
    **kwargs
        Additional parameters for sq.gr.spatial_autocorr
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with spatial autocorrelation results
    """
    try:
        import squidpy as sq
    except ImportError:
        logger.error("Squidpy package not installed. Please install it with 'pip install squidpy'")
        raise ImportError("Squidpy package required for spatial autocorrelation analysis")
    
    logger.info("Calculating spatial autocorrelation for pathway scores")
    if 'spatial_connectivities' not in adata.obsp:
        logger.error("Spatial neighbors graph not found. Build a graph first.")
        raise ValueError("Spatial neighbors graph not found. Build a graph first.")
    if pathway_scores is None:
        pathway_scores = [col for col in adata.obs.columns if 
                        any(col.startswith(prefix) for prefix in 
                           ['pathway_', 'score_'])]
        
        if not pathway_scores:
            logger.error("No pathway scores found in adata.obs")
            raise ValueError("No pathway scores found in adata.obs")
    valid_scores = [col for col in pathway_scores if col in adata.obs]
    
    if not valid_scores:
        logger.error("No valid pathway scores found in adata.obs")
        raise ValueError("No valid pathway scores found in adata.obs")
    
    logger.info(f"Using {len(valid_scores)} pathway scores for analysis")
    temp_adata = adata.copy()
    for score in valid_scores:
        temp_adata.var_names = temp_adata.var_names.append(pd.Index([score]))
        from scipy import sparse
        scores = adata.obs[score].values.reshape(-1, 1)
        
        if sparse.issparse(temp_adata.X):
            scores_sparse = sparse.csr_matrix(scores)
            temp_adata.X = sparse.hstack([temp_adata.X, scores_sparse])
        else:
            temp_adata.X = np.hstack([temp_adata.X, scores])
    sq.gr.spatial_autocorr(
        temp_adata,
        genes=valid_scores,
        mode='moran',
        n_perms=n_perms,
        **kwargs
    )
    if 'moranI' in temp_adata.uns:
        moran_df = temp_adata.uns['moranI'].loc[valid_scores]
        pval_cols = [col for col in moran_df.columns if 'pval' in col.lower()]
        
        if pval_cols:
            pval_col = pval_cols[0]
            moran_df['significant'] = moran_df[pval_col] < 0.05
            moran_df = moran_df.sort_values('I', ascending=False)
            sig_count = moran_df['significant'].sum()
            logger.info(f"Found {sig_count} spatially autocorrelated pathways (p < 0.05)")
            top_pathways = moran_df.head(5)
            for score, row in top_pathways.iterrows():
                logger.info(f"  {score}: I={row['I']:.3f}, p={row[pval_col]:.3e}")
            adata.uns['pathway_spatial_autocorr'] = moran_df
            
            return moran_df
        else:
            logger.warning("No p-value column found in Moran's I results")
            return moran_df
    else:
        logger.warning("No Moran's I results found")
        return pd.DataFrame()
