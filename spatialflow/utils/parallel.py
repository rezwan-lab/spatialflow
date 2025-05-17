import logging
import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Any, Union, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger('spatialflow.utils.parallel')

def parallelize(func: Callable, items: List[Any], n_jobs: int = -1, 
               backend: str = 'processes', show_progress: bool = True,
               **kwargs) -> List[Any]:
    """
    Run a function in parallel over a list of items
    
    Parameters
    ----------
    func : callable
        Function to apply to each item
    items : list
        List of items to process
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all available cores
    backend : str, optional
        Backend to use. Options: 'processes', 'threads', 'serial'
    show_progress : bool, optional
        Whether to show a progress bar
    **kwargs
        Additional arguments to pass to func
        
    Returns
    -------
    List[Any]
        Results of applying func to each item
    """
    if n_jobs <= 0:
        n_jobs = mp.cpu_count()
    n_jobs = min(n_jobs, len(items))
    
    logger.info(f"Running {len(items)} tasks with {n_jobs} parallel jobs using {backend} backend")
    
    if backend == 'serial' or n_jobs == 1:
        if show_progress:
            results = [func(item, **kwargs) for item in tqdm(items, desc="Processing")]
        else:
            results = [func(item, **kwargs) for item in items]
        return results
    item_kwargs = [(item, kwargs) for item in items]
    def _func_wrapper(args):
        item, kwargs = args
        return func(item, **kwargs)
    if backend == 'processes':
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            if show_progress:
                results = list(tqdm(executor.map(_func_wrapper, item_kwargs), total=len(items), desc="Processing"))
            else:
                results = list(executor.map(_func_wrapper, item_kwargs))
    
    elif backend == 'threads':
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            if show_progress:
                results = list(tqdm(executor.map(_func_wrapper, item_kwargs), total=len(items), desc="Processing"))
            else:
                results = list(executor.map(_func_wrapper, item_kwargs))
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return results

def parallelize_dataframe(func: Callable, df: pd.DataFrame, n_jobs: int = -1, 
                        backend: str = 'processes', show_progress: bool = True,
                        **kwargs) -> pd.DataFrame:
    """
    Apply a function to a DataFrame in parallel
    
    Parameters
    ----------
    func : callable
        Function to apply to each chunk of the DataFrame
    df : pandas.DataFrame
        DataFrame to process
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all available cores
    backend : str, optional
        Backend to use. Options: 'processes', 'threads', 'serial'
    show_progress : bool, optional
        Whether to show a progress bar
    **kwargs
        Additional arguments to pass to func
        
    Returns
    -------
    pandas.DataFrame
        Processed DataFrame
    """
    if n_jobs <= 0:
        n_jobs = mp.cpu_count()
    n_jobs = min(n_jobs, len(df))
    
    logger.info(f"Processing DataFrame with {len(df)} rows using {n_jobs} parallel jobs")
    
    if backend == 'serial' or n_jobs == 1:
        return func(df, **kwargs)
    chunk_size = len(df) // n_jobs
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    if show_progress:
        tqdm.pandas(desc="Processing DataFrame")
        results = parallelize(func, chunks, n_jobs=n_jobs, backend=backend, show_progress=show_progress, **kwargs)
    else:
        results = parallelize(func, chunks, n_jobs=n_jobs, backend=backend, show_progress=False, **kwargs)
    if isinstance(results[0], pd.DataFrame):
        return pd.concat(results, axis=0)
    else:
        return results

def parallelize_over_genes(func: Callable, adata, gene_list: Optional[List[str]] = None,
                         n_jobs: int = -1, backend: str = 'processes', 
                         show_progress: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Apply a function to each gene in parallel
    
    Parameters
    ----------
    func : callable
        Function to apply to each gene. Should take gene name as first argument
    adata : AnnData
        AnnData object containing gene expression data
    gene_list : list of str, optional
        List of genes to process. If None, uses all genes in adata
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all available cores
    backend : str, optional
        Backend to use. Options: 'processes', 'threads', 'serial'
    show_progress : bool, optional
        Whether to show a progress bar
    **kwargs
        Additional arguments to pass to func
        
    Returns
    -------
    dict
        Dictionary mapping gene names to results of func
    """
    if gene_list is None:
        gene_list = adata.var_names.tolist()
    else:
        gene_list = [gene for gene in gene_list if gene in adata.var_names]
    
    logger.info(f"Processing {len(gene_list)} genes with {n_jobs} parallel jobs")
    def _process_gene(gene, **kwargs):
        try:
            result = func(gene, adata, **kwargs)
            return gene, result
        except Exception as e:
            logger.warning(f"Error processing gene {gene}: {str(e)}")
            return gene, None
    results = parallelize(_process_gene, gene_list, n_jobs=n_jobs, 
                        backend=backend, show_progress=show_progress, **kwargs)
    result_dict = {gene: result for gene, result in results if result is not None}
    
    logger.info(f"Successfully processed {len(result_dict)} out of {len(gene_list)} genes")
    
    return result_dict

def parallelize_over_spots(func: Callable, adata, spot_indices: Optional[List[int]] = None,
                         n_jobs: int = -1, backend: str = 'processes', 
                         show_progress: bool = True, **kwargs) -> List[Any]:
    """
    Apply a function to each spot in parallel
    
    Parameters
    ----------
    func : callable
        Function to apply to each spot. Should take spot index as first argument
    adata : AnnData
        AnnData object containing gene expression data
    spot_indices : list of int, optional
        List of spot indices to process. If None, uses all spots in adata
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all available cores
    backend : str, optional
        Backend to use. Options: 'processes', 'threads', 'serial'
    show_progress : bool, optional
        Whether to show a progress bar
    **kwargs
        Additional arguments to pass to func
        
    Returns
    -------
    list
        List of results for each spot
    """
    if spot_indices is None:
        spot_indices = list(range(adata.n_obs))
    
    logger.info(f"Processing {len(spot_indices)} spots with {n_jobs} parallel jobs")
    def _process_spot(spot_idx, **kwargs):
        try:
            return func(spot_idx, adata, **kwargs)
        except Exception as e:
            logger.warning(f"Error processing spot {spot_idx}: {str(e)}")
            return None
    results = parallelize(_process_spot, spot_indices, n_jobs=n_jobs, 
                        backend=backend, show_progress=show_progress, **kwargs)
    
    return results

def apply_map_reduce(map_func: Callable, reduce_func: Callable, 
                   items: List[Any], n_jobs: int = -1, 
                   backend: str = 'processes', show_progress: bool = True,
                   **kwargs) -> Any:
    """
    Apply a map-reduce operation in parallel
    
    Parameters
    ----------
    map_func : callable
        Function to apply to each item in the map phase
    reduce_func : callable
        Function to combine results in the reduce phase
    items : list
        List of items to process
    n_jobs : int, optional
        Number of parallel jobs for the map phase. -1 means use all available cores
    backend : str, optional
        Backend to use. Options: 'processes', 'threads', 'serial'
    show_progress : bool, optional
        Whether to show a progress bar
    **kwargs
        Additional arguments to pass to map_func
        
    Returns
    -------
    Any
        Result of reduce_func applied to the outputs of map_func
    """
    mapped_results = parallelize(map_func, items, n_jobs=n_jobs, 
                               backend=backend, show_progress=show_progress, **kwargs)
    mapped_results = [result for result in mapped_results if result is not None]
    if mapped_results:
        return reduce_func(mapped_results)
    else:
        logger.warning("No valid results to reduce")
        return None

class ProgressParallel:
    """
    Class for running jobs in parallel with a progress bar
    
    Example
    -------
    >>> from spatialflow.utils.parallel import ProgressParallel
    >>> import numpy as np
    >>> def process(x):
    ...     return x ** 2
    >>> pp = ProgressParallel(n_jobs=4, total=10, desc="Processing")
    >>> results = pp(process, np.arange(10))
    """
    
    def __init__(self, n_jobs: int = -1, backend: str = 'processes', 
                total: Optional[int] = None, desc: str = "Processing", **kwargs):
        """
        Initialize the parallel runner
        
        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs. -1 means use all available cores
        backend : str, optional
            Backend to use. Options: 'processes', 'threads', 'serial'
        total : int, optional
            Total number of items (for progress bar)
        desc : str, optional
            Description for the progress bar
        **kwargs
            Additional arguments to pass to tqdm
        """
        self.n_jobs = n_jobs
        self.backend = backend
        self.total = total
        self.desc = desc
        self.kwargs = kwargs
    
    def __call__(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """
        Run the function in parallel
        
        Parameters
        ----------
        func : callable
            Function to apply to each item
        items : list
            List of items to process
        **kwargs
            Additional arguments to pass to func
            
        Returns
        -------
        list
            Results of applying func to each item
        """
        total = self.total if self.total is not None else len(items)
        
        return parallelize(func, items, n_jobs=self.n_jobs, backend=self.backend, 
                         show_progress=True, desc=self.desc, total=total, **kwargs)