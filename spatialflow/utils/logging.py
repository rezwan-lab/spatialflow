import logging
import sys
from pathlib import Path
import datetime
import os

def setup_logging(level="INFO", log_file=None, log_format=None):
    """
    Set up logging for the spatialflow package
    
    Parameters
    ----------
    level : str, optional
        Logging level. Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    log_file : str or Path, optional
        Path to log file. If None, logs only to console
    log_format : str, optional
        Format string for log messages. If None, a default format is used
        
    Returns
    -------
    logging.Logger
        Root logger for the spatialflow package
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logger = logging.getLogger('spatialflow')
    logger.setLevel(numeric_level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    logger.info(f"Logging level set to {level}")
    
    return logger

def log_execution_time(logger, start_time=None):
    """
    Log the execution time of a code block
    
    Parameters
    ----------
    logger : logging.Logger
        Logger to use
    start_time : datetime.datetime, optional
        Start time. If None, current time is used
        
    Returns
    -------
    function
        Function to call at the end of the code block
    """
    if start_time is None:
        start_time = datetime.datetime.now()
    
    def log_end(message="Execution completed"):
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        elif minutes > 0:
            time_str = f"{int(minutes)}m {seconds:.2f}s"
        else:
            time_str = f"{seconds:.2f}s"
        
        logger.info(f"{message} in {time_str}")
    
    return log_end

def get_module_logger(module_name):
    """
    Get a logger for a specific module
    
    Parameters
    ----------
    module_name : str
        Name of the module
        
    Returns
    -------
    logging.Logger
        Logger for the module
    """
    return logging.getLogger(f"spatialflow.{module_name}")

class LoggingContext:
    """
    Context manager for temporarily changing logging level
    
    Example
    -------
    >>> with LoggingContext('spatialflow', level=logging.DEBUG):
    ...     # Do something with debug logging
    >>> # Logging level is restored
    """
    
    def __init__(self, logger_name, level=None, handler=None, close=True):
        """
        Initialize the context manager
        
        Parameters
        ----------
        logger_name : str
            Name of the logger
        level : int, optional
            Logging level to set temporarily
        handler : logging.Handler, optional
            Handler to add temporarily
        close : bool, optional
            Whether to close the handler on exit
        """
        self.logger = logging.getLogger(logger_name)
        self.level = level
        self.handler = handler
        self.close = close
        self.old_level = self.logger.level
        
    def __enter__(self):
        """
        Enter the context manager
        """
        if self.level is not None:
            self.logger.setLevel(self.level)
        
        if self.handler is not None:
            self.logger.addHandler(self.handler)
            
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager
        """
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        
        if self.handler is not None:
            self.logger.removeHandler(self.handler)
            
            if self.close:
                self.handler.close()

def log_system_info(logger):
    """
    Log information about the system
    
    Parameters
    ----------
    logger : logging.Logger
        Logger to use
    """
    import platform
    import sys
    import multiprocessing
    import numpy as np
    import scanpy as sc
    import squidpy as sq
    
    logger.info("=" * 80)
    logger.info("System Information:")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Python implementation: {platform.python_implementation()}")
    logger.info(f"Operating system: {platform.system()} {platform.release()}")
    try:
        logger.info(f"Number of CPU cores: {multiprocessing.cpu_count()}")
    except:
        logger.info("Could not determine number of CPU cores")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Memory: {memory.total / (1024**3):.1f} GB (Available: {memory.available / (1024**3):.1f} GB)")
    except:
        logger.info("Could not determine system memory")
    logger.info("Package versions:")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"ScanPy: {sc.__version__}")
    logger.info(f"Squidpy: {sq.__version__}")
    
    try:
        import pandas as pd
        logger.info(f"Pandas: {pd.__version__}")
    except:
        logger.info("Pandas not installed")
    
    try:
        import scipy
        logger.info(f"SciPy: {scipy.__version__}")
    except:
        logger.info("SciPy not installed")
    
    try:
        import matplotlib
        logger.info(f"Matplotlib: {matplotlib.__version__}")
    except:
        logger.info("Matplotlib not installed")
    
    try:
        import anndata
        logger.info(f"AnnData: {anndata.__version__}")
    except:
        logger.info("AnnData not installed")
    
    logger.info("=" * 80)

def capture_warnings(logger=None):
    """
    Capture warnings and redirect them to the logger
    
    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. If None, spatialflow root logger is used
    """
    import warnings
    
    if logger is None:
        logger = logging.getLogger('spatialflow')
    
    logging.captureWarnings(True)
    def warning_filter(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"{category.__name__}: {message}")
    
    warnings.showwarning = warning_filter