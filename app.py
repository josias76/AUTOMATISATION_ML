import streamlit as st
from ydata_profiling import ProfileReport
import pandas as pd
#import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup as setup_reg 
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg


from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class