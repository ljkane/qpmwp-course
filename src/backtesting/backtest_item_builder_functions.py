############################################################################
### QPMwP - BACKTEST ITEM BUILDER FUNCTIONS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     20.03.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Third party imports
import numpy as np
import pandas as pd
import xgboost as xgb





# --------------------------------------------------------------------------
# Backtest item builder functions (bibfn) - Selection
# --------------------------------------------------------------------------

def bibfn_selection_min_volume(bs, rebdate: str, **kwargs) -> pd.DataFrame:

    '''
    Backtest item builder function for defining the selection
    Filter stocks based on minimum volume (i.e., liquidity).
    '''

    # Arguments
    width = kwargs.get('width', 365)
    agg_fn = kwargs.get('agg_fn', np.median)
    min_volume = kwargs.get('min_volume', 500_000)

    # Volume data
    vol = (
        bs.data.get_volume_series(
            end_date=rebdate,
            width=width
        ).fillna(0)
    )
    vol_agg = vol.apply(agg_fn, axis=0)

    # Filtering
    vol_binary = pd.Series(1, index=vol.columns, dtype=int, name='binary')
    vol_binary.loc[vol_agg < min_volume] = 0


    # Output
    filter_values = pd.DataFrame({
        'values': vol_agg,
        'binary': vol_binary,
    }, index=vol_agg.index)

    return filter_values



def bibfn_selection_NA(bs, rebdate: str, **kwargs) -> pd.Series:

    '''
    Backtest item builder function for defining the selection.
    Filters out stocks which have more than 'na_threshold' NA values in the
    return series. Remaining NA values are filled with zeros.
    '''

    # Arguments
    width = kwargs.get('width', 252)
    na_threshold = kwargs.get('na_threshold', 10)

    # Data: get return series
    return_series = bs.data.get_return_series(
        width=width,
        end_date=rebdate,
        fillna_value=None,
    )

    # Identify colums of return_series with more than 10 NA value
    # and remove them from the selection
    na_counts = return_series.isna().sum()
    na_columns = na_counts[na_counts > na_threshold].index

    # Output
    filter_values = pd.Series(1, index=na_counts.index, dtype=int, name='binary')
    filter_values.loc[na_columns] = 0

    return filter_values.astype(int)



def bibfn_selection_gaps(bs, rebdate: str, **kwargs) -> pd.Series:

    '''
    Backtest item builder function for defining the selection.
    Drops elements from the selection when there is a gap
    of more than n_days (i.e., consecutive zero's) in the volume series.
    '''

    # Arguments
    width = kwargs.get('width', 252)
    n_days = kwargs.get('n_days', 21)

    # Volume data
    vol = (
        bs.data.get_volume_series(
            end_date=rebdate,
            width=width
        ).fillna(0)
    )

    # Calculate the length of the longest consecutive zero sequence
    def consecutive_zeros(column):
        return (column == 0).astype(int).groupby(column.ne(0).astype(int).cumsum()).sum().max()

    gaps = vol.apply(consecutive_zeros)

    # Output
    filter_values = pd.DataFrame({
        'values': gaps,
        'binary': (gaps <= n_days).astype(int),
    }, index=gaps.index)

    return filter_values



def bibfn_selection_data(bs: 'BacktestService', rebdate: str, **kwargs) -> pd.Series:

    '''
    Backtest item builder function for defining the selection
    based on all available return series.
    '''

    return_series = bs.data.get('return_series')
    if return_series is None:
        raise ValueError('Return series data is missing.')

    return pd.Series(np.ones(return_series.shape[1], dtype = int),
                     index = return_series.columns, name = 'binary')



def bibfn_selection_data_random(bs: 'BacktestService', rebdate: str, **kwargs) -> pd.Series:

    '''
    Backtest item builder function for defining the selection
    based on a random k-out-of-n sampling of all available return series.
    '''
    # Arguments
    k = kwargs.get('k', 10)
    seed = kwargs.get('seed')
    if seed is None:
        seed = np.random.randint(0, 1_000_000)    
    # Add the position of rebdate in bs.settings['rebdates'] to
    # the seed to make it change with the rebdate
    seed += bs.settings['rebdates'].index(rebdate)
    return_series = bs.data.get('return_series')

    if return_series is None:
        raise ValueError('Return series data is missing.')

    # Random selection
    # Set the random seed for reproducibility
    np.random.seed(seed)
    selected = np.random.choice(return_series.columns, k, replace = False)

    return pd.Series(np.ones(len(selected), dtype = int), index = selected, name = 'binary')



def bibfn_selection_ltr(bs: 'BacktestService', rebdate: str, **kwargs) -> None:
    '''
    This function constructs labels and features for a specific rebalancing date.
    It acts as a filtering since stocks which could not be labeled or which
    do not have features are excluded from the selection.
    '''

    # Define the selection by the ids available for the current rebalancing date
    df_test = bs.data.merged_df[bs.data.merged_df['date'] == rebdate]
    ids = list(df_test['id'].unique())

    # Return a binary series indicating the selected stocks
    return pd.Series(1, index=ids, name='binary', dtype=int)



def bibfn_selection_jkp_factor_scores(bs, rebdate: str, **kwargs) -> pd.DataFrame:

    '''
    Backtest item builder function for defining the selection.
    Filter stocks based on available scores in the jkp factor data.
    '''

    # Arguments
    fields = kwargs.get('fields')

    # Selection
    ids = bs.selection.selected
    if ids is None:
        ids = bs.data.jkp_data.index.get_level_values('id').unique()

    # Filter rows prior to the rebdate and within one year
    df = bs.data.jkp_data[fields]
    filtered_df = df.loc[
        (df.index.get_level_values('date') < rebdate) &
        (df.index.get_level_values('date') >= pd.to_datetime(rebdate) - pd.Timedelta(days=365))
    ]

    # Extract the last available value for each id
    scores = filtered_df.groupby('id').last()

    # Output
    filter_values = scores.copy()
    filter_values['binary'] = scores.notna().all(axis=1).astype(int)

    return filter_values





# --------------------------------------------------------------------------
# Backtest item builder functions (bibfn) - Optimization data
# --------------------------------------------------------------------------

def bibfn_return_series(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for return series.
    Prepares an element of bs.optimization_data with
    single stock return series that are used for optimization.
    '''

    # Arguments
    width = kwargs.get('width')

    # Data: get return series
    if hasattr(bs.data, 'get_return_series'):
        return_series = bs.data.get_return_series(
            width=width,
            end_date=rebdate,
            fillna_value=None,
        )
    else:
        return_series = bs.data.get('return_series')
        if return_series is None:
            raise ValueError('Return series data is missing.')

    # Selection
    ids = bs.selection.selected
    if len(ids) == 0:
        ids = return_series.columns

    # Subset the return series
    return_series = return_series[return_series.index <= rebdate].tail(width)[ids]

    # Remove weekends
    return_series = return_series[return_series.index.dayofweek < 5]

    # Output
    bs.optimization_data['return_series'] = return_series
    return None


def bibfn_bm_series(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for benchmark series.
    Prepares an element of bs.optimization_data with 
    the benchmark series that is be used for optimization.
    '''

    # Arguments
    width = kwargs.get('width')
    align = kwargs.get('align', True)
    name = kwargs.get('name', 'bm_series')

    # Data
    if hasattr(bs.data, name):
        data = getattr(bs.data, name)
    else:
        data = bs.data.get(name)
        if data is None:
            raise ValueError('Benchmark return series data is missing.')

    # Subset the benchmark series
    bm_series = data[data.index <= rebdate].tail(width)

    # Remove weekends
    bm_series = bm_series[bm_series.index.dayofweek < 5]

    # Append the benchmark series to the optimization data
    bs.optimization_data['bm_series'] = bm_series

    # Align the benchmark series to the return series
    if align:
        bs.optimization_data.align_dates(
            variable_names = ['bm_series', 'return_series'],
            dropna = True
        )

    return None



def bibfn_cap_weights(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    # Selection
    ids = bs.selection.selected

    # Data - market capitalization
    mcap = bs.data.market_data['mktcap']

    # Get last available values for current rebdate
    mcap = mcap[mcap.index.get_level_values('date') <= rebdate].groupby(
        level = 'id'
    ).last()

    # Remove duplicates
    mcap = mcap[~mcap.index.duplicated(keep=False)].loc[ids]

    # Attach cap-weights to the optimization data object
    bs.optimization_data['cap_weights'] = mcap / mcap.sum()

    return None


def bibfn_scores(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Copies scores from the selection object to the optimization data object
    '''

    ids = bs.selection.selected
    scores = bs.selection.filtered['scores'].loc[ids]
    # Drop the 'binary' column
    bs.optimization_data['scores'] = scores.drop(columns=['binary'])
    return None


def bibfn_scores_ltr(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Constructs scores based on a Learning-to-Rank model.        
    '''

    # Arguments
    params_xgb = kwargs.get('params_xgb')
    if params_xgb is None or not isinstance(params_xgb, dict):
        raise ValueError('params_xgb is not defined or not a dictionary.')
    training_dates = kwargs.get('training_dates')

    # Extract data
    df_train = bs.data.merged_df[bs.data.merged_df['date'] < rebdate]
    df_test = bs.data.merged_df[bs.data.merged_df['date'] == rebdate]
    df_test = df_test.loc[df_test['id'].drop_duplicates(keep='first').index]
    df_test = df_test.loc[df_test['id'].isin(bs.selection.selected)]

    # Training data
    X_train = (
        df_train.drop(['date', 'id', 'label', 'ret'], axis=1)
        # df_train.drop(['date', 'id', 'label'], axis=1)  # Include ret in the features as a proof of concept
    )
    y_train = df_train['label'].loc[X_train.index]
    grouped_train = df_train.groupby('date').size().to_numpy()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(grouped_train)

    # Test data
    y_test = pd.Series(df_test['label'].values, index=df_test['id'])
    X_test = df_test.drop(['date', 'id', 'label', 'ret'], axis=1)
    # X_test = df_test.drop(['date', 'id', 'label'], axis=1)  # Include ret in the features as a proof of concept
    grouped_test = df_test.groupby('date').size().to_numpy()
    dtest = xgb.DMatrix(X_test)
    dtest.set_group(grouped_test)

    # Train the model using the training data
    if rebdate in training_dates:
        model = xgb.train(params_xgb, dtrain, 100)
        bs.model_ltr = model
    else:
        # Use the previous model for the current rebalancing date
        model = bs.model_ltr

    # Predict using the test data
    pred = model.predict(dtest)
    preds =  pd.Series(pred, df_test['id'], dtype='float64')
    ranks = preds.rank(method='first', ascending=True).astype(int)

    # Output
    scores = pd.concat({
        'scores': preds,
        'ranks': (100 * ranks / len(ranks)).astype(int),  # Normalize the ranks to be between 0 and 100
        'true': y_test,
        'ret': pd.Series(df_test['ret'].values, index=df_test['id']),
    }, axis=1)
    bs.optimization_data['scores'] = scores
    return None

################ Linear Regression #############################

import statsmodels.api as sm


def bibfn_expected_returns(bs, rebdate: str, **kwargs) -> None:
    """
    Predict expected returns using linear regression on factor data from jkp_data.
    The model fits factor exposures to next period returns and predicts expected returns at rebalance date.

    :param bs: BacktestService instance
    :param rebdate: Rebalancing date (string or pd.Timestamp)
    :param kwargs: optional arguments:
        - 'factors': list of factor names to use (default: ['qmj', 'niq_su'])
        - 'lookahead_days': int, days ahead to calculate return as target (default: 63 ~ 3 months)
    """

    factors = kwargs.get('factors', ['qmj', 'niq_su'])
    lookahead_days = kwargs.get('lookahead_days', 63)  # approx 3 months

    # Prepare factor data up to rebdate (exclude rebdate itself)
    jkp = bs.data.jkp_data
    jkp_before = jkp.loc[jkp.index.get_level_values('date') < rebdate, factors]

    # Get corresponding dates and ids
    dates = jkp_before.index.get_level_values('date')
    ids = jkp_before.index.get_level_values('id')

    # Flatten to a DataFrame with columns for date, id, and factors
    df_factors = jkp_before.reset_index()

    # Prepare target returns: next period returns after rebdate
    returns = bs.data.get_return_series(width=lookahead_days + 1, end_date=None)
    # Shift returns by lookahead_days to get future returns as targets
    future_returns = returns.shift(-lookahead_days)

    # Map target returns to factor dates and ids (align by date + id)
    # We want return from date + lookahead_days for each stock id

    # Merge factor data with future returns at matching date + lookahead_days
    df_factors['target_date'] = df_factors['date'] + pd.Timedelta(days=lookahead_days)
    df_factors.set_index(['target_date', 'id'], inplace=True)
    target_returns = future_returns.stack().rename('target_return')
    df_merged = df_factors.join(target_returns, how='inner')

    # Clean: drop rows with missing target or factor data
    df_merged = df_merged.dropna(subset=factors + ['target_return'])

    if df_merged.empty:
        # No data to train on, fallback to zeros
        expected_returns = pd.Series(0, index=bs.selection.selected)
    else:
        # Prepare X and y
        X = df_merged[factors]
        y = df_merged['target_return']

        # Add constant for intercept
        X = sm.add_constant(X)

        # Fit OLS regression
        model = sm.OLS(y, X).fit()

        # Predict expected returns for latest factor values on rebdate
        latest_factors = jkp.loc[jkp.index.get_level_values('date') <= rebdate, factors]
        latest_factors = latest_factors.groupby('id').last()
        X_pred = sm.add_constant(latest_factors.reindex(bs.selection.selected).fillna(0))

        expected_returns = model.predict(X_pred)

        # Normalize expected returns to [-1, 1]
        max_abs = expected_returns.abs().max()
        if max_abs > 0:
            expected_returns = expected_returns / max_abs
        else:
            expected_returns = expected_returns * 0

    # Store expected returns in optimization data
    bs.optimization_data['expected_returns'] = expected_returns

    return None






# --------------------------------------------------------------------------
# Backtest item builder functions - Optimization constraints
# --------------------------------------------------------------------------

def bibfn_budget_constraint(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the budget constraint.
    '''

    # Arguments
    budget = kwargs.get('budget', 1)

    # Add constraint
    bs.optimization.constraints.add_budget(rhs = budget, sense = '=')
    return None


def bibfn_box_constraints(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the box constraints.
    '''

    # Arguments
    lower = kwargs.get('lower', 0)
    upper = kwargs.get('upper', 1)
    box_type = kwargs.get('box_type', 'LongOnly')

    # Constraints
    bs.optimization.constraints.add_box(box_type = box_type,
                                        lower = lower,
                                        upper = upper)
    return None


def bibfn_size_dependent_upper_bounds(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the upper bounds
    in dependence of a stock's market capitalization.
    '''

    # Arguments
    small_cap = kwargs.get('small_cap', {'threshold': 300_000_000, 'upper': 0.02})
    mid_cap = kwargs.get('small_cap', {'threshold': 1_000_000_000, 'upper': 0.05})
    large_cap = kwargs.get('small_cap', {'threshold': 10_000_000_000, 'upper': 0.1})

    # Selection
    ids = bs.optimization.constraints.ids

    # Data: market capitalization
    mcap = bs.data.market_data['mktcap']
    # Get last available valus for current rebdate
    mcap = mcap[mcap.index.get_level_values('date') <= rebdate].groupby(
        level = 'id'
    ).last()

    # Remove duplicates
    mcap = mcap[~mcap.index.duplicated(keep=False)]
    # Ensure that mcap contains all selected ids,
    # possibly extend mcap with zero values
    mcap = mcap.reindex(ids).fillna(0)

    # Generate the upper bounds
    upper = mcap * 0
    upper[mcap > small_cap['threshold']] = small_cap['upper']
    upper[mcap > mid_cap['threshold']] = mid_cap['upper']
    upper[mcap > large_cap['threshold']] = large_cap['upper']

    # Check if the upper bounds have already been set
    if not bs.optimization.constraints.box['upper'].empty:
        bs.optimization.constraints.add_box(
            box_type = 'LongOnly',
            upper = upper,
        )
    else:
        # Update the upper bounds by taking the minimum of the current and the new upper bounds
        bs.optimization.constraints.box['upper'] = np.minimum(
            bs.optimization.constraints.box['upper'],
            upper,
        )

    return None


def bibfn_turnover_constraint(bs, rebdate: str, **kwargs) -> None:
    """
    Function to assign a turnover constraint to the optimization.
    """
    if rebdate > bs.settings['rebdates'][0]:

        # Arguments
        turnover_limit = kwargs.get('turnover_limit')

        # Constraints
        bs.optimization.constraints.add_l1(
            name = 'turnover',
            rhs = turnover_limit,
            x0 = bs.optimization.params['x_init'],
        )

    return None

def bibfn_sector_exposure_constraint(bs, rebdate: str, **kwargs) -> None:
    """
    Adds sector exposure constraints to the optimization.
    
    :param bs: BacktestService instance
    :param rebdate: Rebalance date
    :param kwargs:  
        - max_sector_weight: dict[str, float] mapping sector names to max allowed portfolio weight
                            (default: 0.2 for all sectors)
    """
    max_sector_weight = kwargs.get('max_sector_weight', {})
    default_max = 0.2

    # Get selected stock IDs
    ids = bs.selection.selected

    # Get latest sector data per stock
    sectors = bs.data.market_data['sector']
    sectors_up_to_date = sectors.loc[sectors.index.get_level_values('date') <= rebdate]
    latest_sectors = sectors_up_to_date.groupby('id').last().reindex(ids)

    # Create sector binary exposure matrix: rows = stocks, columns = sectors
    sector_dummies = pd.get_dummies(latest_sectors).fillna(0)

    # For each sector, get max allowed weight (default if not specified)
    sector_limits = {
        sector: max_sector_weight.get(sector, default_max) for sector in sector_dummies.columns
    }

    # Add constraint for each sector
    for sector, max_weight in sector_limits.items():
        exposure_vector = sector_dummies[sector].values
        # Add linear constraint: sum(weights * exposure_vector) <= max_weight
        bs.optimization.constraints.add_linear_inequality(
            name = f"sector_{sector}_max",
            coeffs = exposure_vector,
            rhs = max_weight,
            sense = '<='
        )
    return None

def bibfn_factor_exposure_constraint(bs, rebdate: str, **kwargs) -> None:
    """
    Adds factor exposure constraints to the optimization.
    """
    factors = kwargs.get('factors', [])
    max_factor_exposure = kwargs.get('max_factor_exposure', {})
    default_max = 0.2

    ids = bs.selection.selected
    jkp = bs.data.jkp_data

    jkp_up_to_date = jkp.loc[jkp.index.get_level_values('date') <= rebdate]
    latest_factors = jkp_up_to_date.groupby('id').last().reindex(ids)

    for factor in factors:
        if factor not in latest_factors.columns:
            continue

        exposure_vector = latest_factors[factor].reindex(ids).fillna(0).astype(float)

        max_exp = max_factor_exposure.get(factor, default_max)

        # Add both positive and negative constraints
        bs.optimization.constraints.add_linear_inequality(
            name=f"factor_{factor}_max_pos",
            coeffs=exposure_vector,
            rhs=max_exp,
            sense="<="
        )
        bs.optimization.constraints.add_linear_inequality(
            name=f"factor_{factor}_max_neg",
            coeffs=-exposure_vector,
            rhs=max_exp,
            sense="<="
        )




def bibfn_selection_min_mktcap(bs, rebdate: str, **kwargs) -> pd.DataFrame:
    
    # Arguments
    threshold = kwargs.get('threshold',1e8)
    mcap = bs.data.market_data['mktcap']
    mcap_up_to_date = mcap.loc[mcap.index.get_level_values('date')<= rebdate]

    latest_mcap = mcap_up_to_date.groupby('id').last()

    binary_mask = (latest_mcap >= threshold).astype(int)

    return pd.DataFrame({'binary': binary_mask})


def bibfn_selection_volatility(bs, rebdate: str, **kwargs) -> pd.DataFrame:
    
    # Arguments
    window = kwargs.get('window',63)
    max_vol = kwargs.get('max_vol', 0.05)

    returns = bs.data.get_return_series(end_date=rebdate, width=window)
    vol = returns.std()

    binary_mask = (vol <= max_vol).astype(int)
    return pd.DataFrame({'binary': binary_mask})

# def bibfn_selection_quality(bs, rebdate: str, **kwargs) -> pd.DataFrame:
#     min_qmj = kwargs.get('min_qmj', 0.5)
#     min_niq_su = kwargs.get('min_niq_su', 0)

#     jkp = bs.data.jkp_data

#     # Safety check
#     if 'qmj' not in jkp.columns or 'niq_su' not in jkp.columns:
#         raise ValueError("Missing qmj or niq_su in jkp_data")

#     # Filter jkp up to rebalance date
#     qmj = jkp['qmj'].loc[jkp.index.get_level_values('date') <= rebdate]
#     niq_su = jkp['niq_su'].loc[jkp.index.get_level_values('date') <= rebdate]

#     print(f"[DEBUG] Number of qmj values before groupby: {qmj.shape}")
#     print(f"[DEBUG] Number of niq_su values before groupby: {niq_su.shape}")

#     # Group and get last values per stock
#     latest_qmj = qmj.groupby('id').last()
#     latest_niq_su = niq_su.groupby('id').last()

#     print(f"[DEBUG] latest_qmj has {latest_qmj.isna().sum()} NaNs")
#     print(f"[DEBUG] latest_niq_su has {latest_niq_su.isna().sum()} NaNs")

#     # Combine into DataFrame
#     df = pd.DataFrame({
#         'qmj': latest_qmj,
#         'niq_su': latest_niq_su
#     })

#     # Print example values
#     print(df.head(10))

#     # Drop missing values
#     df = df.dropna()

#     # Apply filter
#     mask = (df['qmj'] >= min_qmj) & (df['niq_su'] >= min_niq_su)

#     print(f"[DEBUG] Final mask sample:\n{mask.head(10)}")
#     print(f"[DEBUG] Final number of selected stocks: {mask.sum()}")

#     return pd.DataFrame({'binary': mask.astype(int)})


