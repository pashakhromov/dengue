import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from sklearn.model_selection import GridSearchCV


def compare_vars(data: pd.DataFrame, x: str, y: str) -> None:
    df = data[[x, y]]
    corr = df.corr().iloc[0, 1]
    print(f'corr = {corr:0.2f}')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    # fig.tight_layout(h_pad=5, w_pad=5)
    df.plot.hist(alpha=0.5, bins=30, ax=ax[0])
    df.plot.scatter(x=x, y=y, ax=ax[1])
    plt.show()


def corr_plot(data: pd.DataFrame, cols: Optional[List[str]] = None, title: Optional[str] = None) -> None:
    y_col = ['total_cases']
    x_col = list(set(data.columns) - set(y_col)) if cols is None else cols
    suffix = '' if title is None else f' | {title}'
    sns.heatmap(
        data[x_col + y_col].corr(),
        cmap=sns.color_palette('vlag', as_cmap=True)
    )
    plt.title('All correlations' + suffix)
    plt.show()

    data[x_col].corrwith(data[y_col[0]]).sort_values().plot.barh()
    plt.title('Correlation with target variable' + suffix)
    plt.show()


def get_seasonal_pattern(
        var: str,
        sj: pd.DataFrame,
        iq: pd.DataFrame,
        plot: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sj_seasonality = sj.groupby('weekofyear')[var].mean()
    sj_seasonality_smoothed = sj_seasonality.rolling(2 * 4, center=True, min_periods=4).mean()

    iq_seasonality = iq.groupby('weekofyear')[var].mean()
    iq_seasonality_smoothed = iq_seasonality.rolling(2 * 4, center=True, min_periods=4).mean()

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
        sj.plot.scatter(x='weekofyear', y=var, ax=ax[0], title=f'{var} | sj')
        iq.plot.scatter(x='weekofyear', y=var, ax=ax[1], title=f'{var} | iq')
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
        sj_seasonality.plot(ax=ax[0], title=f'{var} average | sj')
        sj_seasonality_smoothed.plot(ax=ax[0], title=f'{var} average | sj')
        iq_seasonality.plot(ax=ax[1], title=f'{var} average | iq')
        iq_seasonality_smoothed.plot(ax=ax[1], title=f'{var} average | iq')
        plt.show()

    return sj_seasonality, iq_seasonality


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def split(df: pd.DataFrame, train_fraction: float) -> dict[str, pd.DataFrame]:
    # x = coarse_grain(df)
    x = df[[c for c in df.columns if c != 'total_cases']]
    y = df['total_cases']
    n = int(df.shape[0] * train_fraction)
    return {
        'df_train': df.iloc[:n, :],
        'df_test': df.iloc[n:, :],
        'x_train': x.iloc[:n, :],
        'y_train': y.iloc[:n],
        'x_test': x.iloc[n:, :],
        'y_test': y.iloc[n:],
    }


def split_and_mix(df: pd.DataFrame, train_fraction: float) -> dict[str, pd.DataFrame]:
    sj = split(df[df['is_sj'] == 1], train_fraction)
    iq = split(df[df['is_sj'] == 0], train_fraction)
    return {k: pd.concat([sj[k], iq[k]]) for k in sj.keys()}


def get_scores(search: GridSearchCV) -> pd.Series:
    index = pd.MultiIndex.from_tuples(
        [(x['model__learning_rate'], x['model__max_depth']) for x in search.cv_results_['params']],
        names=['learning_rate', 'max_depth']
    )
    return (-1) * pd.Series(search.cv_results_['mean_test_score'], index=index)


def get_scores_from_loop(score) -> pd.Series:
    score_mean = []
    for p1, d in score.items():
        for p2, arr in d.items():
            score_mean.append({'max_depth': p1, 'learning_rate': p2, 'mae': np.mean(arr)})
    index = pd.MultiIndex.from_tuples(
        [(x['max_depth'], x['learning_rate']) for x in score_mean],
        names=['learning_rate', 'max_depth']
    )
    return pd.Series([x['mae'] for x in score_mean], index=index)


class Data:
    def __init__(self, x_path: str, y_path: Optional[str] = None):
        self.x = pd.read_csv(x_path)
        self.y = pd.read_csv(y_path) if y_path else None

    def get_x(self) -> pd.DataFrame:
        return self.coarse_grain(self.apply_imputation(self.change_temp(self.x)))

    def get_y(self) -> pd.Series:
        if self.y is None:
            return pd.Series(0, index=self.x.index)
        return self.apply_imputation(self.y)['total_cases']

    def get_xy(self) -> pd.DataFrame:
        return pd.concat([self.get_x(), self.get_y()], axis=1)

    def shift_and_get_x(self) -> pd.DataFrame:
        x = self.get_x()
        return pd.concat([
            self.get_shifted(x[x['is_sj'] == 1]),
            self.get_shifted(x[x['is_sj'] == 0]),
        ], axis=0).dropna()

    def shift_and_get_y(self) -> pd.Series:
        x = self.shift_and_get_x()
        y = self.get_y()
        return y.loc[x.index]

    def shift_and_get_xy(self):
        return pd.concat([self.shift_and_get_x(), self.shift_and_get_y()], axis=1)

    # def deseason_and_get_x(self) -> pd.DataFrame:
    #     x = self.get_x()
    #     sj_season = Seasonality(x[x.is_sj == 1], cols=self.get_climate_cols())
    #     iq_season = Seasonality(x[x.is_sj == 0], cols=self.get_climate_cols())
    #     return pd.concat([
    #         sj_season.get_residualized_data(),
    #         iq_season.get_residualized_data(),
    #     ], axis=0)

    def deseason_and_get_xy(self) -> pd.DataFrame:
        d = Deseasoner(cols=self.get_climate_and_target_cols()).fit(
            self.get_x(),
            self.get_y(),
        )
        x, y = d.transform(self.get_x(), self.get_y())
        return pd.concat([x, y], axis=1)

    # def deseason_and_get_xy(self) -> pd.DataFrame:
    #     return pd.concat([self.deseason_and_get_x(), self.get_y()], axis=1)

    @staticmethod
    def change_temp(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy(deep=True)
        for c in df.columns:
            if c.endswith('_k'):
                if c == 'reanalysis_tdtr_k':
                    df[c.replace('_k', '_c')] = df[c]
                else:
                    df[c.replace('_k', '_c')] = df[c] - 273.15
                df = df.drop(columns=[c])
        return df

    @staticmethod
    def apply_imputation(data: pd.DataFrame) -> pd.DataFrame:
        return data.ffill()
        # return data.dropna(how='any')

    @staticmethod
    def coarse_grain(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'temp_avg': data[[
                'station_avg_temp_c',
                'reanalysis_air_temp_c',
                'reanalysis_avg_temp_c'
            ]].mean(axis=1),
            'temp_min': data[[
                'station_min_temp_c',
                'reanalysis_min_air_temp_c'
            ]].mean(axis=1),
            'temp_max': data[[
                'station_max_temp_c',
                'reanalysis_max_air_temp_c'
            ]].mean(axis=1),
            'temp_dew': data['reanalysis_dew_point_temp_c'],
            'temp_rng': data[[
                'station_diur_temp_rng_c',
                'reanalysis_tdtr_c'
            ]].mean(axis=1),
            'precipitation': data[[
                'station_precip_mm',
                'precipitation_amt_mm',
                'reanalysis_sat_precip_amt_mm',
                'reanalysis_precip_amt_kg_per_m2'
            ]].mean(axis=1),
            'humidity_pct': data['reanalysis_relative_humidity_percent'],
            'humidity_spc': data['reanalysis_specific_humidity_g_per_kg'],
            'ndvi_ne': data['ndvi_ne'],
            'ndvi_nw': data['ndvi_nw'],
            'ndvi_se': data['ndvi_se'],
            'ndvi_sw': data['ndvi_sw'],
            'weekofyear': data['weekofyear'],
            'is_sj': (data['city'] == 'sj') * 1,
        })

    @staticmethod
    def get_climate_cols() -> List[str]:
        return [
            'temp_avg',
            'temp_min',
            'temp_max',
            'temp_dew',
            'temp_rng',
            'precipitation',
            'humidity_pct',
            'humidity_spc',
            'ndvi_ne',
            'ndvi_nw',
            'ndvi_se',
            'ndvi_sw',
        ]

    @staticmethod
    def get_target_col() -> str:
        return 'total_cases'

    def get_climate_and_target_cols(self) -> List[str]:
        return self.get_climate_cols() + [self.get_target_col()]

    @staticmethod
    def get_shifted(data: pd.DataFrame) -> pd.DataFrame:
        lags = {
            'temp_avg': 8,
            'temp_min': 8,
            'temp_max': 8,
            'temp_dew': 8,
            'temp_rng': 8,
            'precipitation': 3,
            'humidity_pct': 3,
            'humidity_spc': 3,
            'ndvi_ne': None,
            'ndvi_nw': None,
            'ndvi_se': None,
            'ndvi_sw': None,
        }
        df = data.copy(deep=True)
        for c, lag in lags.items():
            if lag:
                df[f'{c}_lag{lag}'] = data[c].shift(lag)
        return df


class Seasonality:
    def __init__(self, data: pd.DataFrame, cols: List[str]):
        self.data = data
        self.cols = cols

    def get_pattern(self, col: str) -> pd.Series:
        df = pd.DataFrame({
            col: self.data[col].diff(),
            'weekofyear': self.data['weekofyear'],
        })
        pattern_diff = df.groupby('weekofyear')[col].mean()
        return (pattern_diff.cumsum() + pd.Series(0, index=range(1, 53+1))).ffill()

    def get_component(self, col: str, data: Optional[pd.DataFrame] = None) -> pd.Series:
        data = self.data.copy(deep=True) if data is None else data
        pattern = self.get_pattern(col)
        initial = self.data[self.data['weekofyear'] == 1][col].mean()
        comp = pattern + initial
        return pd.Series(comp[data['weekofyear']].values, index=data.index)

    def get_residual(self, col: str, data: Optional[pd.DataFrame] = None) -> pd.Series:
        data = self.data.copy(deep=True) if data is None else data
        return data[col] - self.get_component(data=data, col=col)

    def get_component_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        data = self.data.copy(deep=True) if data is None else data
        df = data.copy(deep=True)
        for c in self.cols:
            df[c] = self.get_component(c, df)
        return df

    def get_residualized_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        data = self.data.copy(deep=True) if data is None else data
        df = data.copy(deep=True)
        for c in self.cols:
            df[c] = self.get_residual(c, df)
        return df


class Deseasoner:
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.sj_season = None
        self.iq_season = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        xy = pd.concat([x, y], axis=1)
        self.sj_season = Seasonality(xy[xy.is_sj == 1], cols=self.cols)
        self.iq_season = Seasonality(xy[xy.is_sj == 0], cols=self.cols)
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        xy = pd.concat([x, y], axis=1)
        xy_t = pd.concat([
            self.sj_season.get_residualized_data(xy[xy.is_sj == 1]),
            self.iq_season.get_residualized_data(xy[xy.is_sj == 0]),
        ], axis=0)
        return xy_t[x.columns], xy_t[y.name]

    def get_component(self, data: pd.Series) -> pd.Series:
        return pd.concat([
            self.sj_season.get_component('total_cases', data[data.is_sj == 1]),
            self.iq_season.get_component('total_cases', data[data.is_sj == 0]),
        ])


features = [
    # Vegetation index
    'ndvi_ne',
    'ndvi_nw',
    'ndvi_se',
    'ndvi_sw',

    # Precipitation
    'station_precip_mm',
    'precipitation_amt_mm',
    'reanalysis_sat_precip_amt_mm',
    'reanalysis_precip_amt_kg_per_m2',

    # Humidity
    'reanalysis_relative_humidity_percent',
    'reanalysis_specific_humidity_g_per_kg',

    # Temperature
    # max
    'station_max_temp_c',
    'reanalysis_max_air_temp_c',
    # min
    'station_min_temp_c',
    'reanalysis_min_air_temp_c',
    # avg
    'station_avg_temp_c',
    'reanalysis_avg_temp_c',
    # diurnal temp range = temp max - temp min
    'station_diur_temp_rng_c',
    'reanalysis_tdtr_c',
    # same as avg?
    'reanalysis_air_temp_c',
    # dew point
    'reanalysis_dew_point_temp_c',
]
