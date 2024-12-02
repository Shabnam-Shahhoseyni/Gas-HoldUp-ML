# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:13:47 2024
REV 03 (7th Oct 2024)
Predictive model for gas hold up in bubble columns using machine learning
@author: shabnam Shahhoseyni
"""

# Standard library imports
import time
import numpy as np
import matplotlib.pyplot as plt

# Third-party library imports
import pandas as pd
import xgboost as xgb

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    validation_curve,
    learning_curve,
)
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.linear_model import (
    Ridge,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingRegressor,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    DotProduct,
    WhiteKernel,
    ConstantKernel,
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
#------------------------------------------------------------------------------

class PlotConfig:
    def __init__(self, font_family='Arial', font_size=14, dpi=300):
        self.settings = {
            'font.family': font_family,
            'font.size': font_size,
            'axes.titlesize': font_size,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size,
            'figure.dpi': dpi
        }

    def apply_settings(self):
        plt.rcParams.update(self.settings)
        
#------------------------------------------------------------------------------

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self, sheet_name):
        """Load and shuffle data from an Excel sheet."""
        df = pd.read_excel(self.file_path, sheet_name)
        df = df.sample(frac=1, random_state=42)  # Shuffle the DataFrame
        
        # Feature selection
        features = [
            'Density of Liquid (kg/m3)',
            'Viscosity of Liquid (Pa.s)',
            'Surface Tension of Liquid (N/m)',
            'Column diameter(m)',
            'Liquid height (m)',
            'Superficial gas velocity (m/s)',
            'Sparger hole diameter (m)',
            'Density of Gas (kg/m3)',
            'Viscosity of Gas (Pa.s)'
        ]
        
        # Extract features and target variable
        X = df[features]
        y = df['Gas holdup'].values.ravel()
        
        return X, y
    
#--------------------------------------------------------------------------------

class ModelEvaluator:
    def __init__(self):
        self.cv_results_all = []

    def plot_learning_curve(self, model, X, y, model_name=None):
        """Plot the learning curve for a given model."""
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(model, X, y, cv=10, scoring='neg_mean_absolute_error', 
                           n_jobs=6, train_sizes=np.linspace(0.1, 1.0, 10),
                           return_times=True)

        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(7, 7), dpi=300)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, '.-', color="r", label="Training error")
        plt.plot(train_sizes, test_scores_mean, '.-', color="g", label="Test error")

        title = "Learning Curve"
        if model_name:
            title += f" - {model_name}"
        plt.title(title)
        plt.xlabel("Number of training samples in the training set")
        plt.ylabel("Mean Absolute Error")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_cross_validation(self, model, X_train, y_train, label, marker, color):
        """Plot cross-validation results for a model."""
        cv_results = cross_validate(model, X_train, y_train, cv=10,
                                    return_estimator=True,
                                    scoring='neg_mean_absolute_error')
        cv_results = pd.DataFrame(cv_results)

        print(f"{label}:\n{cv_results['test_score'].mean() * -1:.3f} ± {cv_results['test_score'].std():.3f}")

        self.cv_results_all.append(cv_results)

        plt.scatter(np.linspace(1, 10, num=10), cv_results["test_score"] * -1, marker=marker,
                    color=color, s=20, label=f"{label}: ({cv_results['test_score'].mean() * -1:.4f} ± {cv_results['test_score'].std():.5f})")
        plt.xlabel('Cross validation folds')
        plt.ylabel('Mean Absolute Error')
        plt.title("Cross Validation Error")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        return cv_results

    def plot_validation_curve(self, estimator, X, y, param_name, param_range, 
                              model_name="", ylim=None, cv=10, n_jobs=7):
        """Plot the validation curve for a model."""
        plt.figure(figsize=(7, 7))
        train_scores, test_scores = validation_curve(
            estimator, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring="neg_mean_absolute_error", n_jobs=n_jobs)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if ylim is not None:
            plt.ylim(*ylim)

        plt.xlabel(param_name)
        plt.ylabel("Mean Absolute Error")
        plt.title(f"Validation Curve ({model_name})")
        lw = 2

        plt.semilogx(param_range, -train_scores_mean, label="Training Error", color="darkorange", lw=lw)
        plt.fill_between(param_range, -train_scores_mean - train_scores_std,
                         -train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)

        plt.semilogx(param_range, -test_scores_mean, label="Test Error", color="navy", lw=lw)
        plt.fill_between(param_range, -test_scores_mean - test_scores_std,
                         -test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)

        plt.legend(loc="best")
        plt.show()

    def nested_grid_search(self, model, X, y, X_train, y_train, param_grid):
        """Perform a nested grid search for hyperparameter tuning."""
        
        model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=7, cv=10, 
                                          scoring=['neg_mean_absolute_error'], refit='neg_mean_absolute_error')
        
        nested_cv_results = cross_validate(model_grid_search, X, y, cv=10, return_estimator=True, 
                                            scoring='neg_mean_absolute_error')
        
        plt.scatter(np.linspace(1, 10, num=10), nested_cv_results["test_score"] * -1, 
                    marker='*', label=f"{model.__class__.__name__}: ({nested_cv_results['test_score'].mean() * -1:.3f} ± {nested_cv_results['test_score'].std():.3f})")
        print(f"Nested Cross-validation: ({nested_cv_results['test_score'].mean() * -1:.3f} ± {nested_cv_results['test_score'].std():.3f})")
        plt.legend()
        
        # Fit the grid search model and get the best parameters
        model_grid_search.fit(X_train, y_train)
        
        # Access the best parameters and the best estimator
        best_params = model_grid_search.best_params_
        best_model = model_grid_search.best_estimator_
        
        print(f"Best parameters: {best_params}")
        print(f"Best model: {best_model}")
        
        return model_grid_search, nested_cv_results

    def evaluate_best_model(self, best_model, X_test, X_train, y_test, y_train, y, model_name=None):
        """Evaluate the best model and visualize its performance."""
        y_pred = best_model.predict(X_test)
        y_pred_train = best_model.predict(X_train)

        test_MAE = mean_absolute_error(y_test, y_pred)
        test_MSE = mean_squared_error(y_test, y_pred)
        test_R2 = r2_score(y_test, y_pred)

        train_MAE = mean_absolute_error(y_train, y_pred_train)
        train_MSE = mean_squared_error(y_train, y_pred_train)
        train_R2 = r2_score(y_train, y_pred_train)

        plt.figure(figsize=(7, 7))

        plt.scatter(y_test, y_pred, c='r', label='Test Data')
        plt.scatter(y_train, y_pred_train, c='b', label='Train Data')

        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Hold up real')
        plt.ylabel('Hold up predicted')
        plt.title(model_name)
        plt.legend()

        plt.annotate("R2", xy=(0.65, 0.15), xycoords='axes fraction', fontsize=12)
        plt.annotate("MSE", xy=(0.65, 0.10), xycoords='axes fraction', fontsize=12)
        plt.annotate("MAE", xy=(0.65, 0.05), xycoords='axes fraction', fontsize=12)

        plt.annotate("Train", xy=(0.75, 0.2), xycoords='axes fraction', fontsize=12)
        plt.annotate(f"{train_R2:.2f}", xy=(0.75, 0.15), xycoords='axes fraction', fontsize=12)
        plt.annotate(f"{train_MSE:.4f}", xy=(0.75, 0.10), xycoords='axes fraction', fontsize=12)
        plt.annotate(f"{train_MAE:.3f}", xy=(0.75, 0.05), xycoords='axes fraction', fontsize=12)

        plt.annotate("Test", xy=(0.9, 0.2), xycoords='axes fraction', fontsize=12)
        plt.annotate(f"{test_R2:.2f}", xy=(0.9, 0.15), xycoords='axes fraction', fontsize=12)
        plt.annotate(f"{test_MSE:.4f}", xy=(0.9, 0.10), xycoords='axes fraction', fontsize=12)
        plt.annotate(f"{test_MAE:.3f}", xy=(0.9, 0.05), xycoords='axes fraction', fontsize=12)

        plt.show()
#------------------------------------------------------------------------------
class PerturbationFeatureImportance:
    def __init__(self, model, X_test, y_test, metric=mean_squared_error):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.baseline_performance = metric(y_test, model.predict(X_test))

    def calculate_importance(self):
        return {
            feature: self.metric(self.y_test, self.model.predict(self._perturb(feature))) - self.baseline_performance
            for feature in self.X_test.columns
        }

    def _perturb(self, feature):
        X_test_perturbed = self.X_test.copy()
        X_test_perturbed[feature] = np.random.permutation(X_test_perturbed[feature])
        return X_test_perturbed

    def plot_importance(self):
        importances = sorted(self.calculate_importance().items(), key=lambda x: abs(x[1]), reverse=True)
        plt.barh([x[0] for x in importances], [x[1] for x in importances], color='blue')
        plt.gca().invert_yaxis()
        plt.xlabel('Increase in MSE')
        plt.ylabel('Features')
        plt.title('Perturbation Feature Importance')
        plt.show()

#------------------------------------------------------------------------------
def print_ridge_coefficients(model, model_name, feature_names):
    """Print coefficients of the Ridge regression model."""
    ridge_coeffs = model.named_steps['ridge'].coef_
    print(f"Coefficients for {model_name} model:")
    for feature, coeff in zip(feature_names, ridge_coeffs):
        print(f"Feature: {feature}, Coefficient: {coeff:.4f}")

def print_transformed_features(model, X_train):
    """Print transformed features from PolynomialFeatures."""
    poly_features = model.named_steps['polynomialfeatures']
    X_train_poly = poly_features.transform(X_train)
    print("\nTransformed features from PolynomialFeatures:")
    print(X_train_poly[:5])  # Display the first 5 rows
    
# -----------------------------------------------------------------------------

def model_validation(model, X, y):
    y_pred=model.predict(X)
        
    MAE = mean_absolute_error(y, y_pred)
    # MSE = mean_squared_error(y, y_pred)#, squared=False)
    # R2 = r2_score(y, y_pred)

    plt.figure (figsize=(7, 7))
    plt.scatter(y, y_pred, c='g', label='validation Data')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Hold up real')
    plt.ylabel('Hold up predicted')
    plt.title('Model performance on validation set')
    plt.legend()
    plt.annotate("MAE", xy=(0.7, 0.05), xycoords='axes fraction')
    plt.annotate(f"{MAE:.3f}", xy=(0.8, 0.05), xycoords='axes fraction')
    plt.show()

def main():
    
    plot_config = PlotConfig()
    plot_config.apply_settings()
    
    
    file_path = 'Data Sheet for Gas Holdup-2.xlsx'
    sheet_name = 'Data For prediction of eG'
    data_loader = DataLoader(file_path)
    X, y = data_loader.load_data(sheet_name)
    
    # Test-Train Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
    
    evaluator = ModelEvaluator()
    
    # # Models
    # LR_model = make_pipeline(StandardScaler(), Ridge())
    # LR_model_FE = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())
    # #LR_ensemble = BaggingRegressor(LR_model_FE)#, random_state=42)
    # SVM_model = make_pipeline(StandardScaler(), SVR())
    # # SVM_model_FE = make_pipeline(StandardScaler(),PolynomialFeatures(), SVR())
    # DT_model= DecisionTreeRegressor()
    # # DT_model_FE= make_pipeline(PolynomialFeatures(), DecisionTreeRegressor())
    # RF_model= RandomForestRegressor()
    # # RF_model_FE= make_pipeline(PolynomialFeatures(),RandomForestRegressor())
    # XG_model= xgb.XGBRegressor()
    # # XG_model_FE= make_pipeline(PolynomialFeatures(), xgb.XGBRegressor())
    
    # GPR_model= GaussianProcessRegressor()
    # # GPR_model_FE= make_pipeline(PolynomialFeatures(), GaussianProcessRegressor())
    
    # ANN_model = make_pipeline(StandardScaler(), MLPRegressor())
    # # ANN_model_FE = make_pipeline(StandardScaler(), PolynomialFeatures(), MLPRegressor())
    # ANN_ensemble = BaggingRegressor(ANN_model)
   
    
   # models------------------------------------------------------------------------------------
    LR_model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    LR_model_FE = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), Ridge(alpha=10))
    
    # LR_ensemble = BaggingRegressor(LR_model_FE, n_estimators=30)
    SVM_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, gamma=1))
    
    # SVM_model_FE = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), SVR(kernel='rbf', C=10, gamma=1))
    DT_model= DecisionTreeRegressor(max_depth=10)#criterion='absolute_error', splitter= 'random')
    
    # DT_model_FE= make_pipeline(PolynomialFeatures(degree=2), DecisionTreeRegressor(max_depth=24))
    RF_model= RandomForestRegressor(max_depth=10, n_estimators= 20)
    
    # RF_model_FE= make_pipeline(PolynomialFeatures(degree=3),RandomForestRegressor(max_depth=16, n_estimators= 41))
    XG_model= xgb.XGBRegressor(max_depth=5, n_estimators= 50)
                          
    # XG_model_FE= make_pipeline(PolynomialFeatures(degree=1), xgb.XGBRegressor(max_depth=5, n_estimators= 90))
    GPR_model= GaussianProcessRegressor(alpha=0.001, kernel=Matern(length_scale=10, nu=1.5))
    
    # GPR_model_FE= make_pipeline(PolynomialFeatures(degree=1), GaussianProcessRegressor())
    ANN_model = make_pipeline(StandardScaler(), MLPRegressor(activation='relu', alpha=0.01,
                  hidden_layer_sizes=(80), solver='lbfgs', max_iter=3000))
    
    # ANN_model_FE = make_pipeline(StandardScaler(), PolynomialFeatures(degree=1), MLPRegressor(activation='relu', alpha=0.1,
                  # hidden_layer_sizes=(140), solver='lbfgs', max_iter=3000))
    # LR_ensemble = BaggingRegressor(LR_model_FE, n_estimators=30)
    ANN_ensemble = BaggingRegressor(ANN_model, n_estimators=30)
    
    
   # models fitting------------------------------------------------------------
    models = {
        'LR_model': LR_model,
        'LR_model_FE': LR_model_FE,
        'SVM_model': SVM_model,
        'DT_model': DT_model,
        'RF_model': RF_model,
        'XG_model': XG_model,
        'GPR_model': GPR_model,
        'ANN_model': ANN_model,
        'ANN_ensemble': ANN_ensemble,
        # 'SVM_model_FE': SVM_model_FE,
        # 'DT_model_FE': DT_model_FE,
        # 'RF_model_FE': RF_model_FE,
        # 'XG_model_FE': XG_model_FE,
        # 'ANN_model_FE': ANN_model_FE,
        # 'GPR_model_FE': GPR_model_FE        
    }
    
    # Initialize a dictionary to store model times
    model_times = {}
    
    # Fit models and store times
    for model_name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        model_times[model_name] = time.time() - start
    
       
    df = pd.DataFrame(list(model_times.items()), columns=['Model', 'Time (seconds)'])
    print(df)

    # feature importance-------------------------------------------------------
    analyzer = PerturbationFeatureImportance(ANN_ensemble, X_test, y_test)
    analyzer.plot_importance()

        
    # Hyperparameter tuning====================================================
    #LR_Ridge------------------------------------------------------------------
    # plt.figure(figsize=(7, 7))
    # print ('LR')
    # param_grid = {'ridge__alpha': np.logspace(-3, 1, 5)}
    # model_grid_searchLR, nested_cv_results= evaluator.nested_grid_search(LR_model, X, y, X_train, y_train, param_grid)

    # print ('LR_FE')
    # param_grid = {
    # 'polynomialfeatures__degree': [1, 2, 3, 4], 
    # 'ridge__alpha': np.logspace(1, 5, 7)}
    # model_grid_searchLRFE, nested_cv_results= evaluator.nested_grid_search(LR_model_FE, X, y, X_train, y_train, param_grid)

    # print ('LR_ensemble')
    # param_grid = {'n_estimators' : np.arange (10, 40, 2)}
    # model_grid_searchLRensemble, nested_cv_results= evaluator.nested_grid_search(LR_ensemble, X, y, X_train, y_train,param_grid)
        
    #SVR-----------------------------------------------------------------------
    # print ('SVR')
    # param_grid = {
    # 'svr__C': np.logspace(-5, 2, 8),
    # 'svr__kernel': ['rbf'],
    # 'svr__gamma': np.logspace(-5, 2, 8)}
    # model_grid_searchSVR, nested_cv_results= evaluator.nested_grid_search(SVM_model, X, y, X_train, y_train, param_grid)

    # print ('SVR_FE')
    # param_grid = {
    # 'svr__C': np.logspace(-6, 4, 11),
    # 'svr__kernel': ['rbf'],
    # 'polynomialfeatures__degree': [1],
    # 'svr__gamma': np.logspace(-6, 3, 10)}
    # model_grid_searchSVRFE, nested_cv_results= evaluator.nested_grid_search(SVM_model_FE, X, y, X_train, y_train, param_grid)
 
    #DT-----------------------------------------------------------------------
    # print ('DT')
    # param_grid = {'max_depth': np.arange(1, 11, 1)}
    # model_grid_searchDT, nested_cv_results= evaluator.nested_grid_search(DT_model, X, y, X_train, y_train, param_grid)
    
    # print ('DT_FE')
    # param_grid = {
    # 'decisiontreeregressor__max_depth': np.arange(20, 40, 4),
    # 'polynomialfeatures__degree': [1, 2, 3]}
    # model_grid_searchDTFE, nested_cv_results= evaluator.nested_grid_search(DT_model_FE, X, y, X_train, y_train, param_grid)

    # RF-----------------------------------------------------------------------
    # print ('RF')
    # param_grid = {
    # 'n_estimators' : np.arange (1, 12, 2),
    # 'max_depth': np.arange(1, 11, 1),
    # #'min_samples_split': [2, 5, 10, 50],
    # #'min_samples_leaf': [1, 2, 4, 10],
    # #'max_leaf_nodes': [None, 5, 10, 20, 30, 50]}
    # model_grid_searchRF, nested_cv_results= evaluator.nested_grid_search(RF_model, X, y, X_train, y_train, param_grid)
    
    # print ('RF_FE')
    # param_grid = {
    # 'polynomialfeatures__degree': [1, 2, 3],
    # 'randomforestregressor__n_estimators' : np.arange (20, 60, 10),
    # 'randomforestregressor__max_depth': np.arange(0, 30, 5)
    # #'min_samples_split': [2, 5, 10, 50],
    # #'min_samples_leaf': [1, 2, 4, 10],
    # #'max_leaf_nodes': [None, 5, 10, 20, 30, 50]}
    # model_grid_searchRFFE, nested_cv_results= evaluator.nested_grid_search(RF_model_FE, X, y, X_train, y_train, param_grid)
    
    #XG-----------------------------------------------------------------------
    # print ('XG')
    # param_grid = {
    #     'n_estimators': np.arange (1, 14, 1),
    #     'max_depth': np.arange(1, 6, 1),
    #     #'reg_lambda': np.arange (0, 1, 4)
    #     #'learning_rate': [0.01, 0.1, 0.2],
    #     #'subsample': [0.8, 1.0],
    #     #'colsample_bytree': [0.8, 1.0],
    #     #'gamma': [0, 0.1, 0.2],
    #     #'min_child_weight': [1, 3, 5]
    # }
    # model_grid_searchXG, nested_cv_results= evaluator.nested_grid_search(XG_model, X, y, X_train, y_train, param_grid)
    
    # print ('XG_FE')
    # param_grid = {
    #     'xgbregressor__n_estimators': np.arange (1, 100, 10),
    #     'xgbregressor__max_depth': np.arange(1, 31, 5),
    #     #'learning_rate': [0.01, 0.1, 0.2],
    #     #'subsample': [0.8, 1.0],
    #     #'colsample_bytree': [0.8, 1.0],
    #     #'gamma': [0, 0.1, 0.2],
    #     #'min_child_weight': [1, 3, 5]
    # }
    # model_grid_searchXGFE= evaluator.nested_grid_search(XG_model_FE, X, y, X_train, y_train, param_grid)

    #GPR-----------------------------------------------------------------------
    # print ('GPR')
    # param_grid = {#'kernel': [RBF(), ConstantKernel(), Matern(), DotProduct(), WhiteKernel()],
    #               'kernel': [Matern()],#length_scale=l, nu=n) for l in np.logspace(-1, 1, 3) for n in [1.5, 2.5]],
    #               'alpha': np.logspace(-3,4, 8)}
    # model_grid_searchGPR, nested_cv_results= evaluator.nested_grid_search(GPR_model, X, y, X_train, y_train, param_grid)
    
    # print ('GPRFE')
    # param_grid = {'gaussianprocessregressor__kernel': [RBF(), ConstantKernel(), Matern(), DotProduct(), WhiteKernel()],
    #               'gaussianprocessregressor__alpha': np.logspace(-3,3, 7)}
    # model_grid_searchGPRFE, nested_cv_results= evaluator.nested_grid_search(GPR_model_FE, X, y, X_train, y_train, param_grid)
    
    #ANN-----------------------------------------------------------------------
    # print ('ANN')
    # param_grid = {
    #     'mlpregressor__hidden_layer_sizes': np.arange (1, 301, 20),#[(n,) for n in range(100)],#+ [(n1, n2) for n1 in range(10) for n2 in range(10) ] + [(n1, n2, n3) for n1 in range(10) for n2 in range(10) for n3 in range(10)],                
    #     'mlpregressor__activation': ['tanh','relu'],
    #     'mlpregressor__solver': ['lbfgs', 'adam', 'sgd'],
    #     'mlpregressor__alpha': np.logspace(-2, 2, 5),
    #     #'mlpregressor__learning_rate': ['constant', 'adaptive'],
    #     #'mlpregressor__learning_rate_init': [ 0.01, 0.1, 0.2],
    #     'mlpregressor__max_iter': [30000]
    # }
    # model_grid_searchANN, nested_cv_results= evaluator.nested_grid_search(ANN_model, X, y, X_train, y_train, param_grid)


    # print ('ANN_FE')
    # param_grid = {
    #     'mlpregressor__hidden_layer_sizes': np.arange (20, 150, 20),#[(n,) for n in range(100)],#+ [(n1, n2) for n1 in range(10) for n2 in range(10) ] + [(n1, n2, n3) for n1 in range(10) for n2 in range(10) for n3 in range(10)],                
    #     'mlpregressor__activation': ['tanh','relu'],
    #     'mlpregressor__solver': ['lbfgs', 'adam', 'sgd'],
    #     'mlpregressor__alpha': np.logspace(-2, 1, 4),
    #     #'mlpregressor__learning_rate': ['constant', 'adaptive'],
    #     #'mlpregressor__learning_rate_init': [ 0.01, 0.1, 0.2],
    #     'mlpregressor__max_iter': [30000]
    # }
    # model_grid_searchANNFE, nested_cv_results= evaluator.nested_grid_search(ANN_model_FE, X, y, X_train, y_train, param_grid)

    
    # print ('ANN_ensemble')
    # param_grid = {'n_estimators': np.arange (1, 31, 5)}
    # model_grid_searchANNensemble, nested_cv_results= evaluator.nested_grid_search(ANN_ensemble, X, y, X_train, y_train, param_grid)


    # plt.xlabel('Validation folds')
    # plt.ylabel('MAE')
    # plt.title('Nested cross validation')
    
    # cross validation results in each fold-----------------------------------------------
        
    print ('Cross_validation')
    
    plt.figure(figsize=(7, 14))
    
    evaluator.plot_cross_validation(LR_model, X_train, y_train, label="LR", marker='s', color='orange')
    evaluator.plot_cross_validation(LR_model_FE, X_train, y_train, label="LR-FE" , marker='s', color='r')
    evaluator.plot_cross_validation(SVM_model, X_train, y_train, label="SVR", marker='^', color='purple')
    # evaluator.plot_cross_validation(SVM_model_FE, X_train, y_train, label="SVR_FE", marker='^', color='magenta')
    evaluator.plot_cross_validation(DT_model, X_train, y_train, label="DT", marker='v', color='g')
    # evaluator.plot_cross_validation(DT_model_FE, X_train, y_train, label="DT_FE", marker='v', color='lime')
    evaluator.plot_cross_validation(RF_model, X_train, y_train, label="RF", marker='o', color='magenta')
    # evaluator.plot_cross_validation(RF_model_FE, X_train, y_train, label="RF_FE", marker='o', color='c')
    evaluator.plot_cross_validation(XG_model, X_train, y_train, label="XG", marker='d', color='crimson')
    # evaluator.plot_cross_validation(XG_model_FE, X_train, y_train, label="XG_FE", marker='d', color='deeppink')
    evaluator.plot_cross_validation(GPR_model, X_train, y_train, label="GPR", marker='*', color='darkcyan')
    # evaluator.plot_cross_validation(GPR_model_FE, X_train, y_train, label="GPR_FE", marker='*', color='lightseagreen')
    evaluator.plot_cross_validation(ANN_model, X_train, y_train, label="ANN", marker='P', color='b')
    evaluator.plot_cross_validation(ANN_ensemble, X_train, y_train, label="ANN_ensemble", marker='P', color='lightseagreen')
    # evaluator.plot_cross_validation(ANN_model_FE, X_train, y_train, label="ANN_FE", marker='P', color='grey')
    
    
     
    #Learning Curves---------------------------------------------------------------
    print ('Learning Curves')
    # plt.figure(figsize=(7, 7))
    evaluator.plot_learning_curve(LR_model, X, y, model_name="LR")
    evaluator.plot_learning_curve(LR_model_FE, X, y, model_name="LR_FE")
    evaluator.plot_learning_curve(SVM_model, X, y, model_name="SVR")
    # evaluator.plot_learning_curve(SVM_model_FE, X, y, model_name="SVR_FE")
    evaluator.plot_learning_curve(DT_model, X, y, model_name=" DT")
    # evaluator.plot_learning_curve(DT_model_FE, X, y, model_name=" DT_FE")
    evaluator.plot_learning_curve(RF_model, X, y, model_name=" RF")
    # evaluator.plot_learning_curve(RF_model_FE, X, y, model_name=" RF_FE")
    evaluator.plot_learning_curve(XG_model, X, y, model_name=" XG")
    # evaluator.plot_learning_curve(XG_model_FE, X, y, model_name=" XG_FE")
    evaluator.plot_learning_curve(ANN_model, X, y, model_name=" ANN")
    # evaluator.plot_learning_curve(ANN_model_FE, X, y, model_name=" ANN_FE")
    evaluator.plot_learning_curve(ANN_ensemble, X, y, model_name=" ANN_ensemble")
    evaluator.plot_learning_curve(GPR_model, X, y, model_name=" GPR")
    # evaluator.plot_learning_curve(GPR_model_FE, X, y, model_name=" GPR_FE")
    
    # Plot models performance-----------------------------------------------------------    
    print('Models')
    evaluator.evaluate_best_model(LR_model, X_test, X_train, y_test, y_train, y, model_name="LR")
    evaluator.evaluate_best_model(LR_model_FE, X_test, X_train, y_test, y_train, y, model_name="LR_FE")
    evaluator.evaluate_best_model(SVM_model, X_test, X_train, y_test, y_train, y, model_name="SVR")
    #evaluator.evaluate_best_model(SVM_model_FE, X_test, X_train, y_test, y_train, y, model_name="SVR-FE")    
    evaluator.evaluate_best_model(DT_model, X_test, X_train, y_test, y_train, y, model_name="DT")
    #evaluator.evaluate_best_model(DT_model_FE, X_test, X_train, y_test, y_train, y, model_name="DT-FE")
    evaluator.evaluate_best_model(RF_model, X_test, X_train, y_test, y_train, y, model_name="RF")
    #evaluator.(RF_model_FE, X_test, X_train, y_test, y_train, y, model_name="RF-FE")
    evaluator.evaluate_best_model(XG_model, X_test, X_train, y_test, y_train, y, model_name="XG")
    #evaluator.evaluate_best_model(XG_model_FE, X_test, X_train, y_test, y_train, y, model_name="XG-FE")    
    evaluator.evaluate_best_model(GPR_model, X_test, X_train, y_test, y_train, y, model_name="GPR")
    #evaluator.evaluate_best_model(GPR_model_FE, X_test, X_train, y_test, y_train, y, model_name="GPR-FE")
    evaluator.evaluate_best_model(ANN_model, X_test, X_train, y_test, y_train, y, model_name="ANN")
    #evaluator.evaluate_best_model(ANN_model_FE, X_test, X_train, y_test, y_train, y, model_name="ANN-FE")    
    evaluator.evaluate_best_model(ANN_ensemble, X_test, X_train, y_test, y_train, y, model_name="ensembles of ANN") 
      

    # Plot Validation curves---------------------------------------------------
    print('validation curves')
    #LR -----------------------------------
    evaluator.plot_validation_curve(LR_model, X_train, y_train,
                          param_name="ridge__alpha", param_range=np.logspace(-6, 6, 13), model_name="LR")

    evaluator.plot_validation_curve(LR_model_FE, X_train, y_train,
                          param_name="ridge__alpha", param_range=np.logspace(-6, 6, 13), model_name="LR_FE")
    
    evaluator.plot_validation_curve(LR_model_FE, X_train, y_train,
                          param_name="polynomialfeatures__degree", param_range=np.arange(0, 7, 1), model_name="LR_FE")

    #DT------------------------------------
    
    evaluator.plot_validation_curve(DT_model, X_train, y_train,
                          param_name="max_depth", param_range=np.arange(1, 60, 2), model_name="DT")
    
    # evaluator.plot_validation_curve(DT_model_FE, X_train, y_train,
    #                       param_name="decisiontreeregressor__max_depth", param_range=np.arange(2, 50, 2), model_name="DT_FE")
    
    # evaluator.plot_validation_curve(DT_model_FE, X_train, y_train,
    #                       param_name="polynomialfeatures__degree", param_range=np.arange(1, 7, 1), model_name="DT_FE")

    #SVR-----------------------------------
    evaluator.plot_validation_curve(SVM_model, X_train, y_train,
                          param_name="svr__C", param_range=np.logspace(-6, 6, 27), model_name="SVR")
    
    evaluator.plot_validation_curve(SVM_model, X_train, y_train,
                          param_name="svr__gamma", param_range=np.logspace(-6, 6, 27), model_name="SVR")
    
    # evaluator.plot_validation_curve(SVM_model_FE, X_train, y_train,
    #                       param_name="svr__C", param_range=np.logspace(-3, 3, 7), model_name="SVR_FE")
    
    # evaluator.plot_validation_curve(SVM_model_FE, X_train, y_train,
    #                       param_name="svr__gamma", param_range=np.logspace(-3, 3, 7), model_name="SVR_FE")
        
    # evaluator.plot_validation_curve(DT_model_FE, X_train, y_train,
    #                       param_name="polynomialfeatures__degree", param_range=np.arange(1, 7, 1), model_name="SVR_FE")
    
    #RF------------------------------------
    evaluator.plot_validation_curve(RF_model, X_train, y_train,
                          param_name="max_depth", param_range=np.arange(2, 50, 2), model_name="RF")
    
    evaluator.plot_validation_curve(RF_model, X_train, y_train,
                          param_name="n_estimators", param_range=np.arange(1, 100, 10), model_name="RF")
    
    # evaluator.plot_validation_curve(RF_model_FE, X_train, y_train,
    #                       param_name="randomforestregressor__max_depth", param_range=np.arange(1, 50, 5), model_name="RF_FE")

    # evaluator.plot_validation_curve(RF_model_FE, X_train, y_train,
    #                       param_name="randomforestregressor__n_estimators", param_range=np.arange(1, 100, 10), model_name="RF_FE") 
    
    # evaluator.plot_validation_curve(RF_model_FE, X_train, y_train,
    #                       param_name="polynomialfeatures__degree", param_range=np.arange(1, 7, 1), model_name="RF_FE")
    
    #XGBoost---------------------------------

    evaluator.plot_validation_curve(XG_model, X_train, y_train,
                          param_name="max_depth", param_range = np.arange(1, 51, 2), model_name="XG")
    
    # evaluator.plot_validation_curve(XG_model, X_train, y_train,
    #                       param_name="reg_lambda", param_range=[0.001, 0.01, 0.1, 0.2,0.3,0.4,0.5], model_name="XG")
    
    evaluator.plot_validation_curve(XG_model, X_train, y_train,
                          param_name="n_estimators", param_range=np.arange(1, 102, 10), model_name="XG")

    # evaluator.plot_validation_curve(XG_model_FE, X_train, y_train,
    #                       param_name="xgbregressor__max_depth", param_range=np.arange(1, 50, 5), model_name="XG_FE")
    
    # evaluator.plot_validation_curve(XG_model_FE, X_train, y_train,
    #                       param_name="xgbregressor__n_estimators", param_range=np.arange(1, 100, 10), model_name="XG_FE") 
    
    # evaluator.plot_validation_curve(XG_model_FE, X_train, y_train,
    #                       param_name="polynomialfeatures__degree", param_range=np.arange(1, 7, 1), model_name="XG_FE")
    
    #GPR-------------------------------------
    
    evaluator.plot_validation_curve(GPR_model, X_train, y_train,
                          param_name="alpha", param_range=np.logspace(-6,1, 8), model_name="GPR")
    
    # evaluator.plot_validation_curve(GPR_model, X_train, y_train,
                          # param_name="n_restarts_optimizer", param_range=np.arange (1,10, 1), model_name="GPR")
    
    # evaluator.plot_validation_curve(GPR_model_FE, X_train, y_train,
    #                       param_name="gaussianprocessregressor__alpha", param_range=np.logspace(-6,6, 13), model_name="GPR_FE")
    
    # evaluator.plot_validation_curve(GPR_model_FE, X_train, y_train,
    #                       param_name="gaussianprocessregressor__n_restarts_optimizer", param_range=np.arange (1,10, 1), model_name="GPR_FE")
    
    # evaluator.plot_validation_curve(GPR_model_FE, X_train, y_train,
    #                       param_name="polynomialfeatures__degree", param_range=np.arange(1, 7, 1), model_name="GPR_FE")
                
    #ANN--------------------------------------
    
    evaluator.plot_validation_curve(ANN_model, X_train, y_train,
                          param_name="mlpregressor__hidden_layer_sizes", param_range=np.arange(1, 1000, 50), model_name="ANN")
    
    evaluator.plot_validation_curve(ANN_model, X_train, y_train,
                          param_name="mlpregressor__alpha", param_range=np.logspace(-6, 6, 26), model_name="ANN")
    
    # evaluator.plot_validation_curve(ANN_model_FE, X_train, y_train,
    #                       param_name="mlpregressor__hidden_layer_sizes", param_range=np.arange(1, 500, 10), model_name="ANN_FE")
 
    # evaluator.plot_validation_curve(ANN_model_FE, X_train, y_train,
    #                       param_name="mlpregressor__alpha", param_range=np.logspace(-6, 6, 26), model_name="ANN_FE")

    # evaluator.plot_validation_curve(ANN_model_FE, X_train, y_train,
    #                       param_name="polynomialfeatures__degree", param_range=np.arange(1, 7, 1), model_name="ANN_FE")
    
    evaluator.plot_validation_curve(ANN_ensemble, X_train, y_train,
                          param_name="n_estimators", param_range=np.arange(1, 101, 10), model_name="ANN_ensemble")
    
    
    
    # Get feature names from X_train
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Print coefficients for models
    print_ridge_coefficients(LR_model, "Linear Regression (Ridge)", feature_names)
    print("\n")
    print_ridge_coefficients(LR_model_FE, "Linear Regression with Feature Engineering (Ridge)", feature_names)
    print_transformed_features(LR_model_FE, X_train)
        
    # Validate each model------------------------------------------------------   
    data_loader = DataLoader(file_path)
    file_path = 'HydrodynamicsDatasetBubbleColumn.xlsx'
    sheet_name = 'Test'
    X_valid, y_valid = data_loader.load_data(sheet_name)

    for model_name, model in models.items():
        model_validation(model, X_valid, y_valid, model_name=model_name)
        
    # -------------------------------------------------------------------------
    # print (cv_results_all)  
    # file_path = 'outputcrossvalidation.xlsx'
    # df = pd.DataFrame(cv_results_all)
    # df.to_excel(file_path, index=False)

if __name__ == "__main__":
    main()
    
