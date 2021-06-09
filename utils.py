import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from typing import Optional, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

class ClassifierGSCV(GridSearchCV):
    """
    Custom implementation of a random forest classifier with grid search and cross validation.
    This implementation currently only supports the parameters described below, as you might know
    there are multiple more parameters for the RandomForestClassifier but I will leave them out
    for the sake of simplicity and computational time of the grid search.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)

    Y : array-like of shape (n_samples, 1)

    n_estimators : list of ints, default = [100]
        List of the numbers of trees in the forest for grid search

    criterion : list containing: "gini" or/and "entropy", default= ["gini"]
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_depth : list, default=[None]
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        2 samples - (default value for min_samples_split of the sklearn RandomForestClassifier)

    """
    def __init__(self,
                 X: Optional[np.array] = None,
                 Y: Optional[np.array] = None,
                 n_estimators: List[int] = [100], 
                 max_depth : List[Optional[int]] = [None],
                 criterion: List[Literal['gini', 'entropy']] = ['gini'],
                 simple: bool = False
                ) -> None:
        
        if not simple:
            if (X is None):
                raise ValueError("""Missing X input either pass it as a np.array or 
                initialize using ClassifierGSCV.from_data(pd.DataFrame)""")
            
            if (Y is None):
                raise ValueError("""Missing Y input either pass it as a np.array or 
                initialize using ClassifierGSCV.from_data(pd.DataFrame)""")        
        
        self.X = X
        self.Y = Y
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        
        self.parameters = {'n_estimators':self.n_estimators, 
                           'max_depth':self.max_depth,
                           'criterion':self.criterion}
        
        self.classifier = RandomForestClassifier()
        
        super().__init__(self.classifier, self.parameters)
        
        return None
    
    @classmethod
    def from_data(cls, data: pd.DataFrame, **kwargs):
        """
        Initialize the Classifier from a pandas dataframe - Static structure if new features arise
        This would need to be updated
        Getting training features - always getting all the features
        Since we are using random forests, each estimator will have a different combination of features
        Therefore there is no need for testing multiple feature combinations
        """
        X = data.iloc[:,0:5].values

        """
        Getting clean Y label
        """
        Y = data.iloc[:,6].values


        return cls(X, Y, **kwargs)


    def fit_classifier(self) -> None:
        """
        Process and fit the data already in the right format
        We will save here the label mapping so we can deal with new labels when predicting
        Using our main class GridSearchCV.fit()
        It will do a 5 cross-fold validation as default.
        """

        le = preprocessing.LabelEncoder()
        le.fit(self.Y)
        self.y_map = dict(zip(le.classes_, le.transform(le.classes_)))
        self._cats = [k for k, _ in self.y_map.items()]

        self.fit(self.X, self.Y)

        """
        Save the best estimator chosen by the search.
        The decision is based on the default RandomForestClassifier Score function:
        Return the mean accuracy on the given test data and labels
        """
        filehandler = open('model.pickle', 'wb') 
        pickle.dump(self.best_estimator_, filehandler)

        return None

    def check_results(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing the performance metrics such
        as split and average for each parameter search combination.
        """
        return pd.DataFrame(self.cv_results_)


    def predict_classifier(self, X = np.array, local: bool = True) -> pd.DataFrame:

        """
        Loads our model previously saved as model.pickle,
        and predicts for new data, it can do batch prediction and also single row prediction.
        """

        if local:
            self = pickle.load(open('model.pickle', 'rb'))

        try:
            pred = self.predict_proba(X)
        except ValueError:
            pred = self.predict_proba(X.reshape(1, -1))
        except Exception as e:
            raise e
        
        return pd.DataFrame(pred, columns = self.classes_)

# data = pd.read_excel("toxicity_xls.xlsx", engine = "openpyxl", index_col=0)
# clsf = ClassifierGSCV.from_data(data, n_estimators = [10])
# clsf.fit_classifier()
# clsf.predict_classifier(data.iloc[0:5,0:5].values)