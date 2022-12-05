import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    roc_auc_score
)


class PrepareDataAndTrainingModels:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        is_only_numeric_cols: bool = True,
        random_state: int = 42,
        test_size: float = 0.3,
        **kwargs,
    ) -> None:

        self.dataframe = dataframe
        self.target = target
        self.is_only_num_cols = is_only_numeric_cols
        self.random_state = random_state
        self.test_size = test_size
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.fitted_models = None
        self.kwargs = kwargs

    def get_name_cat_and_num_cols(self)->None:
        """
        This method get names of columns categorical and numerical columns.
        """
        if self.is_only_num_cols:
            return (
                type(self.dataframe[self.dataframe.columns].columns),
                self.dataframe[self.dataframe.columns]
                .drop(self.target, axis=1)
                .columns,
            )

        possible_cat_cols = list()
        possible_cat_cols.append(self.target)
        for i, type_ in enumerate(self.dataframe.dtypes):
            if type_ == object:
                possible_cat_cols.append(self.dataframe.iloc[:, i].name)
            elif type_ == pd.CategoricalDtype.name:
                possible_cat_cols.append(self.dataframe.iloc[:, i].name)

        return type(
            (
                self.dataframe[possible_cat_cols].columns,
                self.dataframe.drop(possible_cat_cols, axis=1).columns,
            )
        ), (
            self.dataframe[possible_cat_cols].columns,
            self.dataframe.drop(possible_cat_cols, axis=1).columns,
        )

    def splitting_data(self, **kwargs)->None:
        X = self.dataframe.drop(self.target, axis=1)
        Y = self.dataframe[self.target]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y,random_state=self.random_state, test_size=self.test_size,**kwargs      
        )


    def balance_data(self):
        qt_classes = (
            self.dataframe[self.target]
            .value_counts(ascending=True)
            .reset_index(drop=True)
        )        
        if len(qt_classes) == 2:
            print("Its Binary Classification!\n\n")
            print("#" * 50 + "\n\n")
            if qt_classes[0] / qt_classes[1] < 0.35:
                print(f"Using tecnique: {self.kwargs['balancer']}")
                self.X_train, self.Y_train = self.kwargs["balancer"].fit_resample(
                    self.X_train,
                    self.Y_train,
                )    

    def pre_process_data(self):
        if self.is_only_num_cols:
            self.X_train, self.X_test, self.Y_train, self.Y_test = self.balance_data()
            self.X_train = self.kwargs["scaler"].fit_transform(self.X_train)
            self.X_test = self.kwargs["scaler"].fit_transform(self.X_test)
            

    def fit_models(self)->None:        
        models = self.kwargs["models"]
        fitted_models = dict()      
        for _, model in enumerate(models):            
            predictor = model
            predictor.fit(self.X_train, self.Y_train)
            fitted_models[str(model)] = predictor        
        self.fitted_models=pd.DataFrame(fitted_models).T.reset_index().rename(columns={"index": "model"})


    def fit_models_cv(self)->None:        
        models = self.kwargs["models"]
        fitted_models = dict()      
        for _, model in enumerate(models):            
            predictor = model
            predictor.fit(self.X_train, self.Y_train)
            fitted_models[str(model)] = predictor        
        self.fitted_models=pd.DataFrame(fitted_models).T.reset_index().rename(columns={"index": "model"})


    def predict(self)->pd.DataFrame:        
        models = self.kwargs["models"]
        predictions = dict()           
        for _, model in enumerate(models):            
            predictor = model
            prediction = predictor.predict(self.X_test, self.Y_test)
            predictions[str(model)] = prediction
        
        return (
            pd.DataFrame(prediction)
            .T.reset_index()
            .rename(columns={"index": "model"})
        )                
        
    
    def compute_scores(self, **kwargs)->pd.DataFrame:
        score_models = dict()
        models = self.kwargs["models"]
        for _, model in enumerate(models):
            score_models[str(model)] = dict()
            predictor = model
            prediction = predictor.predict(self.X_test)
            score_models[str(model)]["accuracy"] = accuracy_score(self.Y_test, prediction)
            score_models[str(model)]["precision"] = precision_score(self.Y_test, prediction)
            score_models[str(model)]["recall"] = recall_score(self.Y_test, prediction)
            score_models[str(model)]["f1_score"] = f1_score(self.Y_test, prediction)
            score_models[str(model)]["roc_auc_score"] = roc_auc_score(self.Y_test, prediction)

        return (
            pd.DataFrame(score_models)
            .T.reset_index()
            .rename(columns={"index": "model"})
        )


   

    def avaliate_models_cv(self, **kwargs):
        score_models = dict()
        models = self.kwargs["models"]
        X_train, X_test, Y_train, Y_test = self.pre_process_data()
        for _, model in enumerate(models):
            print(f"Model training: {str(model)}\n\n")
            score_models[str(model)] = dict()
            predictor = model
            predictor.fit(X_train, Y_train)
            prediction = predictor.predict(X_test)
            score_models[str(model)]["accuracy"] = accuracy_score(Y_test, prediction)
            score_models[str(model)]["precision"] = precision_score(Y_test, prediction)
            score_models[str(model)]["recall"] = recall_score(Y_test, prediction)
            score_models[str(model)]["f1_score"] = f1_score(Y_test, prediction)
            scores_ = cross_val_score(
                predictor, X_train, Y_train, cv=5, scoring=self.kwargs["score_metric"]
            )
            print(f"Metric: {self.kwargs['score_metric']} - {scores_.mean()}")
            print(f"#" * 50)
        return (
            pd.DataFrame(score_models)
            .T.reset_index()
            .rename(columns={"index": "model"})
        )
