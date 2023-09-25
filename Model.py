import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import L1
from sklearn.metrics import r2_score

class Features():
    def __init__(self):
        self.dfs = {}
        self.rsquared_values = {}
        self.rsquared_df = {}
    
    def imported_filed(self, directory_path, filesname, cleanup, target):
        self.target_num = np.size(target)
        if filesname:
            if directory_path:
                for self.filename in filesname:
                    file_name = self.filename
                    file_path = os.path.join(directory_path, file_name)
                    
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        self.dfs[file_name] = df
                        self.data_set = pd.concat([self.dfs[self.filename]], axis=0)
                        print(f"{file_name} was imported")
                    else:
                        print(f"File not found for {file_name}.")
                self.data_set = pd.concat(list(self.dfs.values()), axis=0)
        else:
            print('Your directory or filesname is not define !!')
        
        if cleanup == True:
            self.dataclear = self.data_set
            mean_values = self.dataclear.mean()
            columns_to_drop = mean_values.index[(mean_values.isna()) | (mean_values == 0)]
            self.dataclear = self.dataclear.drop(columns=columns_to_drop)
            print('Data was cleaned')
            self.data_set = self.dataclear
            
        if  self.target_num == np.size(np.array(target)):
            target = np.array(target)
            self.target_column = target
            self.target = self.data_set[target]
            print('Your targer is ', self.target_column)
        else:
            print('Yourtarget number are not match with target name')
            
        return self.target, self.target_num
    
    def distribution_plot(self, name, quantile):
        self.dataset = self.data_set
        
        if name == False:
                for column_name in [self.dataset.columns]:
                    q = self.dataset[column_name].quantile(quantile)
                    self.dataset = self.dataset[self.dataset[column_name] < q]
    
        
        else:
                old_dis = sns.distplot(self.dataset[name], color='red')
                for column_name in [self.dataset.columns]:
                    q = self.dataset[column_name].quantile(quantile)
                    self.dataset = self.dataset[self.dataset[column_name] < q]
                new_dis = sns.distplot(self.dataset[name], color='green')
                self.dataset = self.dataset.dropna(axis=0)
                self.data_set = self.dataclear

    def describtion(self):
        describeshow = self.data_set.describe(include='all')
        return describeshow
        
    def VIF(self, threshold):
            self.rsquared_values = {}
            self.variables_to_drop = []

            X = self.data_set.iloc[:, :-1]
            for col in X.columns:
                model_VIF = sm.OLS(X[col], sm.add_constant(X.drop(col, axis=1))).fit()
                self.rsquared_values[col] = model_VIF.rsquared

                if self.rsquared_values[col] <= threshold:
                    self.variables_to_drop.append(col)

            self.data_set = self.data_set.drop(columns=self.variables_to_drop)
            return self.data_set


    def plot_VIF(self):
        X = self.data_set.iloc[:, :-1]
        self.rsquared_values = {}
        
        for col in X.columns:
            model_VIF = sm.OLS(X[col], sm.add_constant(X.drop(col, axis=1))).fit()
            self.rsquared_values[col] = model_VIF.rsquared

        self.rsquared_df = pd.DataFrame.from_dict(self.rsquared_values, orient='index', columns=['VIF'])
        self.rsquared_df.reset_index(inplace=True)
        self.rsquared_df.rename(columns={'index': 'Variable'}, inplace=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="VIF", y="Variable", data=self.rsquared_df, orient="h")
        plt.title("VIF Values for Predictor Variables")
        plt.xlabel("VIF")
        plt.ylabel("Variable")
        self.show = plt.show()
        return self.show,self.data_set
    
    def corr_plot(self):
        self.correlation_matrix = self.data_set.corr()
        plt.figure(figsize=(20, 10))
        sns.heatmap(self.correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()
        

    def corr(self, corr_threshold):
        self.selected_features = []
    
        for target_column_name in self.target_column:
            target_column_for_correlation = self.data_set[target_column_name]  
            
        for feature_name in self.data_set.columns:
            correlation_with_output = np.abs(self.data_set[feature_name].corr(target_column_for_correlation))   
            if correlation_with_output > corr_threshold:
                self.selected_features.append(feature_name) 
                    
        self.selected_features = list(set(self.selected_features))

        self.data_set = self.data_set[self.selected_features]

        return self.data_set

    def get_data(self):
        self.feature_set = self.data_set
        self.nums_outputs = self.target_num
        self.target_set = self.target
        return self.target_set, self.feature_set, self.nums_outputs

class ANN:
    def __init__(self) -> None:
        pass
    
    def train(self, target_set, feature_set, nums_outputs, test_size, hidden_layer_number, hidden_size, optimizer_type,lr, ep, batch_size, activation, loss, metrics):
        self.metrics = metrics
        self.loss = loss
        self.r2_set= []
        self.scaler_input = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_output = MinMaxScaler(feature_range=(-1, 1))

        X_scaled = self.scaler_input.fit_transform(np.array(feature_set))
        y_scaled = self.scaler_output.fit_transform(target_set)


        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        model = Sequential()
        regularizer = tf.keras.regularizers.L1L2()
        model.add(Dense(hidden_size, input_shape=(X_train.shape[1],), kernel_initializer='uniform', activation=activation, kernel_regularizer=regularizer))
        for _ in range(hidden_layer_number):
            model.add(Dense(hidden_size, activation=activation, kernel_regularizer=regularizer))
        model.add(Dense(nums_outputs, activation=None, kernel_regularizer=regularizer))

        if optimizer_type == 'SGD':
            opt = tf.keras.optimizers.SGD(learning_rate=lr)
        elif optimizer_type == 'Adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_type == 'RMSprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer_type == 'AdamW':
            opt = tf.keras.optimizers.AdamW(learning_rate=lr)
        elif optimizer_type == 'Adadelta':
            opt = tf.keras.optimizers.Adadelta(learning_rate=lr)
        elif optimizer_type == 'Adagrad':
            opt = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif optimizer_type == 'Adamax':
            opt = tf.keras.optimizers.Adamax(learning_rate=lr)            
        elif optimizer_type == 'Adafactor':
            opt = tf.keras.optimizers.Adafactor(learning_rate=lr)           
        elif optimizer_type == 'Adafactor':
            opt = tf.keras.optimizers.Adafactor(learning_rate=lr)   
        elif optimizer_type == 'Ftrl':
            opt = tf.keras.optimizers.Ftrl(learning_rate=lr)   
                        
        model.compile(loss=loss, optimizer=opt, metrics=[metrics])
        self.history = model.fit(X_train, y_train, epochs=ep, batch_size=batch_size, shuffle=True, validation_data=(X_val, y_val), verbose=0)
        self.test_loss, self.test_metrics = model.evaluate(X_test, y_test)
        print(f'Test loss: {self.test_loss}, Test {metrics}: {self.test_metrics}')
        i = 0
        
        for i in range(nums_outputs):
            self.y_actual = self.scaler_output.inverse_transform(y_test)[:, i].reshape(-1, 1)
            self.y_pred = self.scaler_output.inverse_transform(model.predict(X_test))[:, i].reshape(-1, 1)
            self.regression_model_y = LinearRegression()
            self.regression_model_y.fit(self.y_actual, self.y_pred)
            self.regression_line_y = self.regression_model_y.predict(self.y_actual)

            #epochs = range(1, len(self.history.history[self.metrics]) + 1)
            #loss = self.history.history['loss']
            #met = self.history.history[self.metrics]
            
            self.r2 = r2_score(self.y_actual, self.y_pred)
            
            target_column_name = target_set.columns[i]
            
           #plt.figure(figsize=(24, 8))
           #
           #plt.subplot(1, 3, 1)
           #plt.plot(epochs, loss, 'b', label=f'Training {self.loss}')
           #plt.title(f'Training {self.loss}')
           #plt.xlabel('Epochs')
           #plt.ylabel(f'{self.loss}')
           #plt.legend()
           #
           #plt.subplot(1, 3, 2)
           #plt.plot(epochs, met, 'g', label=f'Training {self.metrics}')
           #plt.title(f'Training {self.metrics}')
           #plt.xlabel('Epochs')
           #plt.ylabel(f'{self.metrics}')
           #plt.legend()
           #
           #plt.subplot(1, 3, 3)
           #plt.scatter(self.y_actual, self.y_pred, c='r', label='Regression Plot')
           #plt.plot(self.y_actual, self.regression_line_y, 'b-', label='Regression Line for y1')
           #plt.title('Regression Plot (Actual vs. Predicted)')
           #plt.xlabel('Actual Values')
           #plt.ylabel('Predicted Values')
           #plt.legend()
            print(f'R-squared (R²) for {target_column_name}: {self.r2:.4f}')  
            self.r2_set.append(self.r2) 
         
        #plt.tight_layout()
        #plt.show()
        return  model, self.r2_set

    
    
        