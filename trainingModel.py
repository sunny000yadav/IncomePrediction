
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
import numpy as np
import pandas as pd


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
    def trainingModel(self):

        self.log_writer.log(self.file_object, 'Start of Training')
        try:

            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()




            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data=preprocessor.remove_columns(data,['education'])
            data=preprocessor.remove_unwanted_spaces(data)
            data.replace('?',np.NaN,inplace=True)



            X,Y=preprocessor.separate_label_feature(data,label_column_name='Income')

            Y = Y.map({'<=50K': 0, '>50K': 1})


            is_null_present,cols_with_missing_values=preprocessor.is_null_present(X)


            if(is_null_present):
                X=preprocessor.impute_missing_values(X,cols_with_missing_values) # missing value imputation


            scaled_num_df=preprocessor.scale_numerical_columns(X)
            cat_df=preprocessor.encode_categorical_columns(X)
            X=pd.concat([scaled_num_df,cat_df], axis=1)


            X,Y=preprocessor.handle_imbalanced_dataset(X,Y)



            kmeans=clustering.KMeansClustering(self.file_object,self.log_writer)
            number_of_clusters=kmeans.elbow_plot(X)


            X=kmeans.create_clusters(X,number_of_clusters)


            X['Labels']=Y


            list_of_clusters=X['Cluster'].unique()



            for i in list_of_clusters:
                cluster_data=X[X['Cluster']==i]


                cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                cluster_label= cluster_data['Labels']


                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)

                model_finder=tuner.Model_Finder(self.file_object,self.log_writer)


                best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)


                file_op = file_methods.File_Operation(self.file_object,self.log_writer)
                save_model=file_op.save_model(best_model,best_model_name+str(i))


            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:

            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception