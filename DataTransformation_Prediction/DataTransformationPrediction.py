from datetime import datetime
from os import listdir
import pandas
from application_logging.logger import App_Logger


class dataTransformPredict:

     def __init__(self):
          self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
          self.logger = App_Logger()


     def replaceMissingWithNull(self):


          try:
               log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
               onlyfiles = [f for f in listdir(self.goodDataPath)]
               for file in onlyfiles:
                    data = pandas.read_csv(self.goodDataPath + "/" + file)

                    columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                               'race', 'sex', 'native-country']

                    for col in columns:
                         data[col] = data[col].apply(lambda x: "'" + str(x) + "'")

                    data.to_csv(self.goodDataPath+ "/" + file, index=None, header=True)
                    self.logger.log(log_file," %s: File Transformed successfully!!" % file)


          except Exception as e:
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)

               log_file.close()
               raise e
          log_file.close()
