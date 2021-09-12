from datetime import datetime
from os import listdir
import pandas
from application_logging.logger import App_Logger


class dataTransform:


     def __init__(self):
          self.goodDataPath = "Training_Raw_files_validated/Good_Raw"
          self.logger = App_Logger()


     def replaceMissingWithNull(self):


          log_file = open("Training_Logs/dataTransformLog.txt", 'a+')
          try:
               onlyfiles = [f for f in listdir(self.goodDataPath)]
               for file in onlyfiles:
                    data = pandas.read_csv(self.goodDataPath + "/" + file)

                    columns = ['Income', 'workclass','education', 'marital-status', 'occupation', 'relationship',
                              'race','sex', 'native-country']

                    for col in columns:
                         data[col] = data[col].apply(lambda x: "'" + str(x) + "'")


                    data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                    self.logger.log(log_file, " %s: Quotes added successfully!!" % file)

          except Exception as e:
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)

               log_file.close()
          log_file.close()
