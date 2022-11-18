import pandas
import sqlalchemy

class DataHandler:
  @staticmethod
  def add_data_to_db(df:pandas.DataFrame, **kwargs):
    df.to_sql(**kwargs)
  
  @staticmethod
  def get_data_from_db(query:str, con, **kwargs):
    return pandas.read_sql_query(query, con, **kwargs)

  @staticmethod
  def dict_list_to_dataframe(data_dict_list:list, **kwargs)->pandas.DataFrame:
    return pandas.DataFrame.from_dict(data_dict_list)
  
  @staticmethod
  def concat_dataframe_list(df_list:list)->pandas.DataFrame:
    return pandas.concat(df_list)

  @staticmethod
  def db_connect(db_url:str):
    try:
      engine = sqlalchemy.create_engine(db_url)    
      metadata = sqlalchemy.MetaData(bind=engine)
      metadata.reflect(only=['test_table'])
      test_table=metadata.tables['test_table']    
      print(f"Connected:\n {test_table}")
      return engine

    except Exception as ex:
            print('Connection could not be made due to the following error: \n', ex)
