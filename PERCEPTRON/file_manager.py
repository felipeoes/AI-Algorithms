import pandas as pd

class FileManager(object):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_csv(self, cols_names: list = None, header=None, save: bool = True, chars_df: bool = False):
        """ Reads a csv file and returns a formatted pandas dataframe."""

        try:
            df = pd.read_csv(self.file_path, header=header)
            
            if chars_df:
                y_length = 7
                x_length = len(df.columns) - y_length

                cols_x = [f"attr{num}" for num in range(x_length)]
                cols_y = [f"label{num}" for num in range(y_length)]
                cols = [*cols_x, *cols_y]

                df.columns = cols
            else:
                df.columns = cols_names if cols_names else df.columns

            if save:
                self.__save_df_excel__(df)

            return df
        except Exception as e:
            print(f"Fail during dataframe generation! | Exception: {e}")
            return None

    def __save_df_excel__(self, df: pd.DataFrame):
        """ Saves a dataframe in an excel file format."""

        try:
            path = self.file_path.replace(".csv", ".xlsx")
            df.to_excel(path, index=False)
        except Exception as e:
            print(f"Fail during dataframe saving! | Exception: {e}")
            return None

    def extract_X_y(self, df: pd.DataFrame, chars_df: bool = False):
        """ Exctracts the training columns (X) and the target column (y) of a dataframe."""

        try:
            if chars_df:
                y_length = 7
                X_length = len(df.columns) - y_length
                
                X = df.iloc[0:, 0: X_length].values.tolist()
                y = df.iloc[0:, X_length: (X_length+y_length)].values.tolist()
                
            else:   
                X = df.loc[0: len(df.columns), :].values.tolist()
                y = []
                
                for col in X:
                    y.append(col.pop())

            

            return X, y
        except Exception as e:
            print(f"Fail during dataframe extraction! | Exception: {e}")
            return None, None
