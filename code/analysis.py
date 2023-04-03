from pandas_profiling import ProfileReport
import numpy as np
import pandas as pd


def generate_report(input_path, output_path):
    # Read in the data
    df = pd.read_excel(input_path, sheet_name='Sheet')
    # Generate the report
    profile = ProfileReport(df, title='Pandas Profiling Report')
    # Save the report
    # profile.to_file(output_file='reports/pandas_profiling_report.html')
    profile.to_file(output_file=output_path)

def data_clean(input_path):
    # Read in the data
    df = pd.read_excel(input_path, sheet_name='Sheet')
    df = df.astype(float)
    # 打印每一列的数据类型
    print(df.dtypes)
    

if __name__=='__main__':
    input_path= '../data/output.xlsx'
    output_path = '../report/pandas_profiling_report.html'
    generate_report(input_path, output_path)
