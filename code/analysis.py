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

def get_csv_data(path):
    result=[]
    tot=1
    with open(path) as f:
        context=f.read().split('\n')
        for line in context:
            if tot==1:
                tot=0
                continue
            if line=='':
                continue
            l=line.split(',')
            left=[]
            right=[]
            for i in range(7):
                left.append(float(l[i+1]))
                right.append(float(l[i+8]))
            result.append(left)
            result.append(right)
    
    # 转置二维list
    result = np.array(result).T.tolist()
    names = ["500Hz","1kHz","2kHz","3kHz","4kHz","6kHz","8kHz"]
    dictionary = dict(zip(names, result))
    df = pd.DataFrame(dictionary)
    profile = ProfileReport(df, title='Pandas Profiling Report')
    # Save the report
    profile.to_file(output_file='../report/1.html')


if __name__=='__main__':
    '''
    input_path= '../data/d.xlsx'
    output_path = '../report/d.html'
    generate_report(input_path, output_path)
    '''
    path = '../data/audiogram_concate_withoutNan_class.csv'
    get_csv_data(path)
