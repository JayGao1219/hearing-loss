import openpyxl
import os

def process_excel(input_path, output_path):
    # 打开 Excel 文件
    workbook = openpyxl.load_workbook(input_path)

    # 新建一个 Excel 文件
    output_workbook = openpyxl.Workbook()
    output_worksheet = output_workbook.active
    iid=[]

    # 遍历每一行
    for worksheet in workbook:
        print(worksheet.title)
        for row in worksheet.iter_rows(values_only=True):
            row = list(row)
            if row[0] in iid and type(row[0]) is not str:
                row[0] = -row[0]
            # 如果只有最后一列是 **，则赋值为倒数第二列的值
            if row[-1] == '**' and row[-2] != '**':
                row[-1] = row[-2]
            # 如果非边界值是 **，则进行插值
            if '**' in row[1:-1]:
                continue

            # 写入新的 Excel 文件
            iid.append(row[0])
            row = tuple(row)
            output_worksheet.append(row)

    # 保存并关闭文件
    output_workbook.save(output_path)



# 测试代码
if __name__=='__main__':
    for f in os.listdir('../data/raw_data'):
        if f.endswith('.xlsx'):
            print(f)
            input_path = '../data/raw_data/' + f
            output_path = '../data/' + f
            process_excel(input_path, output_path)
