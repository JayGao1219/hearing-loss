import openpyxl

def process_excel(input_path, output_path):
    # 打开 Excel 文件
    workbook = openpyxl.load_workbook(input_path)

    # 新建一个 Excel 文件
    output_workbook = openpyxl.Workbook()
    output_worksheet = output_workbook.active

    # 遍历每个工作表
    # 把两个sheet合成一个sheet
    for worksheet in workbook:
        # 遍历每一行
        for row in worksheet.iter_rows(values_only=True):
            # 如果只有最后一列是 **，则赋值为倒数第二列的值
            if row[-1] == '**' and row[-2] != '**':
                row = list(row)
                row[-1] = row[-2]
            # 如果超过两列是 **，则忽略掉这一行
            elif row.count('**') > 1:
                continue
            # 写入新的 Excel 文件
            output_worksheet.append(row)

    # 保存并关闭文件
    output_workbook.save(output_path)

# 测试代码
if __name__=='__main__':
    input_path = '../data/raw_data/audiogram.xlsx'
    output_path = '../data/output.xlsx'
    process_excel(input_path, output_path)
