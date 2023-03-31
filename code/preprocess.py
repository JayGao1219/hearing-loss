import openpyxl

def process_excel(input_path, output_path):
    # 打开 Excel 文件
    workbook = openpyxl.load_workbook(input_path)

    # 选择工作表
    worksheet = workbook.active

    # 新建一个 Excel 文件
    output_workbook = openpyxl.Workbook()
    output_worksheet = output_workbook.active
    iid=[]

    # 遍历每一行
    for row in worksheet.iter_rows(values_only=True):
        row = list(row)
        if row[0] in iid:
            row[0] = -row[0]
        # 如果只有最后一列是 **，则赋值为倒数第二列的值
        if row[-1] == '**' and row[-2] != '**':
            row = list(row)
            row[-1] = row[-2]
        # 如果非边界值是 **，则进行插值
        elif '**' in row[1:-1]:
            try:
                for i in range(1, len(row)-1):
                    if row[i] == '**':
                        j = i - 1
                        while row[j] == '**':
                            j -= 1
                        k = i + 1
                        while row[k] == '**':
                            k += 1
                        row[i] = (row[j] + row[k]) / 2
            except IndexError:
                continue
        # 写入新的 Excel 文件
        iid.append(row[0])
        row = tuple(row)
        output_worksheet.append(row)

    # 保存并关闭文件
    output_workbook.save(output_path)



# 测试代码
if __name__=='__main__':
    input_path = '../data/raw_data/audiogram.xlsx'
    output_path = '../data/output.xlsx'
    process_excel(input_path, output_path)
