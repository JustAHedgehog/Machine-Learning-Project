import sqlite3 as sq
import os


def reSuffix(files):
    extract_name = []
    for file in files:
        if file.endswith('_output.txt'):
            base_name = file.rsplit('-', 1)[0]
            extract_name.append(base_name)
    return extract_name


def readtxt(path, files, name):
    text = []
    for file in files:
        if file == name+'-職能基準_output.txt':
            try:
                with open(path+'/'+file, 'r', encoding='utf-8') as f:
                    for line in f:
                        text.append(line.strip())
            except FileNotFoundError:
                print(f"檔案未找到")
            except Exception as e:
                print(f"發生錯誤: {e}")
    return text


def split_content(data):
    knowledge = []
    skills = []
    current_section = None

    for item in data:
        if item.startswith('職能內涵 (K=knowledge 知識):'):
            current_section = 'knowledge'
        elif item.startswith('職能內涵 (S=skills 技能):'):
            current_section = 'skills'
        else:
            if current_section == 'knowledge':
                knowledge.append(item)
            elif current_section == 'skills':
                skills.append(item)
    return knowledge, skills


def update_db(db_name, name, knowledge, skills):
    # 連接到 SQLite 資料庫（如果不存在則創建）
    conn = sq.connect(db_name)
    cursor = conn.cursor()
    # 將資料插入到表格中
    cursor.execute("UPDATE competency SET 職能內涵K = ?, 職能內涵S = ? WHERE 職業 = ?", (', '
                                                                               .join(knowledge), ', '.join(skills), name))
    # 提交變更並關閉連接
    conn.commit()
    conn.close()

if __name__ == '__main__':
    path = './PDFs'
    comp = []
    db = 'job第一版.db'
    files = os.listdir(path)
    names = reSuffix(files)
    # print(names)
    for name in names:
        content = readtxt(path, files, name)
        knowledge, skills = split_content(content)
        comp.append((knowledge, skills))
        update_db(db, name, knowledge, skills)
    print("資料已成功更新到資料庫中。")
