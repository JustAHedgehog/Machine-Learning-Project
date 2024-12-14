import sqlite3 as sql
import os
def reSuffix(file):
    if file.endswith('_output.txt'):
        base_name = os.path.splitext(file)[0]
        title = base_name.rsplit('-', 1)[0]
    return title
path = "小型汽車維修廠品質技術人員-職能基準_output.txt"
knowledge = []
skills = []
current_section = None
with open(path,'r',encoding="utf-8") as f:
    file = f.name
    jobName = reSuffix(file)
    text = []
    for line in f:
        text.append(line.strip())
    for item in text:
        if item.startswith('職能內涵 (K=knowledge 知識):'):
            current_section = 'knowledge'
        elif item.startswith('職能內涵 (S=skills 技能):'):
            current_section = 'skills'
        else:
            if current_section == 'knowledge':
                knowledge.append(item)
            elif current_section == 'skills':
                skills.append(item)
    conn = sql.connect('job第一版.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE competency SET 職能內涵K=?, 職能內涵S=? WHERE 職業 = ?', (', '.join(knowledge), ', '.join(skills), jobName))
    print('UPDATE competency SET 職能內涵K=?, 職能內涵S=? WHERE 職業 = ?', (', '.join(knowledge), ', '.join(skills), jobName))
    conn.commit()
    cursor.close()
    conn.close()
    # print(knowledge)