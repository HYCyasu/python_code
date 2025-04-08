import re
import sqlite3

def is_valid_email(email):
    pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
    return re.match(pattern, email) is not None

def input_with_check(prompt, check_fn, err_msg="输入无效，请重试"):
    while True:
        value = input(prompt).strip()
        if check_fn(value):
            return value
        print(err_msg)

def init_db(db_name):
    db = sqlite3.connect(db_name)
    db_cursor = db.cursor()
    db_cursor.execute('''
        CREATE TABLE IF NOT EXISTS info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        )
    ''')
    db.commit()
    return db

def add(db):
    name = input("输入你的名字：\n").strip()
    email = input_with_check("输入你的邮箱：\n", is_valid_email, "邮箱格式错误，请重新输入")
    cursor = db.cursor()
    cursor.execute("INSERT INTO info (name, email) VALUES (?, ?)", (name, email))
    db.commit()
    print("添加成功！")

def delete(db):
    name = input("输入你要删除的名字：\n").strip()
    cursor = db.cursor()
    cursor.execute("DELETE FROM info WHERE name = ?", (name,))
    db.commit()
    print(f"删除了 {cursor.rowcount} 条记录。")

def look(db):
    name = input("请输入要查找的联系人姓名：\n").strip()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM info WHERE name = ?", (name,))
    rows = cursor.fetchall()
    if rows:
        print("找到以下联系人：")
        for row in rows:
            print(row)
    else:
        print("没有找到该联系人。")

def list_all(db):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM info")
    rows = cursor.fetchall()
    print("当前联系人列表：")
    for row in rows:
        print(row)

def update(db):
    cursor = db.cursor()
    name = input("请输入要更新的联系人姓名：\n").strip()
    cursor.execute("SELECT * FROM info WHERE name = ?", (name,))
    rows = cursor.fetchall()

    if not rows:
        print("未找到该联系人。")
        return

    new_email = input_with_check("请输入新的邮箱：\n", is_valid_email, "邮箱格式错误，请重新输入")

    if len(rows) >= 2:
        print("\n存在多个同名联系人，请选择你要修改的那一位：")
        print("格式：id | 姓名 | 邮箱")
        for row in rows:
            print(f"{row[0]} | {row[1]} | {row[2]}")
        valid_ids = [str(row[0]) for row in rows]
        target_id = input_with_check("输入要修改的 id：\n", lambda x: x in valid_ids, f"输入错误，请从 {valid_ids} 中选择")
        cursor.execute("UPDATE info SET email = ? WHERE id = ?", (new_email, target_id))
    else:
        cursor.execute("UPDATE info SET email = ? WHERE name = ?", (new_email, name))

    db.commit()
    if cursor.rowcount > 0:
        print(f"更新成功，修改了 {cursor.rowcount} 条记录。")
    else:
        print("未做任何更新。")

def main():
    db = init_db('userdata.db')
    while True:
        print("\n====== 联系人管理系统 ======")
        print("1. 添加联系人")
        print("2. 查找联系人")
        print("3. 查看所有联系人")
        print("4. 更新联系人邮箱")
        print("5. 删除联系人")
        print("6. 退出")
        choice = input("请输入操作编号：\n").strip()
        match choice:
            case "1":
                add(db)
            case "2":
                look(db)
            case "3":
                list_all(db)
            case "4":
                update(db)
            case "5":
                delete(db)
            case "6":
                break
            case _:
                print("无效的选项，请重新输入。")
    db.close()

if __name__ == '__main__':
    main()
