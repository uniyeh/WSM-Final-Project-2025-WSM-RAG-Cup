import sqlite3

class Connection:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
    def execute(self, sql, params=None):
        cursor = self.conn.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        self.conn.commit()
        return cursor
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()