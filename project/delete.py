username_to_delete = 'Janvi'

conn = sqlite3.connect('users.db')
cur = conn.cursor()
cur.execute("DELETE FROM users WHERE username=?", (username_to_delete,))
conn.commit()
conn.close()

print(f"✅ User '{username_to_delete}' deleted.")
