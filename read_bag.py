import sqlite3, sys
from rclpy.serialization import deserialize_message
from unitree_api.msg import Request

db = sys.argv[1]
conn = sqlite3.connect(db)
c = conn.cursor()

c.execute("SELECT id, name FROM topics")
topics = {tid: name for tid, name in c.fetchall()}
print("Topics:", topics, "\n")
c.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp")
for topic_id, ts, data in c.fetchall():
	msg = deserialize_message(bytes(data), Request)
	print(f"api_id={msg.header.identity.api_id}")
	print(f"  parameter: {msg.parameter}")
	print()


