#g1_find_aes.py

import asyncio
from unitree_webrtc_connect.unitree_cloud import fetch_aes_key

async def get_key():
	key = await fetch_aes_key(
		email="innovationhub787@gmail.com",
		password="Lab@2026",
		sn="E2D16000Q1L7CD80",
		region="global",
		device_type="G1"
	)
	print(key)
asyncio.run(get_key())
