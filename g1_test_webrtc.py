import asyncio
from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod
async def main():
	conn = UnitreeWebRTCConnection(
		WebRTCConnectionMethod.LocalSTA,
		ip="192.168.123.161",
		aes_128_key="E2D16000Q1L7CD80",
	)
	try:
		await conn.connect()
		print("Connected")
	except Exception as e:
		print(f"Failed: {type(e).__name__}: {e}")
asyncio.run(main())
