import asyncio
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod

async def main():
    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip="192.168.123.161",
        aes_128_key="7f4aba4b310ffa8b745148aff0acbe46",
    )
    await conn.connect()
    print("Connected!")

    # Test with guaranteed output first
    resp = await conn.datachannel.pub_sub.publish_request_new(
        "rt/api/bashrunner/request",
        {"api_id": 1002, "parameter": {"cmd": "whoami && pwd && cat /proc/version"}}
    )
    print("tty devices:", resp)

    await asyncio.sleep(2)

    resp2 = await conn.datachannel.pub_sub.publish_request_new(
        "rt/api/bashrunner/request",
        {"api_id": 1002, "parameter": {"cmd": "ps aux | grep -i 'brainco\|dex3\|hand\|inspire'"}}
    )
    print("hand processes:", resp2)

    await asyncio.sleep(2)

    resp3 = await conn.datachannel.pub_sub.publish_request_new(
        "rt/api/bashrunner/request",
        {"api_id": 1002, "parameter": {"cmd": "ls /dev/ttyUSB* /dev/ttyACM* /dev/ttyS* 2>&1"}}
    )
    print("serial devices:", resp3)

asyncio.run(main())
