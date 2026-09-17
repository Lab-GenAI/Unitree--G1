import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
ChannelFactoryInitialize(0, 'eth0')
def cb(msg): print([msg.motor_state[i].q for i in range(7)])
sub = ChannelSubscriber('rt/lf/dex3/left/state', HandState_)
sub.Init(cb, 10)
time.sleep(3)
