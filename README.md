# Unitree--G1
The Unitree G1 is a highly compact, cost-effective humanoid robot AI avatar developed by Unitree Robotics


This repository contains detailed information and programs that have been used to test stuff like locomotion, object identification, human detection and ident, factory action calling, etc. on the PwC Gen AI Lab's Unitree G1- that we lovingly call T.O.N.Y- Tool Orchestration, Navigation & Yield.

g1_slam_client_v6 is mainly used for point to point movement and purely has a dependency on the tablet running- the LiDAR sensor's POSTs are routed to the tablet, which are read by this program as a host- keeping a note of waypoints and the origin(or charging station.) You can run the program without any parameters to view all the configurable and usable params. Be sure to use --i-have-cleared-the-area to make the robot actually move from a waypoint to another (couldn't think of a more blatantly obvious name for making sure if the area is cleared.)

right now the slam client does not make T.O.N.Y AVOID objects- rather just stands there menacingly waiting for objects to get out of his way. Working on a v7 that'll invoke active obstacle avoidance. 

Next up is g1_elevenlabs_agent_final.py, the voice agent. Main speech interface is controlled by this program. Controls all the factory functions and movements via speech. 

Working on a new agent from elevenlabs for linking every single possible action with speech.
