@echo off
call conda activate
start python User/user_launcher.py
start python Perception/perception_launcher.py
start python Control/control_launcher.py
start python Condition/condition_launcher.py
