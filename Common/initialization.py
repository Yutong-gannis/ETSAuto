import sys, os

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))

sys.path.insert(0, os.path.abspath(project_path))


def init():
    nav_line = None
    return nav_line
