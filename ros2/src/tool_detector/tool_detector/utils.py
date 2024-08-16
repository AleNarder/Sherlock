import os


def load_tools ():
    tools = []
    for tool in os.listdir('ros2/src/tool_detector/tool_detector/tools'):
        if tool.endswith('.xml'):
            tools.append(tool)