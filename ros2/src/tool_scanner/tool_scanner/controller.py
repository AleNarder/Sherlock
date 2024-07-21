class RobotController:
    def __init__(self, robot):
        self.robot = robot

    def move(self, x, y, z):
        self.robot.move(x, y, z)

    def scan(self):
        self.robot.scan()