import os

import pybullet as p
import time
import pybullet_data

DIR = os.path.dirname(os.path.realpath(__file__))

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setAdditionalSearchPath(os.path.join(DIR, 'pybullet_data'))  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

cube_urdf_name = 'cube_dd.urdf'

boxId = p.loadURDF(cube_urdf_name, cubeStartPos, cubeStartOrientation)


# h = 0.1
# boxId = p.createVisualShape(p.GEOM_BOX,
#                             halfExtents=[h, h, h], rgbaColor=[1, 0, 0, 1],
#                             specularColor=[0.4, .4, 0])
#
# colId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h, h, h])
#
# p.createMultiBody(baseMass=1,
#                   baseInertialFramePosition=cubeStartPos,
#                   baseCollisionShapeIndex=colId,
#                   baseVisualShapeIndex=boxId,
#                   basePosition=cubeStartPos,
#                   useMaximalCoordinates=False)

for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
p.disconnect()
