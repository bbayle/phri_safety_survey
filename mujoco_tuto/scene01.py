import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path("scene01_falling_box.xml")
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

# force camera position directly on the viewer object
viewer.cam.lookat[0] = 0      # look at x=0
viewer.cam.lookat[1] = 0      # look at y=0
viewer.cam.lookat[2] = 1.0    # look at midpoint of fall
viewer.cam.distance  = 5.0    # 5m away
viewer.cam.azimuth   = 90     # side view
viewer.cam.elevation = -20    # looking slightly down

while viewer.is_alive:
    mujoco.mj_step(model, data)
    viewer.render()

    print(f"z={data.qpos[2]:.3f}  vz={data.qvel[2]:.3f}")

    if data.qpos[2] < 0.11 and abs(data.qvel[2]) < 0.01:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 2.0

viewer.close()
