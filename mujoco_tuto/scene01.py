import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path("scene01_falling_box.xml")
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

while viewer.is_alive:
    mujoco.mj_step(model, data)
    viewer.render()

    # afficher la hauteur pour comprendre
    print(f"z={data.qpos[2]:.3f}  vz={data.qvel[2]:.3f}")

    # reset seulement quand la boîte est posée ET ne bouge plus
    if data.qpos[2] < 0.11 and abs(data.qvel[2]) < 0.01:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 2.0

viewer.close()
