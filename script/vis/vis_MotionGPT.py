import open3d as o3d
import smplx
import numpy as np
import time

model = smplx.create(
    'script/retarget/smpl/model/smpl/SMPL_NEUTRAL.pkl',
    model_type="smpl",
)
data = np.load("datas/mocap/MotionGPT/47_out_mesh.npy")  

# data.shape = [N, 6893, 3]
vis = o3d.visualization.Visualizer()
vis.create_window()
N = data.shape[0]
mesh = o3d.geometry.TriangleMesh()

for i in range(N):
    print(i)
    vertices = data[i]

    # 创建三角形网格
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    faces = model.faces.astype(np.int32)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    # 简化网格
    target_number_of_faces = 800  # 设置目标面数
   # mesh = mesh.simplify_quadric_decimation(target_number_of_faces)

    if i == 0:
        vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.05)
vis.destroy_window()

