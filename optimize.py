import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import kaolin as kal
from kaolin.render.camera import Camera
import imageio
from tqdm import tqdm

# Set the cuda device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the mesh
mesh = kal.io.import_mesh("data/Teapot.usdc", triangulate=True).cuda()

# Normalize so it is easy to set up default camera
mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)

def create_camera(height=512, width=512, fov=20.0, 
                 eye=torch.tensor([3.0, 6.9, 2.5]),
                 at=torch.tensor([0.0, 0.0, 0.0]),
                 up=torch.tensor([0.0, 1.0, 0.0]),
                 optimizable=False):
    """
    Create a camera with specified parameters.
    
    Args:
        height (int): Image height
        width (int): Image width
        fov (float): Field of view in degrees
        eye (tensor): Camera position
        at (tensor): Look-at position
        up (tensor): Up direction
        optimizable (bool): Whether camera parameters should be optimizable
    """
    camera = Camera.from_args(
        eye=eye.to(device),
        at=at.to(device),
        up=up.to(device),
        fov=torch.pi * fov / 180,
        height=height,
        width=width
    )
    if optimizable:
        camera.requires_grad_(True)
        ext_mask, int_mask = camera.gradient_mask('t', 'focal_x', 'focal_y')
        ext_params, int_params = camera.parameters()
        ext_params.register_hook(lambda grad: grad * ext_mask.float())
        grad_scale = 1e5    # Used to move the projection matrix elements faster
        int_params.register_hook(lambda grad: grad * int_mask.float() * grad_scale)
    return camera

def render_silhouette(mesh, camera):
    vertices = mesh.vertices.unsqueeze(0)
    vertices_camera = camera.extrinsics.transform(vertices)
    vertices_image = camera.intrinsics.transform(vertices_camera)

    # Index vertices by faces with batch dimension
    faces_batch = mesh.faces.unsqueeze(0)  # [1, F, 3]
    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, mesh.faces)
    face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, mesh.faces)[..., :2]

    # Create face features for albedo (all ones for silhouette) with batch dimension
    in_face_features = torch.ones((1, mesh.faces.shape[0], mesh.faces.shape[1], 3),
                            dtype=camera.dtype, 
                            device=camera.device)

    # Get face vertices z for depth (maintain batch dimension)
    face_vertices_z = face_vertices_camera[..., -1]

    # Perform rasterization with full nvdiffrast backend
    face_features, face_idx = kal.render.mesh.rasterize(
        camera.height, camera.width,
        face_vertices_z=face_vertices_z,
        face_vertices_image=face_vertices_image.contiguous(),
        face_features=in_face_features,
        eps=1e-8,  # Reduced epsilon for better gradient flow
        backend='nvdiffrast'
    )

    return face_features

# Create reference image with initial position
reference_camera = create_camera()
reference_image = render_silhouette(mesh, reference_camera)

class Model(nn.Module):
    def __init__(self, mesh, image_ref):
        super().__init__()
        self.mesh = mesh
        self.device = mesh.vertices.device
        
        # Register reference image
        self.register_buffer('image_ref', image_ref)
        
        # Store initial camera distance
        initial_eye = torch.tensor([2.0, 4.0, 5.0])
        self.initial_distance = torch.norm(initial_eye)
        
        # Create optimizable camera
        self.camera = create_camera(
            eye=initial_eye,
            at=torch.tensor([0.0, 0.0, 0.0]),
            optimizable=True
        )

        # Initialize camera position on unit sphere
        self.camera_direction = nn.Parameter(torch.nn.functional.normalize(initial_eye), requires_grad=True)

    def forward(self):
        # Normalize camera direction and scale by initial distance
        normalized_direction = torch.nn.functional.normalize(self.camera_direction)
        camera_position = normalized_direction * self.initial_distance
        camera_position_reshaped = camera_position.view(-1, 1)
        
        # Update camera extrinsics with fixed distance
        self.camera.extrinsics = kal.render.camera.extrinsics.CameraExtrinsics.from_lookat(
            eye=camera_position_reshaped,
            at=torch.tensor([0.0, 0.0, 0.0], device=self.device),
            up=torch.tensor([0.0, 1.0, 0.0], device=self.device),
            device=self.device,
            requires_grad=True
        )
        
        # Render image
        image = render_silhouette(self.mesh, self.camera)
        
        # Calculate losses
        image_loss = torch.nn.functional.mse_loss(image, self.image_ref)
        
        # Add regularization to maintain up direction approximately [optional]
        up_vector = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        current_up = self.camera.extrinsics.R[:, 1]
        up_loss = 0.1 * (1 - torch.dot(current_up, up_vector))
        
        loss = image_loss + up_loss
        
        return loss, image

# Initialize model and optimizer
model = Model(mesh=mesh, image_ref=reference_image).to(device)
optimizer = torch.optim.Adam([model.camera_direction], lr=0.01)

# Add cosine learning rate scheduler [optional]
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4)

# Save outputs as GIF
filename_output = "teapot_optimization_kaolin.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)
# Optimization loop with gradient checking
loop = tqdm(range(200))
for i in loop:
    optimizer.zero_grad()
    
    # Use detect_anomaly for debugging gradient flow
    with torch.autograd.detect_anomaly():
        loss, current_image = model()
        loss.backward(retain_graph=True)

    # Print gradients for debugging
    print(f"Iteration {i}")
    print("Loss:", loss.item())
    print("Gradient of camera_position:", model.camera_position.grad)

    optimizer.step()
    
    loop.set_description(f'Optimizing (loss {loss.data:.4f})')
    
    if loss.item() < 0.01:
        break
    
    # Save outputs to create a GIF
    if i % 10 == 0:
        img_np = (current_image.detach().cpu().numpy() * 255).astype(np.uint8)
        writer.append_data(img_np[0])
        
        plt.figure()
        plt.imsave(f"tmp_{i}.png", img_np[0])
        plt.title(f"iter: {i}, loss: {loss.data:.2f}")
        plt.axis("off")
        plt.close()

writer.close()

# Final visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(reference_image[0].cpu().numpy())
plt.title("Reference Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(current_image[0].detach().cpu().numpy())
plt.title("Optimized Result")
plt.axis("off")
plt.show()