"""
extract.py: Seam Detection and Visualization on 3D Meshes.

This script is designed to process 3D meshes, primarily for the purpose of identifying seams 
and visualizing them in various forms. Seam edges are crucial in 3D graphics for tasks such as 
UV mapping, where a continuous 3D surface needs to be mapped onto a 2D plane without distortion.

Main Functionalities:
- Load 3D mesh and associated textures.
- Identify seams on the mesh.
- Rasterize the mesh's UVs and seams to generate 2D visualizations.
- Compute signed distance field (SDF) from the UV mask for enhanced visual clarity.
- Output visualization images for further analysis or integration into other systems.

Typical use-case:
- A 3D artist or graphics programmer would utilize this script to understand the seam layout 
  on a given 3D model. This aids in optimizing texture mapping or making manual adjustments 
  to the UV layout.

Usage:
- Import necessary functions and classes into another script or Jupyter notebook.
- Or run this script directly, optionally providing a 3D mesh file as an argument to `main`.

Example:
>>> from mesh_processing import main
>>> main("path_to_3D_mesh.glb")

Dependencies:
- trimesh, PIL, numpy, and scipy libraries.

Author:
[Your Name] - [Your Email Address] - [Any other relevant contact or citation info]
"""
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
import time
import math

start_time = time.time()


class HalfEdge:
    """
    Represents a half edge in UV space, essential for efficient mesh representation and traversal.

    Half edges are foundational elements in the half-edge data structure, commonly used in computational geometry to represent and manipulate mesh topologies. This data structure aids in tasks like seam detection and other mesh operations. A half edge has a start point (`a`) and an end point (`b`), both given in UV coordinates.

    Attributes:
    - a (tuple): Start UV coordinate of the half edge.
    - b (tuple): End UV coordinate of the half edge.

    Parameters:
    - a (tuple): Start UV coordinate to initialize the half edge with.
    - b (tuple): End UV coordinate to initialize the half edge with.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b


class SeamEdge:
    """
    Represents a seam on the mesh, crucial for UV unwrapping and texture mapping.

    Seams are typically edges where the mesh is 'cut' to be unfolded flat for texturing. By defining and distinguishing these edges, we can better manipulate and understand the mesh's UV layout. The `color` attribute can be used to visually distinguish seams, especially when generating a seam map for visualization or debugging purposes.

    Attributes:
    - edges (list): A pair of edges defining the seam. Each edge is represented by its start and end UV coordinates.
    - color (list): The color assigned to this seam for visualization.

    Parameters:
    - edge1 (tuple): The first edge of the seam, represented by its start and end UV coordinates.
    - edge2 (tuple): The second edge of the seam, represented by its start and end UV coordinates.
    """

    def __init__(self, edge1, edge2):
        self.edges = [edge1, edge2]
        self.color = []


class Mesh:
    """
    Represents a 3D triangular mesh, a fundamental data structure in computer graphics.
    The mesh comprises vertices, their corresponding UV coordinates for texturing, and faces that define the structure.
    Using this class simplifies the storage, manipulation, and traversal of 3D objects in various graphical operations.

    Attributes:
    - verts (list of tuples): List of vertices, each represented by a tuple of (x, y, z) coordinates.
    - uvs (list of tuples): List of UV coordinates corresponding to the vertices, each represented by a tuple of (u, v) coordinates.
    - faces (list of tuples): List of faces of the mesh, where each face is represented by a tuple of 3 indices corresponding to `verts`.
    """

    def __init__(self):
        self.verts = []
        self.uvs = []
        self.faces = []


# DEBUG FUNCTION
def draw_red_line(image, point1, point2):
    """
    Draws a red line between point1 and point2 on the given image.

    Parameters:
    - image: PIL Image object
    - point1: Tuple of (x1, y1) coordinates for start point
    - point2: Tuple of (x2, y2) coordinates for end point

    Returns:
    - PIL Image with the drawn line
    """

    draw = ImageDraw.Draw(image)
    # Convert points to tuples, if they're numpy arrays
    point1 = tuple(point1)
    point2 = tuple(point2)

    draw.line([point1, point2], fill="red", width=2)

    return image


def get_translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


def get_rotation_matrix(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def get_scaling_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])


def get_affine_matrix(src, dst):
    A, B = np.array(src)
    C, D = np.array(dst)

    # Translate A to the origin
    TA = get_translation_matrix(-A[0], -A[1])

    # Rotate AB to lie on the x-axis
    theta_AB = np.arctan2(B[1] - A[1], B[0] - A[0])
    R1 = get_rotation_matrix(-theta_AB)

    # Scale the length of AB to match CD
    length_AB = np.linalg.norm(B - A)
    length_CD = np.linalg.norm(D - C)
    s = length_CD / length_AB
    S = get_scaling_matrix(s, s)

    # Rotate the transformed AB to align with CD
    theta_CD = np.arctan2(D[1] - C[1], D[0] - C[0])
    R2 = get_rotation_matrix(theta_CD)

    # Translate so that its start point aligns with C
    TC = get_translation_matrix(C[0], C[1])

    return np.linalg.multi_dot([TC, R2, S, R1, TA])


def UVToScreen(in_vec, W, H):
    out_vec = in_vec.copy()
    out_vec[1] = 1.0 - out_vec[1]
    out_vec[0] *= W
    out_vec[1] *= H
    return out_vec - np.array([0.5, 0.5])


def is_flipped(uv0, uv1, uv2):
    """
    Flipped triangles can lead to visual artifacts or misrepresentations in texture mapping. This function aids in detecting such inconsistencies.
    Identifies texture mapping inconsistencies by checking a triangle's winding order in UV space.

    Parameters:
    - uv0, uv1, uv2 (tuple of float): The UV coordinates of the triangle's vertices.

    Returns:
    - bool: True if the triangle is flipped in UV space, otherwise False.

    Why:

    """

    return (uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (uv2[0] - uv0[0]) * (
        uv1[1] - uv0[1]
    ) < 0


def isInside(x, y, ea, eb):
    """
    Determines if a point (x, y) lies inside or on the line defined by two endpoints (ea, eb) using a determinant method.
    Accurate dtermination of point-line relationships is crucial for various geometry-based algorithms and tasks, such as rasterization and hit-testing.

    Parameters:
    - x, y (float): Coordinates of the point to be checked.
    - ea, eb (tuple of float): Endpoints defining the line segment.

    Returns:
    - bool: True if the point lies inside or on the line segment, otherwise False.
    """
    return (
        x * (eb[1] - ea[1]) - y * (eb[0] - ea[0]) - ea[0] * eb[1] + ea[1] * eb[0] >= 0
    )


def WrapCoordinate(x, size):
    """
    Wraps a coordinate value around a specified range (0 to size), ensuring it stays within bounds.
    This is useful for operations on cyclic or periodic domains, such as texture mapping or modular arithmetic.

    Parameters:
    - x (int or float): The coordinate value to be wrapped.
    - size (int or float): The maximum boundary for wrapping.

    Returns:
    - int or float: The wrapped coordinate value within the range [0, size).
    """
    while x < 0:
        x += size
    while x >= size:
        x -= size
    return x


def is_point_in_triangle(pt, v0, v1, v2):
    """
    Check if a point is inside a triangle in 2D (UV) space.
    """
    # Check if the point is on the "inside" side of all the triangle's edges
    return (
        isInside(pt[0], pt[1], v0, v1)
        and isInside(pt[0], pt[1], v1, v2)
        and isInside(pt[0], pt[1], v2, v0)
    )


def compute_barycentric_coords_vectorized(pt, v0, v1, v2):
    """
    Compute the barycentric coordinates of a point within a triangle.
    """
    detT = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    alpha = (
        (v1[1] - v2[1]) * (pt[..., 0] - v2[0]) + (v2[0] - v1[0]) * (pt[..., 1] - v2[1])
    ) / detT
    beta = (
        (v2[1] - v0[1]) * (pt[..., 0] - v2[0]) + (v0[0] - v2[0]) * (pt[..., 1] - v2[1])
    ) / detT
    gamma = 1.0 - alpha - beta
    return np.stack([alpha, beta, gamma], axis=-1)


def RasterizeFace(uv0, uv1, uv2, coverageBuf):
    """
    Rasterizes a triangle defined by three UV coordinates onto a coverage buffer, filling its interior.
    The rasterization is essential for representing a triangle on pixel-based devices or screens. By translating triangle vertices into screen space,
    we can compute and mark the pixels that lie inside the triangle.

    Parameters:
    - uv0, uv1, uv2 (tuple of float): UV coordinates of the triangle's vertices.
    - coverageBuf (numpy.ndarray): A 2D buffer where the triangle will be rasterized.

    Returns:
    - numpy.ndarray: The updated coverage buffer with the rasterized triangle.
    """
    uv0 = UVToScreen(uv0, coverageBuf.shape[1], coverageBuf.shape[0])
    uv1 = UVToScreen(uv1, coverageBuf.shape[1], coverageBuf.shape[0])
    uv2 = UVToScreen(uv2, coverageBuf.shape[1], coverageBuf.shape[0])

    # Axis aligned bounds of the triangle
    minx = int(min(uv0[0], uv1[0], uv2[0]))
    maxx = int(max(uv0[0], uv1[0], uv2[0])) + 1
    miny = int(min(uv0[1], uv1[1], uv2[1]))
    maxy = int(max(uv0[1], uv1[1], uv2[1])) + 1

    # The three edges we will test
    e0a, e0b = uv0, uv1
    e1a, e1b = uv1, uv2
    e2a, e2b = uv2, uv0

    # Now just loop over a screen aligned bounding box around the triangle, and test each pixel against all three edges
    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            if (
                isInside(x, y, e0a, e0b)
                and isInside(x, y, e1a, e1b)
                and isInside(x, y, e2a, e2b)
            ):
                coverageBuf[
                    WrapCoordinate(y, coverageBuf.shape[0]),
                    WrapCoordinate(x, coverageBuf.shape[1]),
                ] = 255
    return coverageBuf


def create_mask_from_line(start, end, img_width, img_height, rect_width=4):
    # Calculate the direction vector of the line
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    # Normalize the direction vector
    length = np.sqrt(dx * dx + dy * dy)
    dx /= length
    dy /= length

    # Calculate the normal to the line (perpendicular)
    nx = -dy
    ny = dx

    # Calculate the four corners of the rectangle
    half_width = rect_width / 2.0
    p1 = (start[0] - nx * half_width, start[1] - ny * half_width)
    p2 = (start[0] + nx * half_width, start[1] + ny * half_width)
    p3 = (end[0] + nx * half_width, end[1] + ny * half_width)
    p4 = (end[0] - nx * half_width, end[1] - ny * half_width)

    # Create a blank mask
    mask = Image.new("L", (img_width, img_height), 0)

    # Draw the gradient rectangle on the mask
    draw = ImageDraw.Draw(mask)

    # Create a linear gradient from 255 to 0
    gradient = np.linspace(0, 255, rect_width)

    for i in range(rect_width):
        intensity = int(gradient[i])
        fill_color = (intensity,)
        draw.polygon([p1, p2, p3, p4], fill=fill_color)

        # Update corners for next iteration
        p1 = (p1[0] + nx, p1[1] + ny)
        p2 = (p2[0] + nx, p2[1] + ny)
        p3 = (p3[0] + nx, p3[1] + ny)
        p4 = (p4[0] + nx, p4[1] + ny)

    return mask


def create_mask_from_line_reversed(start, end, img_width, img_height, rect_width=4):
    # Calculate the direction vector of the line
    dx = start[0] - end[0]
    dy = start[1] - end[1]

    # Normalize the direction vector
    length = np.sqrt(dx * dx + dy * dy)
    dx /= length
    dy /= length

    # Calculate the normal to the line (perpendicular)
    nx = -dy
    ny = dx

    # Calculate the four corners of the rectangle
    half_width = rect_width / 2.0
    p1 = (start[0] - nx * half_width, start[1] - ny * half_width)
    p2 = (start[0] + nx * half_width, start[1] + ny * half_width)
    p3 = (end[0] + nx * half_width, end[1] + ny * half_width)
    p4 = (end[0] - nx * half_width, end[1] - ny * half_width)

    # Create a blank mask
    mask = Image.new("L", (img_width, img_height), 0)

    # Draw the gradient rectangle on the mask
    draw = ImageDraw.Draw(mask)

    # Create a linear gradient from 255 to 0
    gradient = np.linspace(0, 255, rect_width)

    for i in range(rect_width):
        intensity = int(gradient[i])
        fill_color = (intensity,)
        draw.polygon([p1, p2, p3, p4], fill=fill_color)

        # Update corners for next iteration
        p1 = (p1[0] + nx, p1[1] + ny)
        p2 = (p2[0] + nx, p2[1] + ny)
        p3 = (p3[0] + nx, p3[1] + ny)
        p4 = (p4[0] + nx, p4[1] + ny)

    return mask


# This old funtion draws line by line whoch have aliasing issues
def create_mask_from_line_reverse_OLD(A_screen, B_screen, W, H, width=5, power=4):
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    # Compute the edge direction
    deltaX = B_screen[0] - A_screen[0]
    deltaY = B_screen[1] - A_screen[1]

    # Normalize the direction vector
    length = np.sqrt(deltaX**2 + deltaY**2)
    deltaX /= length
    deltaY /= length

    # Compute the left and right normals
    normal_left = [-deltaY, deltaX]
    normal_right = [deltaY, -deltaX]

    for offset in range(-width, width + 1):
        # Calculate intensity based on the offset and width
        fraction_left = (width + offset) / (
            2 * width
        )  # Linear fraction ranging from 0.5 to 1
        fraction_right = (width - offset) / (
            2 * width
        )  # Linear fraction ranging from 0 to 0.5

        # Apply non-linear adjustment using power
        intensity_left = 255 * (fraction_left**power)
        intensity_right = 255 * (fraction_right**power)

        A_offset_left = (
            A_screen[0] + offset * normal_left[0],
            A_screen[1] + offset * normal_left[1],
        )
        B_offset_left = (
            B_screen[0] + offset * normal_left[0],
            B_screen[1] + offset * normal_left[1],
        )
        draw.line([A_offset_left, B_offset_left], fill=int(intensity_left), width=3)

        A_offset_right = (
            A_screen[0] + offset * normal_right[0],
            A_screen[1] + offset * normal_right[1],
        )
        B_offset_right = (
            B_screen[0] + offset * normal_right[0],
            B_screen[1] + offset * normal_right[1],
        )
        draw.line([A_offset_right, B_offset_right], fill=int(intensity_right), width=3)

    return mask


def load_mesh_obj(fname):
    """
    Load 3D model data from an OBJ file into a Mesh representation.

    The OBJ file format is commonly used to represent 3D models, containing vertices,
    texture coordinates, and face definitions. This function parses these essential
    components to facilitate the use of 3D model data within Python applications.

    The parsing specifically targets:
    - Vertices ("v"): Points in 3D space.
    - Texture Coordinates ("vt"): 2D points used for image mapping onto the 3D model.
    - Faces ("f"): Defines the shape of the model using vertex and texture coordinate indices.

    Note: Indices in OBJ are 1-based, but are converted to 0-based for Python compatibility.

    Parameters:
    - fname (str): The path to the OBJ file to be loaded.

    Returns:
    - Mesh: A mesh representation containing vertices, UV coordinates, and face data.

    Raises:
    - AssertionError: If any face index is not positive or if the format is unexpected.
    """
    mesh = Mesh()

    with open(fname, "r") as f:
        for line in f:
            tokens = line.split()

            if not tokens:
                continue

            if tokens[0] == "v" and len(tokens) == 4:
                x, y, z = map(float, tokens[1:])
                mesh.verts.append(np.array([x, y, z]))

            elif tokens[0] == "vt" and len(tokens) == 3:
                x, y = map(float, tokens[1:])
                mesh.uvs.append(np.array([x, y]))

            elif tokens[0] == "f" and len(tokens) == 4:
                components = tokens[1].split("/")
                if len(components) == 3:
                    v0, uv0, n0 = map(int, components)
                    v1, uv1, n1 = map(int, tokens[2].split("/"))
                    v2, uv2, n2 = map(int, tokens[3].split("/"))

                    assert v0 > 0
                    assert uv0 > 0
                    assert v1 > 0
                    assert uv1 > 0
                    assert v2 > 0
                    assert uv2 > 0

                    mesh.faces.append(
                        np.array([v0 - 1, v1 - 1, v2 - 1, uv0 - 1, uv1 - 1, uv2 - 1])
                    )
                elif len(components) == 2:
                    v0, uv0 = map(int, components)
                    v1, uv1 = map(int, tokens[2].split("/"))
                    v2, uv2 = map(int, tokens[3].split("/"))

                    assert v0 > 0
                    assert uv0 > 0
                    assert v1 > 0
                    assert uv1 > 0
                    assert v2 > 0
                    assert uv2 > 0

                    mesh.faces.append(
                        np.array([v0 - 1, v1 - 1, v2 - 1, uv0 - 1, uv1 - 1, uv2 - 1])
                    )
    return mesh


def load_mesh_glb(filename):
    """
    Load 3D model data from a GLB file into a Mesh representation.

    This function extracts vertices, UV coordinates, and face definitions from a GLB file,
    a binary form of the glTF format, using the trimesh library.

    Parameters:
    - filename (str): The path to the GLB file to be loaded.

    Returns:
    - Mesh: A mesh representation containing vertices, UV coordinates, and face data.
    """
    mesh = Mesh
    scene = trimesh.load(filename, process=False)
    geometries = list(scene.geometry.values())
    geometry = geometries[0]
    mesh.verts = geometry.vertices
    mesh.uvs = geometry.visual.uv
    mesh.faces = geometry.faces

    return mesh


def load_image_diffuse_glb(filename):
    """
    Extracting the diffuse texture from a GLB 3D model.

    Parameters:
    - filename (str): Path to the GLB file.

    Returns:
    - Image or Texture: Model's base color texture.
    """
    scene = trimesh.load(filename, process=False)
    geometries = list(scene.geometry.values())
    geometry = geometries[0]
    return geometry.visual.material.baseColorTexture


def load_image_normal_glb(filename):
    """
    Extracting the normal texture from a GLB 3D model.

    Parameters:
    - filename (str): Path to the GLB file.

    Returns:
    - Image or Texture: Model's normal texture.
    """
    scene = trimesh.load(filename, process=False)
    geometries = list(scene.geometry.values())
    geometry = geometries[0]

    return geometry.visual.material.normalTexture


def find_seam_edges_obj(mesh):
    """
    Identify and store texture seams in the SeamEdge format for enhanced 3D texture mapping.

    Recognizing seams allows for better texture unwrapping and manipulation, ensuring
    that the 3D model retains its intended aesthetic when textured.

    Parameters:
    - mesh (Mesh): A 3D mesh representation with vertices, UVs, and faces.

    Returns:
    - list: Seam edges stored in the SeamEdge format.
    """

    # ... rest of the function ...

    if not len(mesh.faces):
        print("No faces in the mesh!")
        return []

    seam_edges = []
    edge_map = {}

    for f in mesh.faces:
        # Representing edges and their UVs as tuples
        print(f)
        edges = [
            (f[0], f[1], f[3], f[4]),
            (f[1], f[2], f[4], f[5]),
            (f[2], f[0], f[5], f[3]),
        ]

        for e in edges:
            v0, v1 = mesh.verts[e[0]], mesh.verts[e[1]]
            uv0, uv1 = mesh.uvs[e[2]], mesh.uvs[e[3]]

            other_edge = edge_map.get(
                (v1.tobytes(), v0.tobytes())
            )  # using numpy tobytes to hash arrays
            if other_edge is None:
                edge_map[(v0.tobytes(), v1.tobytes())] = (uv0, uv1)
            else:
                other_uv0, other_uv1 = other_edge
                if not np.array_equal(other_uv0, uv1) or not np.array_equal(
                    other_uv1, uv0
                ):
                    s = SeamEdge(HalfEdge(uv0, uv1), HalfEdge(other_uv1, other_uv0))
                    seam_edges.append(s)

                # Remove the edge from the map
                del edge_map[(v1.tobytes(), v0.tobytes())]

    return seam_edges


def find_seam_edges_glb(mesh):
    """
    Identify and store texture seams in the SeamEdge format for enhanced 3D texture mapping.

    Recognizing seams allows for better texture unwrapping and manipulation, ensuring
    that the 3D model retains its intended aesthetic when textured.

    Parameters:
    - mesh (Mesh): A 3D mesh representation with vertices, UVs, and faces.

    Returns:
    - list: Seam edges stored in the SeamEdge format.
    """
    if not len(mesh.faces):
        print("No faces in the mesh!")
        return []

    seam_edges = []  # we define a list that we will populate with pairs of edges
    edge_map = {}  # a pool of edges where we check for pairs

    for f in mesh.faces:
        """Representing edges in the face as tuples"""
        edges = [
            (f[0], f[1]),
            (f[1], f[2]),
            (f[2], f[0]),
        ]

        for e in edges:
            """Extracting pos, and uv for both vertices on each edge"""
            v0, v1 = mesh.verts[e[0]], mesh.verts[e[1]]
            uv0, uv1 = mesh.uvs[e[0]], mesh.uvs[e[1]]

            other_edge = edge_map.get(
                (v1.tobytes(), v0.tobytes())
            )  # using numpy tobytes to hash arrays(position in our case) so we check against hashes
            if other_edge is None:
                edge_map[(v0.tobytes(), v1.tobytes())] = (
                    uv0,
                    uv1,
                )  # we add the uv coord to that hash key in edge_map if we don't find a match
            else:
                (
                    other_uv0,
                    other_uv1,
                ) = other_edge  # If we find a match we reverse the edge
                if not np.array_equal(other_uv0, uv1) or not np.array_equal(
                    other_uv1, uv0
                ):
                    s = SeamEdge(
                        HalfEdge(uv0, uv1), HalfEdge(other_uv1, other_uv0)
                    )  # if a match has been found, we add the pair.
                    seam_edges.append(s)

                # Remove the edge from the map
                del edge_map[(v1.tobytes(), v0.tobytes())]

    return seam_edges


def calculate_step_size(total_edge_seams):
    """
    Determine an optimal step size to evenly distribute colors across seam pairs.

    By calculating the step size, the function aids in allocating distinct colors
    to different seam pairs, ensuring clear visual distinction when rendering or analyzing.

    Parameters:
    - total_edge_seams (int): Total number of seam pairs in the model.

    Returns:
    - int: Optimal step size for color distribution.
    """
    total_combinations = 256 * 256 * 256
    step_size = int((total_combinations / total_edge_seams) ** (1 / 3))
    return max(1, step_size)


def encode_seam_color(seam_count, edge_seams):
    """
    Generate a unique RGB color for a seam pair to ensure clear distinction during visualization.
    The RGB color can be read as information by other apps/tools and be used / manipulated.

    Assigning unique colors to seam pairs enhances model analysis, making it easier
    to identify and work with specific seams in the context of 3D texture mapping.

    Parameters:
    - seam_count (int): The sequential number of the seam pair being processed.
    - edge_seams (int): Total number of seam pairs in the model.

    Returns:
    - tuple: An RGB color (r, g, b) for the seam pair.
    """
    step_size = calculate_step_size(edge_seams)
    # Determine the number of possible values for each channel given the step size
    values_per_channel = 256 // step_size

    # The number of cycles for each channel
    blue_cycles = seam_count % values_per_channel
    green_cycles = (seam_count // values_per_channel) % values_per_channel
    red_cycles = seam_count // (values_per_channel**2)

    # Convert cycle counts into RGB values
    r = red_cycles * step_size
    g = green_cycles * step_size
    b = blue_cycles * step_size

    return (r, g, b)


def generate_seam_map(edge_seams, texture):
    """
    Create a visual representation of the 3D model's texture seams for improved texture mapping analysis.

    By rendering each seam with a distinct color on a 2D texture map, it offers clarity
    and ease during texture unwrapping tasks, ensuring precise texture placements on the 3D model.

    Parameters:
    - edge_seams (list): List of seam pairs.
    - texture (Texture): The texture associated with the 3D model.

    Returns:
    - Image: A 2D representation highlighting the texture seams.
    """
    W, H = texture.width, texture.height
    for seam_count, seam in enumerate(edge_seams, start=1):
        seam.color = encode_seam_color(seam_count, len(edge_seams))
    # Create a new PIL Image and draw each seam pair with its unique color
    img = Image.new("RGB", (W, H))  # define your width and height
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    draw = ImageDraw.Draw(img)

    for seam in edge_seams:
        draw.line(
            [
                (seam.edges[0].a[0] * W, H - seam.edges[0].a[1] * H),
                (seam.edges[0].b[0] * W, H - seam.edges[0].b[1] * H),
            ],
            fill=seam.color,
        )
        draw.line(
            [
                (seam.edges[1].a[0] * W, H - seam.edges[1].a[1] * H),
                (seam.edges[1].b[0] * W, H - seam.edges[1].b[1] * H),
            ],
            fill=seam.color,
        )
    return img


def generate_uv_mask(mesh, texture):
    """
    Generate a mask that indicates the UV coverage of a 3D model on its texture map.

    Creating a UV coverage mask aids in understanding and optimizing texture space utilization,
    ensuring that the texture is applied efficiently and uniformly across the model's surface.

    Parameters:
    - mesh (Mesh): The 3D model's representation with vertices, UVs, and faces.
    - texture (Texture): The texture associated with the 3D model.

    Returns:
    - Image: A 2D mask highlighting the areas covered by the UVs.
    """
    coverage_buf = np.zeros((texture.width, texture.height), dtype=np.uint8)

    for f in mesh.faces:
        # print(f)
        uv0 = mesh.uvs[f[0]]
        uv1 = mesh.uvs[f[1]]
        uv2 = mesh.uvs[f[2]]

        if is_flipped(uv0, uv1, uv2):
            uv1, uv2 = uv2, uv1  # Swap the vertices

        coverage_buf = RasterizeFace(uv0, uv1, uv2, coverage_buf)
    img = Image.fromarray(coverage_buf)
    return img


def generate_sdf_from_image(uv_mask):
    """
    Generate a Signed Distance Field (SDF) from a given UV mask to enable smooth texture boundary interpolations.

    By transforming a UV mask into an SDF, it facilitates advanced rendering techniques such as
    anti-aliased texture borders and procedural texture blending, ensuring visually smooth transitions
    at texture boundaries on 3D models.

    Parameters:
    - uv_mask (Image): A 2D UV coverage mask.

    Returns:
    - Image: A normalized Signed Distance Field derived from the UV mask.
    """
    # Convert uv_mask to binary (0 and 1)
    uv_mask_array = np.array(uv_mask)
    binary_mask = (uv_mask_array > 127).astype(
        np.uint8
    )  # assuming a threshold of 127 for binarization

    # Compute the distance transform for inside and outside the shape
    dist_trans_inside = distance_transform_edt(binary_mask)
    dist_trans_outside = distance_transform_edt(1 - binary_mask)

    # Construct the SDF
    sdf = dist_trans_inside - dist_trans_outside

    # Normalize for visualization if needed (map to [0, 255])
    sdf_normalized = ((sdf - sdf.min()) / (sdf.max() - sdf.min()) * 255).astype(
        np.uint8
    )
    sdf_image = Image.fromarray(sdf_normalized)

    return sdf_image


def encode_normal_to_azimuth_altitude(normal_map):
    """
    Encode the normal directions in a normal map to azimuth and altitude angles.

    Parameters:
    - normal_map (Image): Input normal map.

    Returns:
    - Image: An image where R represents azimuth and G represents altitude.
    """
    # Ensure the input is a valid image
    if not isinstance(normal_map, Image.Image):
        raise ValueError("Input should be a PIL Image.")

    # Convert the normal map to an array
    normal_array = np.array(normal_map).astype(np.float32) / 255.0

    # Convert the [0, 1] range to [-1, 1]
    normal_array = 2 * normal_array - 1

    # Calculate azimuth (θ) and altitude (φ) angles
    azimuth = np.arctan2(normal_array[..., 1], normal_array[..., 0])

    # Clip values to ensure they are within [-1, 1] range
    normal_array[..., 2] = np.clip(normal_array[..., 2], -1, 1)
    altitude = np.arccos(normal_array[..., 2])

    # Normalize angles to [0, 1] range
    azimuth = (azimuth + np.pi) / (2 * np.pi)
    altitude = altitude / np.pi

    # Convert to [0, 255] range
    azimuth = (azimuth * 255).astype(np.uint8)
    altitude = (altitude * 255).astype(np.uint8)

    # Construct the encoded image
    encoded_image = np.zeros_like(normal_array, dtype=np.uint8)
    encoded_image[..., 0] = azimuth
    encoded_image[..., 1] = altitude
    encoded_image[..., 2] = 128  # Set to mid-gray

    # Return the encoded image
    return Image.fromarray(encoded_image)


def pack_uv_mask_into_texture(azimuth_altitude_img, uv_mask):
    """
    Pack the UV coverage mask into the blue channel of the azimuth-altitude texture.

    Parameters:
    - azimuth_altitude_img (Image): The azimuth-altitude encoded texture.
    - uv_mask (Image): The UV coverage mask.

    Returns:
    - Image: The modified azimuth-altitude texture with the UV mask in the blue channel.
    """
    # Convert images to numpy arrays for manipulation
    azimuth_altitude_arr = np.array(azimuth_altitude_img)
    uv_mask_arr = np.array(uv_mask)

    # Extract the red and green channels from the azimuth-altitude texture
    red_channel = azimuth_altitude_arr[..., 0]
    green_channel = azimuth_altitude_arr[..., 1]

    # Create a new image array
    new_image_arr = np.zeros_like(azimuth_altitude_arr)

    # Assign the red, green, and UV mask to the respective channels
    new_image_arr[..., 0] = red_channel
    new_image_arr[..., 1] = green_channel
    new_image_arr[..., 2] = uv_mask_arr

    # Convert the numpy array back to an image
    packed_image = Image.fromarray(new_image_arr)

    return packed_image


def transform_image(image, seam_edge, px_width=10):
    W, H = image.width, image.height
    image_np = np.array(image)

    # image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

    for seam in seam_edge:
        # Convert the UV coordinates to screen (texture) coordinates
        src = [
            UVToScreen(np.array(seam.edges[0].a), W, H),
            UVToScreen(np.array(seam.edges[0].b), W, H),
        ]

        dst = [
            UVToScreen(np.array(seam.edges[1].a), W, H),
            UVToScreen(np.array(seam.edges[1].b), W, H),
        ]

        M_1 = get_affine_matrix(dst, src)

        single_mask_1 = create_mask_from_line_reversed(dst[0], dst[1], W, H, px_width)

        # single_mask.show()
        # Convert the single_mask Image object to a numpy array and normalize it
        mask_np_1 = np.array(single_mask_1).astype(float) / 255.0

        # Apply the transformation
        transformed_image_np_1 = np.array(
            image.transform(
                image.size, Image.AFFINE, data=M_1.ravel()[:6], resample=Image.BILINEAR
            )
        )
        # Blend the images
        blended_np = (
            image_np * (1 - mask_np_1[:, :, None])
            + transformed_image_np_1 * mask_np_1[:, :, None]
        )

        image_np = blended_np.astype(np.uint8)

        # If you want to see the intermediate results, convert back to Image and show:
        # Image.fromarray(transformed_image_np).show()
    for seam in seam_edge:
        # Convert the UV coordinates to screen (texture) coordinates
        src = [
            UVToScreen(np.array(seam.edges[0].a), W, H),
            UVToScreen(np.array(seam.edges[0].b), W, H),
        ]

        dst = [
            UVToScreen(np.array(seam.edges[1].a), W, H),
            UVToScreen(np.array(seam.edges[1].b), W, H),
        ]

        M_2 = get_affine_matrix(src, dst)

        single_mask_2 = create_mask_from_line(src[0], src[1], W, H, px_width)

        # Convert the single_mask Image object to a numpy array and normalize it
        mask_np_2 = np.array(single_mask_2).astype(float) / 255.0
        # Apply the transformation

        transformed_image_np_2 = np.array(
            image.transform(
                image.size, Image.AFFINE, data=M_2.ravel()[:6], resample=Image.BILINEAR
            )
        )
        # Blend the images
        blended_np = (
            image_np * (1 - mask_np_2[:, :, None])
            + transformed_image_np_2 * mask_np_2[:, :, None]
        )
        image_np = blended_np.astype(np.uint8)

        # If you want to see the intermediate results, convert back to Image and show:
        # Image.fromarray(transformed_image_np).show()

    return Image.fromarray(image_np)


def compute_object_space_normals(mesh):
    """
    Compute the object space normals for a mesh.

    Args:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        np.ndarray: Array of object space normals for each vertex.
    """
    normals = np.zeros_like(mesh.verts)

    for face in mesh.faces:
        # Calculate the triangle's normal
        v0, v1, v2 = mesh.verts[face]
        normal = np.cross(v1 - v0, v2 - v0)
        normal /= np.linalg.norm(normal)  # Normalize

        # Assign/accumulate the triangle's normal to its vertices
        normals[face] += normal

    # Normalize the accumulated normals
    for i in range(len(normals)):
        normals[i] /= np.linalg.norm(normals[i])

    return normals


def bake_object_space_normals(mesh, object_space_normals, image):
    """
    Bake the object space normals of a mesh into a 2D texture map.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        uv_data (np.array): The UV coordinates of the mesh vertices.
        image_size (tuple): Size of the target image as (width, height).

    Returns:
        np.ndarray: Baked normal map.
    """
    uv_data = mesh.uvs
    W, H = image.size

    # Compute object space normals
    normals = object_space_normals

    # Flatten UV space to compute barycentric coordinates for all points
    y, x = np.mgrid[:H, :W]
    uv_points = np.column_stack((x.ravel(), y.ravel()))

    # Normalize UV points
    uv_points_normalized = np.column_stack(
        (uv_points[:, 0] / W, 1 - uv_points[:, 1] / H)
    )

    baked_normals = np.zeros((W * H, 3), dtype=np.float32)

    for face in mesh.faces:
        v0_uv, v1_uv, v2_uv = uv_data[face]
        bcoords = compute_barycentric_coords_vectorized(
            uv_points_normalized, v0_uv, v1_uv, v2_uv
        )

        # Mask for the points inside the triangle
        mask = np.all((bcoords >= 0) & (bcoords <= 1), axis=1)

        # Use broadcasting to compute the interpolated normals for all relevant UVs
        interpolated_normals = (
            bcoords[mask][:, 0, np.newaxis] * normals[face[0]]
            + bcoords[mask][:, 1, np.newaxis] * normals[face[1]]
            + bcoords[mask][:, 2, np.newaxis] * normals[face[2]]
        )

        # Normalize the interpolated normals
        normalized_normals = (
            interpolated_normals
            / np.linalg.norm(interpolated_normals, axis=1)[:, np.newaxis]
        )
        baked_normals[mask] = normalized_normals

    # Reshape the flattened array back to image dimensions
    baked_normals = baked_normals.reshape(H, W, 3)

    # Normalize the normals to fit in [0, 1] range
    baked_normals = 0.5 * (baked_normals + 1)
    return (baked_normals * 255).astype(np.uint8)


def main(obj_filename="InputMesh.glb"):
    """
    Generate visualization assets from a 3D model to streamline the texture mapping process.

    By producing visual aids like a seam map, UV mask, and SDF mask, it facilitates a deeper
    understanding of texture utilization on a 3D model, promoting efficient texture mapping
    and improved rendering outcomes.

    Parameters:
    - obj_filename (str): Path to the 3D model in GLB format.

    Outputs:
    - Saves visualization assets (seam map, UV mask, and SDF mask) as PNG images.
    """
    # loading data
    mesh = load_mesh_glb(obj_filename)
    diffuse_texture = load_image_diffuse_glb(obj_filename)
    normal_texture = load_image_normal_glb(obj_filename)

    object_space_normals = compute_object_space_normals(mesh)

    # Generating maps
    edge_seams = find_seam_edges_glb(mesh)

    seam_map = generate_seam_map(edge_seams, diffuse_texture)
    uv_mask = generate_uv_mask(mesh, diffuse_texture)
    sdf_mask = generate_sdf_from_image(uv_mask)

    baked_normals_image = bake_object_space_normals(
        mesh, object_space_normals, diffuse_texture
    )

    # image should be only at the end

    Object_normal = Image.fromarray(baked_normals_image)

    az_alt_map = encode_normal_to_azimuth_altitude(Object_normal)
    packed_azimuth_uv_mask = pack_uv_mask_into_texture(az_alt_map, uv_mask)
    # test = transform_image(
    #     diffuse_texture, edge_seams, 8
    # )  # the last param indicates the edge thickness

    # Saving maps
    seam_map.save("seam_map.png")
    uv_mask.save("uv_mask.png")
    sdf_mask.save("sdf_mask.png")
    az_alt_map.save("azimuth_altitute_map.png")
    normal_texture.save("normal_texture.png")
    diffuse_texture.save("diffuse_texture.png")
    Object_normal.save("Object_normal.png")
    packed_azimuth_uv_mask.save("packed_az_uv_mask.png")

    # Visualizing
    # test.save()
    # test.show()


if __name__ == "__main__":
    """
    Entry point of the script when run as a standalone application.

    Allows the script to be both imported as a module in other scripts or executed directly
    to generate visualization assets for a given 3D model. The filename of the 3D model
    can be specified by passing it as an argument to the main function (e.g., main("custom_filename.glb")).
    """
    main()
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")
