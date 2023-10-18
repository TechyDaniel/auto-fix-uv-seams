from PIL import Image, ImageDraw
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt


class HalfEdge:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class SeamEdge:
    def __init__(self, edge1, edge2):
        self.edges = [edge1, edge2]
        self.color = []


class Mesh:
    def __init__(self):
        self.verts = []
        self.uvs = []
        self.faces = []


def UVToScreen(in_vec, W, H):
    out_vec = in_vec.copy()
    out_vec[1] = 1.0 - out_vec[1]
    out_vec[0] *= W
    out_vec[1] *= H
    return out_vec - np.array([0.5, 0.5])


def is_flipped(uv0, uv1, uv2):
    return (uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (uv2[0] - uv0[0]) * (
        uv1[1] - uv0[1]
    ) < 0


def isInside(x, y, ea, eb):
    return (
        x * (eb[1] - ea[1]) - y * (eb[0] - ea[0]) - ea[0] * eb[1] + ea[1] * eb[0] >= 0
    )


def WrapCoordinate(x, size):
    while x < 0:
        x += size
    while x >= size:
        x -= size
    return x


def RasterizeFace(uv0, uv1, uv2, coverageBuf):
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


# Alternatively we can use tinyGltf lib to load GLB
def load_mesh_obj(fname):
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
    mesh = Mesh
    scene = trimesh.load(filename, process=False)
    geometries = list(scene.geometry.values())
    geometry = geometries[0]
    mesh.verts = geometry.vertices
    mesh.uvs = geometry.visual.uv
    mesh.faces = geometry.faces

    return mesh


def load_image_glb(filename):
    scene = trimesh.load(filename, process=False)
    geometries = list(scene.geometry.values())
    geometry = geometries[0]
    return geometry.visual.material.baseColorTexture


def find_seam_edges_obj(mesh):
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
    """Cycles though the faces and finds edges that share pos but not uv coord."""
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
    """Calculates the step size based on max seams found"""
    total_combinations = 256 * 256 * 256
    step_size = int((total_combinations / total_edge_seams) ** (1 / 3))
    return max(1, step_size)


def encode_seam_color(seam_count, edge_seams):
    "Give each seam pair an unique color"
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


def main(obj_filename="InputMesh.glb"):
    mesh = load_mesh_glb(obj_filename)
    texture = load_image_glb(obj_filename)
    edge_seams = find_seam_edges_glb(mesh)

    seam_map = generate_seam_map(edge_seams, texture)
    uv_mask = generate_uv_mask(mesh, texture)
    sdf_mask = generate_sdf_from_image(uv_mask)

    seam_map.save("seam_map.png")
    uv_mask.save("uv_mask.png")
    sdf_mask.save("sdf_mask.png")
    # seam_map.show()


if __name__ == "__main__":
    main()
