from PIL import Image, ImageDraw
import numpy as np


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


def load_mesh(fname):
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


def find_seam_edges(mesh):
    if not mesh.faces:
        print("No faces in the mesh!")
        return []

    seam_edges = []
    edge_map = {}

    for f in mesh.faces:
        # Representing edges and their UVs as tuples
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


def calculate_step_size(total_edge_seams):
    """Calculates the step size based on max seams found"""
    total_combinations = 256 * 256 * 256
    step_size = int((total_combinations / total_edge_seams) ** (1 / 3))
    return max(1, step_size)


def encode_seam_color(seam_count, edge_seams):
    step_size = calculate_step_size(edge_seams)
    print(step_size)
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

    print(seam_count, r, g, b)
    return (r, g, b)


def main(obj_filename):
    mesh = load_mesh(obj_filename)
    edge_seams = find_seam_edges(mesh)

    # A counter for the seam pairs to encode the RGB value
    seam_count = 1
    print(len(edge_seams))
    for seam in edge_seams:
        seam.color = encode_seam_color(seam_count, len(edge_seams))
        seam_count += 1

    # Create a new PIL Image and draw each seam pair with its unique color
    img = Image.new("RGB", (512, 512))  # define your width and height
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    draw = ImageDraw.Draw(img)

    for seam in edge_seams:
        draw.line(
            [
                (seam.edges[0].a[0] * 512, 512 - seam.edges[0].a[1] * 512),
                (seam.edges[0].b[0] * 512, 512 - seam.edges[0].b[1] * 512),
            ],
            fill=seam.color,
        )
        draw.line(
            [
                (seam.edges[1].a[0] * 512, 512 - seam.edges[1].a[1] * 512),
                (seam.edges[1].b[0] * 512, 512 - seam.edges[1].b[1] * 512),
            ],
            fill=seam.color,
        )
    img.save("SeamMap.png")


if __name__ == "__main__":
    main("InputMesh.obj")
