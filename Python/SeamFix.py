import numpy as np
import struct
from scipy.sparse import csr_matrix
import math
from PIL import Image
import time
import sys
from pygltflib import GLTF2, BufferFormat

EDGE_CONSTRAINTS_WEIGHT = 5.0
COVERED_PIXELS_WEIGHT = 2.0
NONCOVERED_PIXELS_WEIGHT = 0.1
TOLERANCE = 0.1

USE_ISPC = 0


def min(x, y):
    return x if x < y else y


def max(x, y):
    return x if x > y else y


def min3(x, y, z):
    return min(x, min(y, z))


def max3(x, y, z):
    return max(x, max(y, z))


class Array2D:
    def __init__(self, width, height, zero=0):
        self.data = np.full((height, width), zero)
        self.width = width
        self.height = height

    def get(self, column, row):
        return self.data[row, column]

    def set(self, column, row, value):
        self.data[row, column] = value


class VectorX:
    def __init__(self, size, initial_value=0.0):
        self.vec = np.full(size, initial_value, dtype=float)

    def __getitem__(self, ix):
        return self.vec[ix]

    def __setitem__(self, ix, value):
        self.vec[ix] = value

    def __add__(self, other):
        return VectorX.from_numpy(self.vec + other.vec)

    def __sub__(self, other):
        return VectorX.from_numpy(self.vec - other.vec)

    def __mul__(self, scalar):
        return VectorX.from_numpy(self.vec * scalar)

    @classmethod
    def from_numpy(cls, np_array):
        instance = cls(np_array.size)
        instance.vec = np_array
        return instance

    @staticmethod
    def dot(a, b):
        return np.dot(a.vec, b.vec)

    @staticmethod
    def add(out, x, y):
        out.vec = x.vec + y.vec

    @staticmethod
    def sub(out, x, y):
        out.vec = x.vec - y.vec

    @staticmethod
    def mul(out, x, scalar):
        out.vec = x.vec * scalar

    @staticmethod
    def mul_add(out, v, a, b):
        out.vec = v.vec * a + b.vec


class SparseMat:
    def __init__(self, numRows, numCols):
        self.numRows = numRows
        self.numCols = numCols
        self.data = []
        self.indices = []
        self.indptr = [0] * (numRows + 1)

    def insert(self, row, col, val):
        assert row < self.numRows and row >= 0
        assert col < self.numCols and col >= 0

        # Check if there's already a value at (row, col).
        for ix, idx in enumerate(self.indices[self.indptr[row] : self.indptr[row + 1]]):
            if idx == col:
                self.data[self.indptr[row] + ix] = val
                return

        # Inserting (val, col) into self.data and self.indices.
        self.data.insert(self.indptr[row], val)
        self.indices.insert(self.indptr[row], col)

        # Increment subsequent pointers.
        for i in range(row + 1, self.numRows + 1):
            self.indptr[i] += 1

    def as_csr(self):
        return csr_matrix(
            (self.data, self.indices, self.indptr), shape=(self.numRows, self.numCols)
        )

    @staticmethod
    def mul(out, A, x):
        assert out.vec.size == A.numRows
        assert x.vec.size == A.numCols

        A_csr = A.as_csr()
        out.vec = A_csr.dot(x.vec)


class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return np.isclose([self.x, self.y, self.z], [other.x, other.y, other.z]).all()

    def __ne__(self, other):
        return not (self == other)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return self * (1.0 / scalar)


class Vec2:
    def __init__(self, u=0.0, v=0.0):
        self.u = u
        self.v = v

    def __eq__(self, other):
        return math.isclose(self.u, other.u) and math.isclose(self.v, other.v)

    def __ne__(self, other):
        return not (self == other)

    def __sub__(self, other):
        return Vec2(self.u - other.u, self.v - other.v)

    def __add__(self, other):
        return Vec2(self.u + other.u, self.v + other.v)

    def __iadd__(self, other):
        self.u += other.u
        self.v += other.v
        return self

    def __mul__(self, scalar):
        return Vec2(self.u * scalar, self.v * scalar)

    def __truediv__(self, scalar):
        return self * (1.0 / scalar)

    def length(self):
        return math.sqrt(self.u**2 + self.v**2)


def vec2_mul(scalar, vec):
    return vec * scalar


class Face:
    def __init__(self, v0=0, v1=0, v2=0, uv0=0, uv1=0, uv2=0):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.uv0 = uv0
        self.uv1 = uv1
        self.uv2 = uv2


class HalfEdge:
    def __init__(self, a=None, b=None):
        if a is None:
            a = Vec2()
        if b is None:
            b = Vec2()
        self.a = a
        self.b = b


def uv_to_screen(in_uv, W, H):
    in_uv = np.array(in_uv)
    in_uv[1] = 1.0 - in_uv[1]
    in_uv[0] *= W
    in_uv[1] *= H
    return in_uv - np.array([0.5, 0.5])


def wrap_coordinate(x, size):
    """
    Do a modulo operation with the assumption that it's usually a no-op.
    Note: this guarantees positive return values.
    """
    # Branch predictor will hopefully skip these two loops most of the time
    while x < 0:
        x += size
    while x >= size:
        x -= size
    return x


class SeamEdge:
    def __init__(self):
        self.edges = [HalfEdge(), HalfEdge()]

    def num_samples(self, W, H):
        e0 = uv_to_screen(self.edges[0].b, W, H) - uv_to_screen(self.edges[0].a, W, H)
        e1 = uv_to_screen(self.edges[1].b, W, H) - uv_to_screen(self.edges[1].a, W, H)
        len_ = max3(2, e0.length(), e1.length())
        return int(len_ * 3)


# class Mesh:

#         faces = []
#         verts = []
#         uvs = []

# replaced by pywavefront
# def load_mesh(fname):
#     try:
#         with open(fname, "r") as f:
#             for line in f:
#                 tokens = line.split()
#                 if not tokens:
#                     continue

#                 if tokens[0] == 'v':
#                     x, y, z = map(float, tokens[1:4])
#                     mesh.verts.append(Vec3(x, y, z))
#                 elif tokens[0] == 'vt':
#                     u, v = map(float, tokens[1:3])
#                     mesh.uvs.append(Vec2(u, v))
#                 elif tokens[0] == 'f':
#                     tri0, uv0, tri1, uv1, tri2, uv2 = map(int, [x for token in tokens[1:4] for x in token.split('/')])
#                     mesh.faces.append(Face(tri0-1, tri1-1, tri2-1, uv0-1, uv1-1, uv2-1))
#     except FileNotFoundError:
#         return False
#     return True


# this is the OLD function
# def find_seam_edges(mesh, W, H):
#     seam_edges = []
#     edge_map = {}

#     for f in mesh.faces:
#         # Define the edges of the face
#         edges = [
#             (f.v0, f.v1, f.uv0, f.uv1),
#             (f.v1, f.v2, f.uv1, f.uv2),
#             (f.v2, f.v0, f.uv2, f.uv0)
#         ]

#         for e in edges:
#             v0, v1, uv0, uv1 = mesh.verts[e[0]], mesh.verts[e[1]], mesh.uvs[e[2]], mesh.uvs[e[3]]

#             # Try to find the opposite edge
#             other_edge = edge_map.pop((v1, v0), None)
#             if other_edge is None:
#                 # If edge is not found, store this edge in the map
#                 edge_map[(v0, v1)] = (uv0, uv1)
#             else:
#                 other_uv0, other_uv1 = other_edge
#                 if other_uv0 != uv1 or other_uv1 != uv0:
#                     # If UVs don't match, we found a seam
#                     seam_edge = SeamEdge(HalfEdge(uv0, uv1), HalfEdge(other_uv1, other_uv0))
#                     seam_edges.append(seam_edge)


#     return seam_edges
def access_data(accessor, glb: GLTF2):
    """
    Retrieves binary data from a glb accessor.

    Parameters:
        accessor: The accessor object containing information about the data format and location.
        glb: The glb2 object containing the .glb file information.

    Returns:
        numpy.ndarray: The data as a NumPy array.
    """
    buffer_view = glb.bufferViews[accessor.bufferView]
    buffer = glb.buffers[buffer_view.buffer]

    # Obtain the byte offset and length of the data
    byte_offset = buffer_view.byteOffset + (accessor.byteOffset or 0)
    byte_length = (
        accessor.count * np.dtype(accessor.componentType).itemsize * accessor.type.count
    )

    # Get the raw binary data from the buffer
    raw_data = buffer.uri[byte_offset : byte_offset + byte_length]

    # Convert the raw data to a NumPy array
    data = np.frombuffer(raw_data, dtype=accessor.componentType)

    # Reshape the array according to accessor details
    if accessor.type != "SCALAR":
        data = data.reshape((accessor.count, accessor.type.count))

    return data


def extract_mesh_data(file_path):
    glb = GLTF2().load(file_path)

    # We can iterrate over meshes and over primitives, but let's keep it simple.
    mesh = glb.meshes[0].primitives

    # Assume mesh 0 and primitive 0, adjust as needed
    primitive = glb.meshes[0].primitives[0]

    # Get vertices
    # for primitive in mesh.primitives:
    # get the binary data for this mesh primitive from the buffer
    accessor = glb.accessors[primitive.attributes.POSITION]
    bufferView = glb.bufferViews[accessor.bufferView]
    buffer = glb.buffers[bufferView.buffer]
    data = glb.get_data_from_buffer_uri(buffer.uri)

    # pull each vertex from the binary buffer and convert it into a tuple of python floats
    vertices = []
    for i in range(accessor.count):
        index = (
            bufferView.byteOffset + accessor.byteOffset + i * 12
        )  # the location in the buffer of this vertex
        d = data[index : index + 12]  # the vertex data
        v = struct.unpack("<fff", d)  # convert from base64 to three floats
        vertices.append(v)

    verts = np.array(vertices)

    # Get faces
    # Get binary data for indices
    index_accessor = glb.accessors[primitive.indices]
    index_buffer_view = glb.bufferViews[index_accessor.bufferView]
    index_buffer = glb.buffers[index_buffer_view.buffer]
    index_data = glb.get_data_from_buffer_uri(index_buffer.uri)

    # Extract faces (vertex indices)
    faces = []
    for i in range(index_accessor.count // 3):  # assuming triangles
        index = (
            index_buffer_view.byteOffset + index_accessor.byteOffset + i * 6
        )  # assuming uint16 type
        face = struct.unpack("<HHH", index_data[index : index + 6])
        faces.append(face)
    np_faces = np.array(faces)
    # Get UVs
    # Get binary data for UVs
    uv_accessor = glb.accessors[primitive.attributes.TEXCOORD_0]
    uv_buffer_view = glb.bufferViews[uv_accessor.bufferView]
    uv_buffer = glb.buffers[uv_buffer_view.buffer]
    uv_data = glb.get_data_from_buffer_uri(uv_buffer.uri)

    # Extract UVs
    uvs = []
    for i in range(uv_accessor.count):
        index = (
            uv_buffer_view.byteOffset + uv_accessor.byteOffset + i * 8
        )  # assuming vec2 of float32 type
        uv = struct.unpack("<ff", uv_data[index : index + 8])
        uvs.append(uv)

    np_uvs = np.array(uvs)

    return {
        "verts": verts.tolist(),
        "faces": np_faces.tolist(),
        "uvs": np_uvs.tolist(),
    }


def find_seam_edges(mesh, W, H):
    seam_edges = []

    edge_map = {}
    for face in mesh["faces"]:
        edges = [
            (face[0], face[1], mesh["uvs"][face[0]], mesh["uvs"][face[1]]),
            (face[1], face[2], mesh["uvs"][face[1]], mesh["uvs"][face[2]]),
            (face[2], face[0], mesh["uvs"][face[2]], mesh["uvs"][face[0]]),
        ]

        for edge in edges:
            v0, v1, uv0, uv1 = edge

            # Ensure a consistent order of vertices
            if v1 < v0:
                v0, v1, uv0, uv1 = v1, v0, uv1, uv0

            # Ensure that uv0 and uv1 are tuples so that they can be used in dictionary keys
            uv0, uv1 = tuple(uv0), tuple(uv1)
            current_edge = (v0, v1, uv0, uv1)
            opposite_edge = (v1, v0, uv1, uv0)

            if opposite_edge not in edge_map:
                # First time we're seeing this edge, so add it to the map
                edge_map[current_edge] = (uv0, uv1)
            else:
                # This edge has already been added once
                other_uv0, other_uv1 = edge_map[opposite_edge]
                if not np.allclose(uv0, other_uv1) or not np.allclose(uv1, other_uv0):
                    # UVs don't match, so we have a seam
                    seam_edges.append(
                        {
                            "half_edge_1": {"uv0": uv0, "uv1": uv1},
                            "half_edge_2": {"uv0": other_uv1, "uv1": other_uv0},
                        }
                    )

                # No longer need this edge
                del edge_map[opposite_edge]
    return seam_edges


class RGB:
    def __init__(self, R, G, B):
        self.channels = [R, G, B]

    def __getitem__(self, ix):
        return self.channels[ix]

    def __setitem__(self, ix, value):
        self.channels[ix] = value


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def rgb_to_y_co_cg(color):
    r, g, b = color

    return (
        r * 0.25 + g * 0.5 + b * 0.25,
        -r * 0.25 + g * 0.5 - b * 0.25,
        r * 0.5 - b * 0.5,
    )


def y_co_cg_to_rgb(color):
    y, co, cg = color

    r = clamp(y - co + cg, 0, 255)
    g = clamp(y + co, 0, 255)
    b = clamp(y - co - cg, 0, 255)

    return (round(r), round(g), round(b))


def is_inside(x, y, ea, eb):
    return (
        x * (eb[1] - ea[1]) - y * (eb[0] - ea[0]) - ea[0] * eb[1] + ea[1] * eb[0] >= 0
    )


class CoverageBuffer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = [[0] * width for _ in range(height)]

    def __setitem__(self, index, value):
        x, y = index
        self.data[y][x] = value

    def __getitem__(self, index):
        x, y = index
        return self.data[y][x]


def RasterizeFace(uv0, uv1, uv2, coverageBuf):
    # Using .shape to get width and height instead of non-existent .width and .height
    width, height = coverageBuf.shape[1], coverageBuf.shape[0]

    uv0 = uv_to_screen(uv0, width, height)
    uv1 = uv_to_screen(uv1, width, height)
    uv2 = uv_to_screen(uv2, width, height)

    # Axis aligned bounds of the triangle
    minx = int(np.min([uv0[0], uv1[0], uv2[0]]))
    maxx = int(np.max([uv0[0], uv1[0], uv2[0]]) + 1)
    miny = int(np.min([uv0[1], uv1[1], uv2[1]]))
    maxy = int(np.max([uv0[1], uv1[1], uv2[1]]) + 1)

    # The three edges we will test
    e0a, e0b = uv0, uv1
    e1a, e1b = uv1, uv2
    e2a, e2b = uv2, uv0

    # Now just loop over a screen aligned bounding box around the triangle, and test each pixel against all three edges
    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            if (
                is_inside(x, y, e0a, e0b)
                and is_inside(x, y, e1a, e1b)
                and is_inside(x, y, e2a, e2b)
            ):
                coverageBuf[
                    wrap_coordinate(x, width),
                    wrap_coordinate(y, height),
                ] = 1


def DilatePixel(centerx, centery, image, coverageBuf):
    numPixels = 0
    sum = Vec3(0, 0, 0)

    for yix in range(centery - 1, centery + 2):
        for xix in range(centerx - 1, centerx + 2):
            x = wrap_coordinate(xix, image.width)
            y = wrap_coordinate(yix, image.height)

            if coverageBuf[x, y]:
                numPixels += 1
                c = image[x, y]
                sum += Vec3(float(c.R), float(c.G), float(c.B))

    if numPixels > 0:
        sum = sum / float(numPixels)
        sum.x = min(255.0, round(sum.x))
        sum.y = min(255.0, round(sum.y))
        sum.z = min(255.0, round(sum.z))

        return RGB(int(sum.x), int(sum.y), int(sum.z))
    else:
        return RGB(0, 0, 0)


def calculate_samples_and_weights(pixel_map, sample):
    truncu = int(sample.u)
    truncv = int(sample.v)

    xs = np.array([truncu, truncu + 1, truncu + 1, truncu])
    ys = np.array([truncv, truncv, truncv + 1, truncv + 1])

    out_ixs = np.zeros(4, dtype=int)
    for i in range(4):
        x = wrap_coordinate(xs[i], pixel_map.shape[1])
        y = wrap_coordinate(ys[i], pixel_map.shape[0])
        out_ixs[i] = pixel_map[y, x]

    fracX = sample.u - truncu
    fracY = sample.v - truncv
    out_weights = (
        np.array(
            [
                (1.0 - fracX) * (1.0 - fracY),
                fracX * (1.0 - fracY),
                fracX * fracY,
                (1.0 - fracX) * fracY,
            ]
        )
        * EDGE_CONSTRAINTS_WEIGHT
    )

    return out_ixs, out_weights


def rcp(x):
    return np.reciprocal(x).astype(np.float32)


def conjugate_gradient_optimize(out, A, guess, b, num_iterations, tolerance):
    n = len(guess)
    x = out

    # r = b - A @ x
    r = b - A @ x

    p = r.copy()
    rsq = np.dot(r, r)

    for i in range(num_iterations):
        # Ap = A @ p
        Ap = A @ p

        alpha = rsq / np.dot(p, Ap)

        # x = x + alpha*p
        x += alpha * p

        # r = r - alpha*Ap
        r -= alpha * Ap

        rsqnew = np.dot(r, r)

        if np.abs(rsqnew - rsq) < tolerance * n:
            break

        beta = rsqnew / rsq

        # p = r + beta*p
        p = r + beta * p

        rsq = rsqnew


class PixelInfo:
    def __init__(self, x, y, is_covered, color_y_cocg):
        self.x = x
        self.y = y
        self.is_covered = is_covered
        self.color_y_cocg = color_y_cocg


def compute_pixel_info(seam_edges, coverage_buf, image, pixel_to_pixel_info_map):
    W, H = coverage_buf.shape
    pixel_info = []

    for s in seam_edges:
        # TODO: do conservative rasterization instead of brute force sampling
        num_samples = s.num_samples(W, H)

        for e in s.edges:
            e0 = uv_to_screen(e.a, W, H)
            e1 = uv_to_screen(e.b, W, H)
            dt = (e1 - e0) / (num_samples - 1)
            sample_point = e0

            for _ in range(num_samples):
                xs = np.array(
                    [
                        int(sample_point[0]),
                        int(sample_point[0]) + 1,
                        int(sample_point[0]) + 1,
                        int(sample_point[0]),
                    ]
                )
                ys = np.array(
                    [
                        int(sample_point[1]),
                        int(sample_point[1]),
                        int(sample_point[1]) + 1,
                        int(sample_point[1]) + 1,
                    ]
                )

                for tap in range(4):
                    x = wrap_coordinate(xs[tap], W)
                    y = wrap_coordinate(ys[tap], H)

                    if pixel_to_pixel_info_map[x, y] == -1:
                        is_covered = bool(coverage_buf[x, y])

                        if is_covered:
                            color_y_cocg = rgb_to_y_co_cg(image[x, y])
                        else:
                            # Do dilation...
                            color_y_cocg = rgb_to_y_co_cg(
                                DilatePixel(x, y, image, coverage_buf)
                            )

                        pixel_info.append(PixelInfo(x, y, is_covered, color_y_cocg))
                        pixel_to_pixel_info_map[x, y] = len(pixel_info) - 1

                sample_point += dt
    return pixel_info


def setup_least_squares(
    seam_edges,
    pixel_to_pixel_info_map,
    pixel_info,
    AtA,
    AtbR,
    AtbG,
    AtbB,
    initial_guess_R,
    initial_guess_G,
    initial_guess_B,
):
    W, H = pixel_to_pixel_info_map.shape

    for seam_index, s in enumerate(seam_edges):
        # Step through the samples of this edge, and compute sample locations for each side of the seam
        num_samples = s.num_samples(W, H)

        first_half_edge_start = uv_to_screen(s.edges[0].a, W, H)
        first_half_edge_end = uv_to_screen(s.edges[0].b, W, H)
        second_half_edge_start = uv_to_screen(s.edges[1].a, W, H)
        second_half_edge_end = uv_to_screen(s.edges[1].b, W, H)

        first_half_edge_step = (first_half_edge_end - first_half_edge_start) / float(
            num_samples - 1
        )
        second_half_edge_step = (second_half_edge_end - second_half_edge_start) / float(
            num_samples - 1
        )

        first_half_edge_sample = first_half_edge_start
        second_half_edge_sample = second_half_edge_start

        for _ in range(num_samples):
            # Sample locations for the two corresponding sets of sample points
            first_half_edge = [0] * 4
            second_half_edge = [0] * 4
            first_half_edge_weights = [0] * 4
            second_half_edge_weights = [0] * 4

            calculate_samples_and_weights(
                pixel_to_pixel_info_map,
                first_half_edge_sample,
                first_half_edge,
                first_half_edge_weights,
            )
            calculate_samples_and_weights(
                pixel_to_pixel_info_map,
                second_half_edge_sample,
                second_half_edge,
                second_half_edge_weights,
            )

            for i in range(4):
                for j in range(4):
                    # + a*a^t
                    AtA[first_half_edge[i], first_half_edge[j]] += (
                        first_half_edge_weights[i] * first_half_edge_weights[j]
                    )
                    # + b*b^t
                    AtA[second_half_edge[i], second_half_edge[j]] += (
                        second_half_edge_weights[i] * second_half_edge_weights[j]
                    )
                    # - a*b^t
                    AtA[first_half_edge[i], second_half_edge[j]] -= (
                        first_half_edge_weights[i] * second_half_edge_weights[j]
                    )
                    # - b*a^t
                    AtA[second_half_edge[i], first_half_edge[j]] -= (
                        second_half_edge_weights[i] * first_half_edge_weights[j]
                    )

            first_half_edge_sample += first_half_edge_step
            second_half_edge_sample += second_half_edge_step

    for i, pi in enumerate(pixel_info):
        # Set up equality cost
        weight = COVERED_PIXELS_WEIGHT if pi.is_covered else NONCOVERED_PIXELS_WEIGHT
        AtA[i, i] += weight

        # Set up the three right hand sides (one for R, G, and B).
        AtbR[i] += pi.color_y_cocg[0] * weight
        AtbG[i] += pi.color_y_cocg[1] * weight
        AtbB[i] += pi.color_y_cocg[2] * weight

        # Set up the initial guess for the solution.
        initial_guess_R[i] = pi.color_y_cocg[0]
        initial_guess_G[i] = pi.color_y_cocg[1]
        initial_guess_B[i] = pi.color_y_cocg[2]


def main():
    t0 = time.time()

    # if len(argv) != 4:
    # print("No arguments provided or incorrect number of arguments. Using default values.")
    mesh_file_name = "InputMesh.glb"
    tex_file_name = "InputTexture.png"
    tex_out_file_name = "OutputTexture.png"
    # else:
    #     mesh_file_name = argv[1]
    #     tex_file_name = argv[2]
    #     tex_out_file_name = argv[3]

    # Load mesh
    glb = GLTF2().load(mesh_file_name)
    if glb is None:
        print("Error loading glb file!")
        return 1

    # Load image using PIL and convert it to NumPy array
    raw_img = Image.open(tex_file_name)
    image = np.array(raw_img)

    if image is None or len(image.shape) != 3:
        print("Failed to load input texture!")
        return -1

    H, W, _ = image.shape
    mesh = extract_mesh_data(mesh_file_name)
    # Find all edges that have different UVs on the two sides
    seam_edges = find_seam_edges(mesh, W, H)

    # Produce a mask for all valid pixels
    coverage_buf = np.zeros((W, H), dtype=np.uint8)

    for face in mesh["faces"]:
        uv0, uv1, uv2 = mesh["uvs"][face[0]], mesh["uvs"][face[1]], mesh["uvs"][face[2]]
        RasterizeFace(uv0, uv1, uv2, coverage_buf)

    pixel_to_pixel_info_map = np.full((W, H), -1, dtype=int)

    pixel_info = compute_pixel_info(
        seam_edges, coverage_buf, image, pixel_to_pixel_info_map
    )
    num_pixels_to_optimize = len(pixel_info)

    # Set up matrices and vectors for least squares problem
    (
        AtA,
        AtbR,
        AtbG,
        AtbB,
        initial_guessR,
        initial_guessG,
        initial_guessB,
    ) = setup_least_squares(
        seam_edges, pixel_to_pixel_info_map, pixel_info, num_pixels_to_optimize
    )

    # Run conjugate gradient optimization for each color channel
    solutionR, solutionG, solutionB = (
        np.zeros(num_pixels_to_optimize) for _ in range(3)
    )

    conjugate_gradient_optimize(solutionR, AtA, initial_guessR, AtbR, 10000, TOLERANCE)
    conjugate_gradient_optimize(solutionG, AtA, initial_guessG, AtbG, 10000, TOLERANCE)
    conjugate_gradient_optimize(solutionB, AtA, initial_guessB, AtbB, 10000, TOLERANCE)

    # Store the resulting optimized pixels
    for i in range(num_pixels_to_optimize):
        pi = pixel_info[i]
        colorYCoCg = np.array([solutionR[i], solutionG[i], solutionB[i]])
        image[pi.x, pi.y] = y_co_cg_to_rgb(colorYCoCg)

    # Save the image using PIL
    output_image = Image.fromarray(image)
    output_image.save(tex_out_file_name)

    t1 = time.time()
    elapsed_seconds = t1 - t0
    print(f"Time: {elapsed_seconds:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
