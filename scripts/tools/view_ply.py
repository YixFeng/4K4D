import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


WINDOW_FAILURE_HELP = """\
Open3D failed to create a visualization window.

This viewer needs an interactive OpenGL/GLFW window. If you are running over SSH,
inside a container, or on a headless machine, start it from a desktop session,
enable X11 forwarding/VirtualGL, or install an Open3D build with OSMesa support.
"""

KEY_LEFT = 263
KEY_RIGHT = 262
KEY_A = ord("A")
KEY_D = ord("D")
LABEL_CELL_PX = 4
LABEL_MARGIN_PX = 24


FONT = {
    "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "# #  ", "  #  ", "  #  ", "  #  ", "#####"],
    "2": [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
    "3": [" ### ", "#   #", "    #", " ### ", "    #", "#   #", " ### "],
    "4": ["#   #", "#   #", "#   #", "#####", "    #", "    #", "    #"],
    "5": ["#####", "#    ", "#    ", "#### ", "    #", "#   #", " ### "],
    "6": [" ### ", "#   #", "#    ", "#### ", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "#   #", " ### "],
    "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
    "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    "I": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"],
    "M": ["#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"],
    "R": ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
    "_": ["     ", "     ", "     ", "     ", "     ", "     ", "#####"],
    "-": ["     ", "     ", "     ", " ### ", "     ", "     ", "     "],
    "/": ["    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "],
    " ": ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Browse .ply files in a directory with Open3D."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="A .ply file or a directory containing .ply files.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Point size for point cloud rendering.",
    )
    parser.add_argument(
        "--reset_view",
        action="store_true",
        help="Reset camera every time a new .ply is loaded.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search .ply files recursively when path is a directory.",
    )
    return parser.parse_args()


def collect_ply_files(path: Path, recursive: bool):
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".ply":
            raise ValueError(f"Input file is not a .ply: {path}")
        directory = path.parent
        files = sorted(directory.rglob("*.ply") if recursive else directory.glob("*.ply"))
        start_index = files.index(path)
    elif path.is_dir():
        files = sorted(path.rglob("*.ply") if recursive else path.glob("*.ply"))
        start_index = 0
    else:
        raise FileNotFoundError(path)

    if not files:
        raise FileNotFoundError(f"No .ply files found under: {path}")
    return files, start_index


def load_ply(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh.has_triangles():
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.has_points():
        return pcd

    raise ValueError(f"Open3D could not load geometry from: {path}")


def create_label(text: str):
    text = text.upper()
    char_w = 5
    char_h = 7
    char_gap = 1
    points = []
    lines = []
    colors = []

    def add_square(x, y):
        idx = len(points)
        x0 = x * LABEL_CELL_PX
        y0 = y * LABEL_CELL_PX
        size = LABEL_CELL_PX * 0.8
        points.extend(
            [
                [x0, y0],
                [x0 + size, y0],
                [x0 + size, y0 + size],
                [x0, y0 + size],
            ]
        )
        lines.extend(
            [
                [idx, idx + 1],
                [idx + 1, idx + 2],
                [idx + 2, idx + 3],
                [idx + 3, idx],
            ]
        )
        colors.extend([[1.0, 0.05, 0.02]] * 4)

    cursor = 0
    for char in text:
        bitmap = FONT.get(char, FONT["-"])
        for row, line in enumerate(bitmap):
            for col, pixel in enumerate(line):
                if pixel != " ":
                    add_square(cursor + col, row)
        cursor += char_w + char_gap

    label = o3d.geometry.LineSet()
    points = np.asarray(points, dtype=np.float64)
    if len(points):
        label.points = o3d.utility.Vector3dVector(
            np.zeros((len(points), 3), dtype=np.float64)
        )
        label.lines = o3d.utility.Vector2iVector(lines)
        label.colors = o3d.utility.Vector3dVector(colors)
    width = max(float(points[:, 0].max()), 0.0) if len(points) else 0.0
    height = char_h * LABEL_CELL_PX
    return label, points, width, height


def label_points_to_world(pixel_points, label_width, label_height, geometry, view):
    params = view.convert_to_pinhole_camera_parameters()
    intrinsic = params.intrinsic
    width = max(int(intrinsic.width), 1)
    height = max(int(intrinsic.height), 1)
    k = intrinsic.intrinsic_matrix
    fx = max(float(k[0, 0]), 1e-6)
    fy = max(float(k[1, 1]), 1e-6)
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    bbox = geometry.get_axis_aligned_bounding_box()
    center = np.append(bbox.get_center(), 1.0)
    center_camera = params.extrinsic @ center
    depth = abs(float(center_camera[2]))
    depth = max(depth * 0.35, 0.1)

    offset = np.array(
        [
            LABEL_MARGIN_PX,
            max(LABEL_MARGIN_PX, height - LABEL_MARGIN_PX - label_height),
        ],
        dtype=np.float64,
    )
    uv = pixel_points + offset
    camera_points = np.column_stack(
        [
            (uv[:, 0] - cx) / fx * depth,
            (uv[:, 1] - cy) / fy * depth,
            np.full(len(uv), depth, dtype=np.float64),
            np.ones(len(uv), dtype=np.float64),
        ]
    )
    world_points = (np.linalg.inv(params.extrinsic) @ camera_points.T).T[:, :3]
    return world_points


class PlyBrowser:
    def __init__(self, files, start_index=0, reset_view=False, point_size=2.0):
        self.files = files
        self.index = start_index
        self.reset_view = reset_view
        self.point_size = point_size
        self.geometry = None
        self.label = None
        self.label_pixels = None
        self.label_width = 0.0
        self.label_height = 0.0

    @property
    def current_file(self):
        return self.files[self.index]

    @property
    def current_id(self):
        return self.current_file.stem

    def label_text(self):
        return f"ID {self.current_id} {self.index + 1}/{len(self.files)}"

    def show_current(self, vis, reset_view=False):
        view = None
        if self.geometry is not None and not (reset_view or self.reset_view):
            view = vis.get_view_control().convert_to_pinhole_camera_parameters()

        geometry = load_ply(self.current_file)
        label, label_pixels, label_width, label_height = create_label(self.label_text())
        vis.clear_geometries()
        vis.add_geometry(geometry, reset_bounding_box=reset_view or self.reset_view)
        vis.add_geometry(label, reset_bounding_box=False)
        self.geometry = geometry
        self.label = label
        self.label_pixels = label_pixels
        self.label_width = label_width
        self.label_height = label_height

        if view is not None:
            vis.get_view_control().convert_from_pinhole_camera_parameters(
                view, allow_arbitrary=True
            )

        vis.get_render_option().point_size = self.point_size
        self.update_label(vis)
        vis.update_renderer()
        print(self.label_text())
        print(self.current_file)

    def update_label(self, vis):
        if self.geometry is None or self.label is None or self.label_pixels is None:
            return False
        if len(self.label_pixels) == 0:
            return False
        points = label_points_to_world(
            self.label_pixels,
            self.label_width,
            self.label_height,
            self.geometry,
            vis.get_view_control(),
        )
        self.label.points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(self.label)
        return False

    def step(self, vis, delta):
        self.index = (self.index + delta) % len(self.files)
        self.show_current(vis)
        return False


def main():
    args = parse_args()
    files, start_index = collect_ply_files(args.path, args.recursive)
    browser = PlyBrowser(
        files,
        start_index,
        reset_view=args.reset_view,
        point_size=args.point_size,
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name="Open3D PLY Browser"):
        raise SystemExit(WINDOW_FAILURE_HELP)

    render_option = vis.get_render_option()
    if render_option is None:
        vis.destroy_window()
        raise SystemExit(WINDOW_FAILURE_HELP)
    render_option.point_size = args.point_size

    vis.register_key_callback(KEY_LEFT, lambda v: browser.step(v, -1))
    vis.register_key_callback(KEY_RIGHT, lambda v: browser.step(v, 1))
    vis.register_key_callback(KEY_A, lambda v: browser.step(v, -1))
    vis.register_key_callback(KEY_D, lambda v: browser.step(v, 1))
    vis.register_animation_callback(browser.update_label)

    print("Controls: Left/A = previous, Right/D = next")
    browser.show_current(vis, reset_view=True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
