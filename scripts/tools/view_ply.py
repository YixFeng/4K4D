import argparse
from pathlib import Path

import open3d as o3d


KEY_LEFT = 263
KEY_RIGHT = 262
KEY_A = ord("A")
KEY_D = ord("D")


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


class PlyBrowser:
    def __init__(self, files, start_index=0, reset_view=False, point_size=2.0):
        self.files = files
        self.index = start_index
        self.reset_view = reset_view
        self.point_size = point_size
        self.geometry = None

    @property
    def current_file(self):
        return self.files[self.index]

    @property
    def current_id(self):
        return self.current_file.stem

    def window_title(self):
        return (
            f"PLY id={self.current_id} "
            f"({self.index + 1}/{len(self.files)}) "
            f"{self.current_file.name}"
        )

    def show_current(self, vis, reset_view=False):
        view = None
        if self.geometry is not None and not (reset_view or self.reset_view):
            view = vis.get_view_control().convert_to_pinhole_camera_parameters()

        geometry = load_ply(self.current_file)
        vis.clear_geometries()
        vis.add_geometry(geometry, reset_bounding_box=reset_view or self.reset_view)
        self.geometry = geometry

        if view is not None:
            vis.get_view_control().convert_from_pinhole_camera_parameters(
                view, allow_arbitrary=True
            )

        title = self.window_title()
        vis.get_render_option().point_size = self.point_size
        vis.update_renderer()
        try:
            vis.set_window_name(title)
        except (AttributeError, RuntimeError):
            pass
        print(title)
        print(self.current_file)

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
    vis.create_window(window_name=browser.window_title())
    vis.get_render_option().point_size = args.point_size

    vis.register_key_callback(KEY_LEFT, lambda v: browser.step(v, -1))
    vis.register_key_callback(KEY_RIGHT, lambda v: browser.step(v, 1))
    vis.register_key_callback(KEY_A, lambda v: browser.step(v, -1))
    vis.register_key_callback(KEY_D, lambda v: browser.step(v, 1))

    print("Controls: Left/A = previous, Right/D = next")
    browser.show_current(vis, reset_view=True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
