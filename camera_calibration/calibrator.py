from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class CheckerboardSpec:
    """棋盘格规格定义。

    Attributes:
        cols: 棋盘格内角点列数。
        rows: 棋盘格内角点行数。
        square_size: 单个方格物理尺寸（如 mm）。
    """

    cols: int
    rows: int
    square_size: float


@dataclass(slots=True)
class CalibrationResult:
    """标定结果容器。"""

    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    rms: float


class CameraCalibrator:
    """纯 NumPy 相机标定组件。

    流程：
    1) 从灰度图提取棋盘格角点。
    2) 用 Zhang 线性法估计内参初值。
    3) 进行非线性优化，最小化重投影误差。
    """

    def __init__(self, board: CheckerboardSpec) -> None:
        self.board = board
        # 固定世界坐标系下的棋盘格 3D 点（Z=0 平面）
        self._obj_points_3d = self._build_object_points_3d()

    def extract_checkerboard_corners(self, gray_image: np.ndarray) -> np.ndarray:
        """从单张灰度图提取有序棋盘格内角点。"""
        img = np.asarray(gray_image, dtype=np.float64)
        harris = self._harris_response(img)
        candidates = self._nms_points(harris, k=max(200, self.board.cols * self.board.rows * 4))
        return self._fit_grid_from_candidates(candidates)

    def calibrate_from_images(self, gray_images: Sequence[np.ndarray]) -> CalibrationResult:
        """输入多张灰度图，自动提角点并完成标定。"""
        corners = [self.extract_checkerboard_corners(img) for img in gray_images]
        return self.calibrate_from_corners(corners)

    def calibrate_from_corners(self, corners_per_image: Sequence[np.ndarray]) -> CalibrationResult:
        """输入多视图角点，执行标定与优化。"""
        # Step1: 用 2D-2D 对应估计各视图单应矩阵
        homographies = [self._estimate_homography(self._obj_points_3d[:, :2], np.asarray(c, float)) for c in corners_per_image]

        # Step2: Zhang 线性法求内参初值 K
        k = self._estimate_intrinsics_from_homographies(homographies)

        # Step3: 分解单应矩阵得到外参初值
        rvecs, tvecs = self._init_extrinsics(k, homographies)
        dist = np.zeros(2, dtype=float)  # 初始畸变参数 (k1, k2)

        # Step4: 非线性优化，最小化重投影误差
        k, dist, rvecs, tvecs = self._refine_parameters(k, dist, rvecs, tvecs, corners_per_image)
        rms = self._compute_rms(k, dist, rvecs, tvecs, corners_per_image)
        return CalibrationResult(k, dist, rvecs, tvecs, rms)

    def save(self, result: CalibrationResult, path: str | Path) -> None:
        """保存标定结果到 npz 文件。"""
        np.savez(path, camera_matrix=result.camera_matrix, distortion_coeffs=result.distortion_coeffs, rms=result.rms)

    def _build_object_points_3d(self) -> np.ndarray:
        """生成棋盘格世界坐标点 (N, 3)，其中 Z=0。"""
        xy = np.mgrid[0 : self.board.cols, 0 : self.board.rows].T.reshape(-1, 2) * self.board.square_size
        return np.hstack([xy, np.zeros((xy.shape[0], 1))]).astype(float)

    def _refine_parameters(self, k, dist, rvecs, tvecs, corners_per_image, iters: int = 15):
        """LM 风格非线性优化（数值雅可比）。"""
        p = self._pack_params(k, dist, rvecs, tvecs)
        lam = 1e-3
        for _ in range(iters):
            res = self._residual_vector(p, corners_per_image)
            j = self._numeric_jacobian(p, corners_per_image, res)
            h = j.T @ j + lam * np.eye(j.shape[1])
            g = j.T @ res
            try:
                dp = np.linalg.solve(h, -g)
            except np.linalg.LinAlgError:
                break

            p_new = p + dp
            # 若误差下降则接受并减小阻尼；否则增大阻尼
            if np.linalg.norm(self._residual_vector(p_new, corners_per_image)) < np.linalg.norm(res):
                p = p_new
                lam *= 0.5
            else:
                lam *= 2.0

        return self._unpack_params(p, len(corners_per_image))

    def _pack_params(self, k, dist, rvecs, tvecs):
        """将优化变量打平为一维向量。"""
        base = np.array([k[0, 0], k[1, 1], k[0, 1], k[0, 2], k[1, 2], dist[0], dist[1]], dtype=float)
        ext = []
        for r, t in zip(rvecs, tvecs):
            ext.extend(r.ravel())
            ext.extend(t.ravel())
        return np.concatenate([base, np.array(ext, dtype=float)])

    def _unpack_params(self, p, n_views):
        """将一维向量还原为 K / 畸变 / 外参。"""
        fx, fy, skew, cx, cy, k1, k2 = p[:7]
        k = np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        dist = np.array([k1, k2], dtype=float)
        rvecs, tvecs = [], []
        idx = 7
        for _ in range(n_views):
            rvecs.append(p[idx : idx + 3].copy())
            tvecs.append(p[idx + 3 : idx + 6].copy())
            idx += 6
        return k, dist, rvecs, tvecs

    def _residual_vector(self, p, corners_per_image):
        """构造残差向量：所有点的 (u,v) 重投影误差拼接。"""
        k, dist, rvecs, tvecs = self._unpack_params(p, len(corners_per_image))
        residuals = []
        for corners, rv, tv in zip(corners_per_image, rvecs, tvecs):
            proj = self._project_points(self._obj_points_3d, k, dist, rv, tv)
            residuals.append((proj - np.asarray(corners, float)).ravel())
        return np.concatenate(residuals)

    def _numeric_jacobian(self, p, corners_per_image, res0):
        """有限差分数值雅可比。"""
        eps = 1e-6
        j = np.zeros((res0.size, p.size), dtype=float)
        for i in range(p.size):
            pp = p.copy()
            pp[i] += eps
            j[:, i] = (self._residual_vector(pp, corners_per_image) - res0) / eps
        return j

    def _compute_rms(self, k, dist, rvecs, tvecs, corners_per_image):
        """计算整体重投影 RMS。"""
        errs = []
        for corners, rv, tv in zip(corners_per_image, rvecs, tvecs):
            proj = self._project_points(self._obj_points_3d, k, dist, rv, tv)
            e = proj - np.asarray(corners, float)
            errs.append(np.sum(e * e, axis=1))
        return float(np.sqrt(np.mean(np.concatenate(errs))))

    def _project_points(self, points3d, k, dist, rvec, tvec):
        """3D 点投影到像素平面（含 k1/k2 径向畸变）。"""
        r = self._rodrigues_to_matrix(rvec)
        pc = (r @ points3d.T).T + tvec.reshape(1, 3)
        x = pc[:, 0] / pc[:, 2]
        y = pc[:, 1] / pc[:, 2]
        r2 = x * x + y * y
        radial = 1.0 + dist[0] * r2 + dist[1] * r2 * r2
        xd, yd = x * radial, y * radial
        u = k[0, 0] * xd + k[0, 1] * yd + k[0, 2]
        v = k[1, 1] * yd + k[1, 2]
        return np.column_stack([u, v])

    def _init_extrinsics(self, k, homographies):
        """从单应矩阵初始化每个视图外参。"""
        invk = np.linalg.inv(k)
        rvecs, tvecs = [], []
        for h in homographies:
            h1, h2, h3 = h[:, 0], h[:, 1], h[:, 2]
            lam = 1.0 / np.linalg.norm(invk @ h1)
            r1 = lam * (invk @ h1)
            r2 = lam * (invk @ h2)
            r3 = np.cross(r1, r2)
            r = np.column_stack([r1, r2, r3])
            # 用 SVD 投影到最近旋转矩阵，避免数值偏差
            u, _, vt = np.linalg.svd(r)
            r = u @ vt
            t = lam * (invk @ h3)
            rvecs.append(self._matrix_to_rodrigues(r))
            tvecs.append(t)
        return rvecs, tvecs

    @staticmethod
    def _rodrigues_to_matrix(rvec):
        """Rodrigues 向量 -> 旋转矩阵。"""
        theta = np.linalg.norm(rvec)
        if theta < 1e-12:
            return np.eye(3)
        k = rvec / theta
        kx, ky, kz = k
        k_mat = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        return np.eye(3) + np.sin(theta) * k_mat + (1 - np.cos(theta)) * (k_mat @ k_mat)

    @staticmethod
    def _matrix_to_rodrigues(r):
        """旋转矩阵 -> Rodrigues 向量。"""
        trace = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(trace)
        if theta < 1e-12:
            return np.zeros(3)
        rx = (r[2, 1] - r[1, 2]) / (2 * np.sin(theta))
        ry = (r[0, 2] - r[2, 0]) / (2 * np.sin(theta))
        rz = (r[1, 0] - r[0, 1]) / (2 * np.sin(theta))
        return theta * np.array([rx, ry, rz])

    def _harris_response(self, img: np.ndarray, kappa: float = 0.04) -> np.ndarray:
        """Harris 响应图。"""
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
        gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
        ixx, iyy, ixy = gx * gx, gy * gy, gx * gy
        sxx = self._box_filter(ixx, 5)
        syy = self._box_filter(iyy, 5)
        sxy = self._box_filter(ixy, 5)
        return (sxx * syy - sxy * sxy) - kappa * (sxx + syy) ** 2

    @staticmethod
    def _box_filter(img: np.ndarray, k: int) -> np.ndarray:
        """积分图实现的均值盒滤波。"""
        p = np.pad(img, ((k // 2, k // 2), (k // 2, k // 2)), mode="reflect")
        ii = p.cumsum(0).cumsum(1)
        return (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) / (k * k)

    @staticmethod
    def _nms_points(response: np.ndarray, k: int) -> np.ndarray:
        """非极大值抑制，返回稀疏角点候选。"""
        ys, xs = np.where(response >= np.percentile(response, 98))
        ord_ = np.argsort(response[ys, xs])[::-1]
        pts = []
        for y, x in zip(ys[ord_], xs[ord_]):
            if all((y - py) ** 2 + (x - px) ** 2 >= 64 for py, px in pts):
                pts.append((y, x))
                if len(pts) >= k:
                    break
        return np.array([[x, y] for y, x in pts], float)

    def _fit_grid_from_candidates(self, candidates: np.ndarray) -> np.ndarray:
        """将候选角点拟合到规则 rows x cols 网格。"""
        if len(candidates) < self.board.rows * self.board.cols:
            raise RuntimeError("候选角点不足")
        c = candidates.mean(0)
        p = candidates - c
        _, _, vt = np.linalg.svd(p, full_matrices=False)
        uv = p @ vt[:2].T
        ci = self._quantize_axis(uv[:, 0], self.board.cols)
        ri = self._quantize_axis(uv[:, 1], self.board.rows)

        g = np.full((self.board.rows, self.board.cols, 2), np.nan)
        for pt, cc, rr in zip(candidates, ci, ri):
            if np.isnan(g[rr, cc, 0]):
                g[rr, cc] = pt
        if np.isnan(g).any():
            raise RuntimeError("网格拟合失败")
        return g.reshape(-1, 2)

    @staticmethod
    def _quantize_axis(v, bins):
        """将连续坐标量化到离散网格索引。"""
        n = (v - v.min()) / (v.max() - v.min() + 1e-12)
        return np.clip(np.round(n * (bins - 1)).astype(int), 0, bins - 1)

    @staticmethod
    def _estimate_homography(world_xy: np.ndarray, image_uv: np.ndarray) -> np.ndarray:
        """DLT 估计平面单应矩阵。"""
        n = world_xy.shape[0]
        a = np.zeros((2 * n, 9))
        for i, ((x, y), (u, v)) in enumerate(zip(world_xy, image_uv)):
            a[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
            a[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
        _, _, vt = np.linalg.svd(a)
        h = vt[-1].reshape(3, 3)
        return h / h[2, 2]

    @staticmethod
    def _v_ij(h: np.ndarray, i: int, j: int) -> np.ndarray:
        """Zhang 约束中的 v_ij 构造。"""
        return np.array(
            [
                h[0, i] * h[0, j],
                h[0, i] * h[1, j] + h[1, i] * h[0, j],
                h[1, i] * h[1, j],
                h[2, i] * h[0, j] + h[0, i] * h[2, j],
                h[2, i] * h[1, j] + h[1, i] * h[2, j],
                h[2, i] * h[2, j],
            ]
        )

    def _estimate_intrinsics_from_homographies(self, homographies):
        """根据多个单应矩阵线性求解内参矩阵 K。"""
        rows = []
        for h in homographies:
            ht = h.T
            rows += [self._v_ij(ht, 0, 1), self._v_ij(ht, 0, 0) - self._v_ij(ht, 1, 1)]

        _, _, vt = np.linalg.svd(np.vstack(rows))
        b11, b12, b22, b13, b23, b33 = vt[-1]
        v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12**2)
        lam = b33 - (b13**2 + v0 * (b12 * b13 - b11 * b23)) / b11
        alpha = np.sqrt(abs(lam / b11))
        beta = np.sqrt(abs(lam * b11 / (b11 * b22 - b12**2)))
        gamma = -b12 * alpha**2 * beta / lam
        u0 = gamma * v0 / beta - b13 * alpha**2 / lam
        return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]], dtype=float)
