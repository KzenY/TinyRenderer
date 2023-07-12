#include "tgaimage.h"
#include <Eigen/Dense>
#include "gl.h"

typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector2i Vec2i;

Shader::~Shader() {}

Eigen::Matrix4f viewport(int x, int y, int w, int h) {
	Eigen::Matrix4f M_Viewport;
	M_Viewport <<
		w / 2.f, 0, 0, x + w / 2.f,
		0, h / 2.f, 0, y + h / 2.f,
		0, 0, 255.f / 2.f, 255.f / 2.f,
		0, 0, 0, 1;
	return M_Viewport;
}

Eigen::Matrix4f lookat(Vec3f eye, Vec3f center, Vec3f up) {
	/*
   normalized（）是返回一个新向量，normalize是在原向量上修改无返回值
   Vec3f u = n.cross(up);
   u.normalize();
   */
	Vec3f n = (eye - center).normalized();
	Vec3f u = n.cross(up).normalized();
	Vec3f v = u.cross(n).normalized();
	Eigen::Matrix4f M_View;
	M_View <<
		u[0], u[1], u[2], -center[0],
		v[0], v[1], v[2], -center[1],
		n[0], n[1], n[2], -center[2],
		0, 0, 0, 1;
	return M_View;
}

Eigen::Matrix4f projection(float coeff) {
	Eigen::Matrix4f M_Projection = Eigen::Matrix4f::Identity();
	M_Projection(3, 2) = coeff;
	return M_Projection;
}

Vec3f barycentric(Vec2f* pts, Vec2i p) {
	Vec3f U = Vec3f(pts[2][0] - pts[0][0], pts[1][0] - pts[0][0], pts[0][0] - p[0]).cross(Vec3f(pts[2][1] - pts[0][1], pts[1][1] - pts[0][1], pts[0][1] - p[1]));
	if (std::abs(U[2]) > 1e-2) return Vec3f(1 - (U[0] + U[1]) / U[2], U[1] / U[2], U[0] / U[2]);
	return Vec3f(-1, 1, 1);
}

void triangle(Vec4f* pts, Shader& shader, TGAImage& image, TGAImage& zbuffer) {
	Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			bboxmin[j] = std::min(bboxmin[j], pts[i][j] / pts[i][3]);
			bboxmax[j] = std::max(bboxmax[j], pts[i][j] / pts[i][3]);
		}
	}
	Vec2i P;
	TGAColor color;
	for (P[0] = bboxmin[0]; P[0] <= bboxmax[0]; P[0]++) {
		for (P[1] = bboxmin[1]; P[1] <= bboxmax[1]; P[1]++) {
			Vec2f pts1[3];
			pts1[0] = (pts[0] / pts[0][3]).head<2>();
			pts1[1] = (pts[1] / pts[1][3]).head<2>();
			pts1[2] = (pts[2] / pts[2][3]).head<2>();
			Vec3f c = barycentric(pts1, P.head<2>());
			float z = pts[0][2] * c[0] + pts[1][2] * c[1] + pts[2][2] * c[2];
			float w = pts[0][3] * c[0] + pts[1][3] * c[1] + pts[2][3] * c[2];
			int frag_depth = std::max(0, std::min(255, int(z / w + 0.5)));
			if (c[0] < 0 || c[1] < 0 || c[2]<0 || zbuffer.get(P[0], P[1])[0]>frag_depth) continue;
			bool discard = shader.fragment(c, color);
			if (!discard) {
				zbuffer.set(P[0], P[1], TGAColor(frag_depth));
				image.set(P[0], P[1], color);
			}
		}
	}
}