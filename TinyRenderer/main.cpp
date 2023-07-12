#include <vector>
#include <iostream>
#include "tgaimage.h"
#include "model.h"
#include <Eigen/Dense>
#include "gl.h"

Model* model = NULL;
const int width = 800;
const int height = 800;

Vec3f light_dir(1, -1, 1);
Vec3f       eye(0, 1, 3);
Vec3f    center(0, 0, 0);
Vec3f        up(0, 1, 0);

Eigen::Matrix4f M_View = lookat(eye, center, up);
Eigen::Matrix4f M_Viewport = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
Eigen::Matrix4f M_Projection = projection(-1.f / (eye - center).norm());

struct GouraudShader : public Shader {
	Eigen::Matrix<float, 2, 3> varying_uv;
	Eigen::Matrix4f M;
	Eigen::Matrix4f M_IT;

	virtual Vec4f vertex(int iface, int nthvert) {
		Vec4f gl_Vertex = (model->vert(iface, nthvert)).homogeneous();
		varying_uv.col(nthvert) = model->uv(iface, nthvert);
		return M_Viewport * M_Projection * M_View * gl_Vertex;
	}

	virtual bool fragment(Vec3f bar, TGAColor& color) {
		Vec2f uv = varying_uv * bar;
		M = M_Projection * M_View;
		//法向量变换矩阵
		M_IT = M.inverse().transpose();
		//法向量处理
		Vec3f n = (M_IT * model->normal(uv).homogeneous()).head<3>();
		n.normalize();
		//光线向量变换到世界空间
		Vec3f l = (M * light_dir.homogeneous()).head<3>();
		l.normalize();
		float intensity = std::max(0.f, n.dot(l));
		color = model->diffuse(uv) * intensity;
		return false;
	}
};

struct PhongShader :public Shader {
	Eigen::Matrix<float, 2, 3> varying_uv;
	Eigen::Matrix3f tri;
	Eigen::Matrix3f varying_norm;
	Eigen::Matrix4f M = M_Projection * M_View;
	//法向量变换矩阵
	Eigen::Matrix4f M_IT = M.inverse().transpose();
	float ambient = 0.1;
	float diff;
	float spec;

	virtual Vec4f vertex(int iface, int nthvert) {
		//顶点坐标齐次化
		Vec4f gl_Vertex = (model->vert(iface, nthvert)).homogeneous();
		//uv坐标
		varying_uv.col(nthvert) = model->uv(iface, nthvert);
		//世界空间顶点法线
		varying_norm.col(nthvert) = (M_IT * model->normal(iface, nthvert).homogeneous()).head<3>();
		//世界空间顶点
		tri.col(nthvert) = (M * gl_Vertex).head<3>();
		return  M_Viewport * M_Projection * M_View * gl_Vertex;
	}

	virtual bool fragment(Vec3f bar, TGAColor& color) {
		//插值
		Vec2f uv = varying_uv * bar;
		Vec3f bn = (varying_norm * bar).normalized();

		Vec3f P1 = tri.col(1) - tri.col(0);
		Vec3f P2 = tri.col(2) - tri.col(0);

		Vec3f uv1 = Vec3f(varying_uv(0, 1) - varying_uv(0, 0), varying_uv(1, 1) - varying_uv(1, 0), 0);
		Vec3f uv2 = Vec3f(varying_uv(0, 2) - varying_uv(0, 0), varying_uv(1, 2) - varying_uv(1, 0), 0);

		Vec3f T = (P1 * uv2[1] - P2 * uv1[1]) / (uv1[0] * uv2[1] - uv2[0] * uv1[1]);
		Vec3f B = (P2 * uv1[0] - P1 * uv2[0]) / (uv1[0] * uv2[1] - uv2[0] * uv1[1]);
		Vec3f t_ = T - (T.dot(bn)) * bn;
		t_.normalize();
		Vec3f b_ = B - B.dot(bn) * bn - B.dot(t_) * t_;
		b_.normalize();
		Eigen::Matrix3f M_tbn;
		M_tbn.col(0) = t_;
		M_tbn.col(1) = b_;
		M_tbn.col(2) = bn;

		//切线空间法线变换到世界空间
		Vec3f n = M_tbn * model->normal(uv);
		n.normalize();
		//入射光向量
		Vec3f l = (M * light_dir.homogeneous()).head<3>();
		l.normalize();
		diff = std::max(n.dot(l), 0.f);
		//视线向量
		Vec3f v = (M * (eye.homogeneous())).head<3>();
		v.normalize();
		//半程向量
		Vec3f h = (v + l).normalized();
		spec = pow(std::max(h.dot(n), 0.f), model->specular(uv));
		float intensity = ambient + diff + .6f * spec;
		color = model->diffuse(uv) * intensity;
		return false;
	}
};

int main(int argc, char** argv) {
	if (2 == argc) {
		model = new Model(argv[1]);
	}
	else {
		model = new Model("obj/african_head.obj");
	}

	TGAImage image(width, height, TGAImage::RGB);
	TGAImage zbuffer(width, height, TGAImage::GRAYSCALE);

	PhongShader shader;
	for (int i = 0; i < model->nfaces(); i++) {
		Vec4f screen_coords[3];
		for (int j = 0; j < 3; j++) {
			screen_coords[j] = shader.vertex(i, j);
		}
		triangle(screen_coords, shader, image, zbuffer);
	}

	image.flip_vertically();
	image.write_tga_file("Phong_7.tga");
	delete model;
	return 0;
}