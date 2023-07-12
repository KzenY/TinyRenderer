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
		return M_Viewport * M_Projection * M_View*gl_Vertex;
	}

	virtual bool fragment(Vec3f bar, TGAColor& color) {
		Vec2f uv = varying_uv * bar;
		M = M_Projection * M_View;
		//法向量变换矩阵
		M_IT = M_View.inverse().transpose();
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
	Eigen::Matrix4f M;
	Eigen::Matrix4f M_IT;
	float ambient = 0.1;
	float diff;
	float spec;

	virtual Vec4f vertex(int iface, int nthvert) {
		//顶点坐标齐次化
		Vec4f gl_Vertex = (model->vert(iface, nthvert)).homogeneous();
		varying_uv.col(nthvert) = model->uv(iface, nthvert);
		return M_Viewport * M_Projection * M_View * gl_Vertex;
	}

	virtual bool fragment(Vec3f bar, TGAColor& color) {
		Vec2f uv = varying_uv * bar;
		M = M_Projection * M_View;
		//法向量变换矩阵
		M_IT = M_View.inverse().transpose();
		//法向量处理
		Vec3f n = (M_IT * model->normal(uv).homogeneous()).head<3>();
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
		TGAColor c = model->diffuse(uv);
		for(int i =0;i<3;i++)
			color[i] = std::min<float>(c[i] * intensity, 255);
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
	light_dir.normalize();

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
	zbuffer.flip_vertically();
	image.write_tga_file("Phong.tga");
	zbuffer.write_tga_file("zbuffer.tga");

	delete model;
	return 0;
}