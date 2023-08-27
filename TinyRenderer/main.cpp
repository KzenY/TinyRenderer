#include <vector>
#include <iostream>
#include <fstream>
#include "tgaimage.h"
#include "model.h"
#include <Eigen/Dense>
#include "gl.h"

Model* model = NULL;
const int width = 800;
const int height = 800;
const float depth = 2000.f;
float* shadowbuffer = NULL;
Eigen::Matrix4f M;
//法向量变换矩阵
Eigen::Matrix4f M_IT;
Eigen::Matrix4f M_Shadow;
Vec3f light_dir(-1, 1, 1);
Vec3f       eye(0, 1, 3);
Vec3f    center(0, 0, 0);
Vec3f        up(0, 1, 0);

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

struct DepthShader :public Shader { 
	Eigen::Matrix3f tri;

	DepthShader() : tri() {}

	virtual Vec4f vertex(int iface, int nthvert) {
		Vec4f gl_Vertex = (model->vert(iface, nthvert)).homogeneous();
		gl_Vertex = M_Viewport * M_Projection * M_View * gl_Vertex;
		tri.col(nthvert) = (gl_Vertex / gl_Vertex[3]).head<3>();
		return gl_Vertex;
	}
	virtual bool fragment(Vec3f bar, TGAColor& color) {
		Vec3f p = tri * bar;
		color = TGAColor(255, 255, 255) * (2.f * p[2] / depth);
		return false;
	}

};

struct PhongShader :public Shader {
	Eigen::Matrix<float, 2, 3> varying_uv;
	Eigen::Matrix3f tri;
	Eigen::Matrix3f ndc_tri;
	Eigen::Matrix3f varying_norm;
	float ambient = 0.2;
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
		//ndc顶点
		gl_Vertex = M_Viewport * M_Projection * M_View * gl_Vertex;
		ndc_tri.col(nthvert) = (gl_Vertex/gl_Vertex[3]).head<3>();
		return  gl_Vertex;
	}

	virtual bool fragment(Vec3f bar, TGAColor& color) {
		//framebuffer screen变换到shadowbuffer screen
		Vec4f p = M_Shadow * ((ndc_tri * bar).homogeneous());
		//齐次处理
		p =p / p[3];
		//ndc
		int idx = int(p[0]) + int(p[1]) * width;
		float shadow=.3 + .7 * (shadowbuffer[idx] < p[2]+72.27);


		Vec2f uv = varying_uv * bar;
		//插值顶点法向
		Vec3f bn = (varying_norm * bar).normalized();

		//构建TBN矩阵
		Vec3f P1 = tri.col(1) - tri.col(0);
		Vec3f P2 = tri.col(2) - tri.col(0);

		Vec3f uv1 = Vec3f(varying_uv(0, 1) - varying_uv(0, 0), varying_uv(1, 1) - varying_uv(1, 0), 0);
		Vec3f uv2 = Vec3f(varying_uv(0, 2) - varying_uv(0, 0), varying_uv(1, 2) - varying_uv(1, 0), 0);

		Vec3f T = (P1 * uv2[1] - P2 * uv1[1]) / (uv1[0] * uv2[1] - uv2[0] * uv1[1]);
		Vec3f B = (P2 * uv1[0] - P1 * uv2[0]) / (uv1[0] * uv2[1] - uv2[0] * uv1[1]);
		Vec3f t_ = T - (T.dot(bn)) * bn;
		t_.normalize();
		Vec3f b_ = t_.cross(bn);
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
		float glow = model->glow(uv);
		float intensity = ambient + diff + spec+1.2f * glow;
		TGAColor c = model->diffuse(uv);
		for (int i = 0; i < 3; i++) color[i] = std::min<float>(c[i] * shadow * intensity, 255);
		return false;
	}
};


int main(int argc, char** argv) {
	if (2 == argc) {
		model = new Model(argv[1]);
	}
	else {
		model = new Model("obj/diablo3_pose/diablo3_pose.obj");
	}

	float* zbuffer = new float[width * height];
	shadowbuffer = new float[width * height];

	for (int i = width * height; --i; ) {
		zbuffer[i] = shadowbuffer[i] = -std::numeric_limits<float>::max();
	}

	light_dir.normalize();

	//rendering the shadow buffer
	{
		TGAImage depth(width, height, TGAImage::RGB);
		//对比frame buffer的lookat矩阵，相机位置替代光源位置
		lookat(light_dir, center, up);
		viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
		projection(0.f);
		DepthShader depthshader;
		Vec4f screen_coords[3];
		for (int i = 0; i < model->nfaces(); i++) {
			for (int j = 0; j < 3; j++) {
				screen_coords[j] = depthshader.vertex(i, j);
			}
			triangle(screen_coords, depthshader, depth, shadowbuffer);
		}
		depth.flip_vertically();
		depth.write_tga_file("depth.tga");
	}

	Eigen::Matrix4f MVP_shadow = M_Viewport * M_Projection * M_View;

	//rendering the frame buffer
	{
		lookat(eye, center, up);
		viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
		projection(-1.f/(eye-center).norm());
		TGAImage image(width, height, TGAImage::RGB);
		M = M_View;
		M_IT = (M_Projection * M_View).inverse().transpose();
		//framebuffer->object coordinates->shadowbuffer
		M_Shadow = MVP_shadow * ((M_Viewport * M_Projection * M_View).inverse());
		PhongShader shader;
		Vec4f screen_coords[3];
		for (int i = 0; i < model->nfaces(); i++) {
			for (int j = 0; j < 3; j++) {
				screen_coords[j] = shader.vertex(i, j);
			}  
			triangle(screen_coords, shader, image, zbuffer);
		}

		image.flip_vertically();
		image.write_tga_file("diablo_shadow_glow.tga");
	}

	delete model;
	delete[] zbuffer;
	delete[] shadowbuffer;
	return 0;
}