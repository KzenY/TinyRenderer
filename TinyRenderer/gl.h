#pragma once
#ifndef GL_H__
#define GL_H__

#include "tgaimage.h"
#include <Eigen/Dense>

typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector4f Vec4f;

Eigen::Matrix4f viewport(int x, int y, int w, int h);
Eigen::Matrix4f projection(float coeff = 0.f);
Eigen::Matrix4f lookat(Vec3f eye, Vec3f center, Vec3f up);

struct Shader {
	virtual ~Shader();
	virtual Vec4f vertex(int iface, int nthvert) = 0;
	virtual bool fragment(Vec3f bar, TGAColor& color) = 0;
};

void triangle(Vec4f* pts, Shader& shader, TGAImage& image, TGAImage& zbuffer);




















#endif