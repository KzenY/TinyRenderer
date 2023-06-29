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

Eigen::Matrix4f M_View=lookat(eye, center, up);
Eigen::Matrix4f M_Viewport=viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
Eigen::Matrix4f M_Projection=projection(-1.f / (eye - center).norm());


struct GouraudShader : public Shader {
    Vec3f varying_intensity; 

    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = (model->vert(iface, nthvert)).homogeneous(); 
        Eigen::Matrix4f M = M_Viewport * M_Projection * M_View;
        gl_Vertex = M * gl_Vertex;    
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert).dot(light_dir)); 
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color) {
        float intensity = varying_intensity.dot(bar);  
        color = TGAColor(255, 255, 255) * intensity; 
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

    GouraudShader shader;
    for (int i = 0; i < model->nfaces(); i++) {
        Vec4f screen_coords[3];
        for (int j = 0; j < 3; j++) {
            screen_coords[j] = shader.vertex(i, j);
        }
        triangle(screen_coords, shader, image, zbuffer);
    }

    image.flip_vertically(); 
    zbuffer.flip_vertically();
    image.write_tga_file("output.tga");
    zbuffer.write_tga_file("zbuffer.tga");

    delete model;
    return 0;
}


