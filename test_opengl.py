from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import cv2


class Renderer(object):
    def __init__(self, focal_length=600, img_w=512, img_h=512, faces=None,
                 same_mesh_color=False):
        self.focal_length = focal_length
        self.img_w = img_w
        self.img_h = img_h
        self.faces = faces
        self.same_mesh_color = same_mesh_color

        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.img_w, self.img_h)
        glutCreateWindow(b"Renderer")

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
        glClearColor(*bg_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-self.img_w/2.0, self.img_w/2.0, -self.img_h/2.0, self.img_h/2.0, 1, 1000)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)

        # Set lights
        light_diffuse = [1.0, 1.0, 1.0, 1.0]
        light_position = [1.0, 1.0, 1.0, 0.0]
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        # Set camera
        glPushMatrix()
        glTranslatef(0, 0, -self.focal_length)
        glRotatef(180, 1, 0, 0)
        glTranslatef(-self.img_w/2.0, -self.img_h/2.0, 0)

        # Render mesh
        for n, vert in enumerate(verts):
            glPushMatrix()
            if not self.same_mesh_color:
                color = colorsys.hsv_to_rgb(n / len(verts), 0.5, 1.0)
                glColor(*color)
            glBegin(GL_TRIANGLES)
            for face in self.faces:
                for vertex in face:
                    glVertex(*vert[vertex])
            glEnd()
            glPopMatrix()

        glPopMatrix()
        glFlush()

        # Get rendered image
        pixels = glReadPixels(0, 0, self.img_w, self.img_h, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.img_h, self.img_w, 3)
        image = np.flipud(image)

        # Update background image
        if bg_img_rgb is not None:
            mask = np.all(image == bg_color[:3], axis=-1)
            image[mask] = bg_img_rgb[mask]

        return image

    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def delete(self):
        pass

# Example Usage
if __name__ == "__main__":
    focal_length = 600
    img_w = 512
    img_h = 512
    faces = [[0, 1, 2], [0, 2, 3]]  # Example faces list
    same_mesh_color = False  # Example same_mesh_color flag

    renderer = Renderer(focal_length, img_w, img_h, faces, same_mesh_color)

    # Example input vertices
    verts = np.array([[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]], dtype=np.float32)

    # Example background image
    bg_img_rgb = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # Render front and side views
    front_view = renderer.render_front_view(verts, bg_img_rgb)
    side_view = renderer.render_side_view(verts)

    # Display the rendered views
    cv2.imshow("Front View", front_view)
    cv2.imshow("Side View", side_view)
    cv2.waitKey(0)

    renderer.delete()


sudo apt-get update
sudo apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev

pip install PyOpenGL PyOpenGL_accelerate

sudo apt-get install python3
sudo apt-get install python3-pip