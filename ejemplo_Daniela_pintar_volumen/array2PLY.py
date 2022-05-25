#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:28:45 2019

@author: iesteban
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid
from plyfile import PlyData, PlyElement
from shutil import copyfile
from binaryornot.check import is_binary
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def ch2num(CHANNEL):
    if CHANNEL == "r":
        return 0
    if CHANNEL == "g":
        return 1
    if CHANNEL == "b":
        return 2


def index2rgb(value, minimum=0, maximum=255):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def header_verts_EDGES(len_verts, len_edges):
    phrases = []
    phrases.append("ply")
    phrases.append("format ascii 1.0")
    phrases.append("comment made by iesteban")
    phrases.append("element vertex " + str(len_verts))
    phrases.append("property float x")
    phrases.append("property float y")
    phrases.append("property float z")
    phrases.append("element edge " + str(len_edges))
    phrases.append("property int vertex1")
    phrases.append("property int vertex2")
    phrases.append("end_header")

    return phrases


def header_verts_EDGES_colores(len_verts, len_edges):
    phrases = []
    phrases.append("ply")
    phrases.append("format ascii 1.0")
    phrases.append("comment made by iesteban")
    phrases.append("element vertex " + str(len_verts))
    phrases.append("property float x")
    phrases.append("property float y")
    phrases.append("property float z")
    phrases.append("property uchar red")
    phrases.append("property uchar green")
    phrases.append("property uchar blue")
    phrases.append("property uchar alpha")
    phrases.append("element edge " + str(len_edges))
    phrases.append("property int vertex1")
    phrases.append("property int vertex2")
    phrases.append("end_header")

    return phrases


def header_XYZ(len_verts):
    phrases = []
    phrases.append("ply")
    phrases.append("format ascii 1.0")
    phrases.append("comment made by iesteban")
    phrases.append("element vertex " + str(len_verts))
    phrases.append("property float x")
    phrases.append("property float y")
    phrases.append("property float z")
    phrases.append("element face " + str(0))
    phrases.append("property list uchar int vertex_indices")
    phrases.append("end_header")

    return phrases


def header_XYZ_color(len_verts):
    phrases = []
    phrases.append("ply")
    phrases.append("format ascii 1.0")
    phrases.append("comment made by iesteban")
    phrases.append("element vertex " + str(len_verts))
    phrases.append("property float x")
    phrases.append("property float y")
    phrases.append("property float z")
    phrases.append("property uchar red")
    phrases.append("property uchar green")
    phrases.append("property uchar blue")
    phrases.append("property uchar alpha")
    phrases.append("element face " + str(0))
    phrases.append("property list uchar int vertex_indices")
    # phrases.append('property uchar alpha')
    phrases.append("end_header")

    return phrases


def header(len_verts, len_faces):
    phrases = []
    phrases.append("ply")
    phrases.append("format ascii 1.0")
    phrases.append("comment made by iesteban")
    phrases.append("element vertex " + str(len_verts))
    phrases.append("property float x")
    phrases.append("property float y")
    phrases.append("property float z")
    phrases.append("property uchar red")
    phrases.append("property uchar green")
    phrases.append("property uchar blue")
    phrases.append("element face " + str(len_faces))
    phrases.append("property list uchar int vertex_indices")
    phrases.append("end_header")

    return phrases


def header_no_color(len_verts, len_faces):
    phrases = []
    phrases.append("ply")
    phrases.append("format ascii 1.0")
    phrases.append("comment made by iesteban")
    phrases.append("element vertex " + str(len_verts))
    phrases.append("property float x")
    phrases.append("property float y")
    phrases.append("property float z")
    phrases.append("element face " + str(len_faces))
    phrases.append("property list uchar int vertex_indices")
    phrases.append("end_header")

    return phrases


def header_no_color_UV(len_verts, len_faces):
    phrases = []
    phrases.append("ply")
    phrases.append("format ascii 1.0")
    phrases.append("comment made by iesteban")
    phrases.append("element vertex " + str(len_verts))
    phrases.append("property float x")
    phrases.append("property float y")
    phrases.append("property float z")
    phrases.append("property float s")
    phrases.append("property float t")
    phrases.append("element face " + str(len_faces))
    phrases.append("property list uchar int vertex_indices")
    phrases.append("end_header")

    return phrases


def header_vertex_and_normals(len_verts):
    phrases = []
    phrases.append("ply")
    phrases.append("format ascii 1.0")
    phrases.append("comment made by iesteban")
    phrases.append("element vertex " + str(len_verts))
    phrases.append("property float x")
    phrases.append("property float y")
    phrases.append("property float z")
    phrases.append("property float nx")
    phrases.append("property float ny")
    phrases.append("property float nz")
    phrases.append("element face " + str(0))
    phrases.append("property list uchar int vertex_indices")
    phrases.append("end_header")

    return phrases


def header_PLY2(len_verts, len_faces):
    phrases = []
    phrases.append(str(len_verts))
    phrases.append(str(len_faces))

    return phrases


"""
Esta funcion toma un nube de puntos N,3 ordenados, que forman un contorno lineal

PATH.ply + verts + faces --> create & save .PLY2

"""


def verts_and_edges_2PLY(output_file, verts, edges, dc="."):
    # New ply file
    file1 = open(output_file, "w")
    # Escribir header
    phrases = header_verts_EDGES(len(verts), len(edges))
    [file1.write(phrase + "\n") for phrase in phrases]
    # Escribir vertices
    for vert in verts:
        [file1.write(str(coord).replace(".", dc) + " ") for coord in vert]
        file1.write("\n")
    # Escribir edges
    for edge in edges:
        [file1.write(str(f).replace(".", dc) + " ") for f in edge]
        file1.write("\n")
    file1.close()


"""
Esta funcion toma un nube de puntos N,3 ordenados, que forman un contorno lineal

PATH.ply + verts + faces --> create & save .PLY2

"""


def verts_and_edges_and_colors_2PLY(output_file, verts, edges, colors, dc="."):
    # New ply file
    file1 = open(output_file, "w")
    # Escribir header
    phrases = header_verts_EDGES_colores(len(verts), len(edges))
    [file1.write(phrase + "\n") for phrase in phrases]
    # Escribir vertices
    for vert, color in zip(verts, colors):
        [file1.write(str(coord).replace(".", dc) + " ") for coord in vert]
        [file1.write(str(c) + " ") for c in color]
        file1.write("\n")
    # Escribir edges
    for edge in edges:
        [file1.write(str(f).replace(".", dc) + " ") for f in edge]
        file1.write("\n")
    file1.close()


"""
Esta funcion toma un nube de puntos, en forma de array de N,3

PATH.ply + verts --> .PLY

"""


def verts_2PLY(output_file, verts, dc="."):
    # New ply file
    file1 = open(output_file, "w")
    # Escribir header
    phrases = header_XYZ(len(verts))
    [file1.write(phrase + "\n") for phrase in phrases]
    # Escribir vertices
    for vert in verts:
        [file1.write(str(coord).replace(".", dc) + " ") for coord in vert]
        file1.write("\n")
    file1.close()


"""
Esta funcion toma un nube de puntos, en forma de array de N,3

PATH.ply + verts + faces + colors --> .PLY + colors

Incorpora calor

"""


def verts_and_color_2PLY(output_file, verts, colors, dc="."):
    # New ply file
    file1 = open(output_file, "w")
    # Escribir header
    phrases = header_XYZ_color(len(verts))
    [file1.write(phrase + "\n") for phrase in phrases]
    # Escribir vertices
    for vert, color in zip(verts, colors):
        [file1.write(str(coord).replace(".", dc) + " ") for coord in vert]
        [file1.write(str(c) + " ") for c in color]
        file1.write("\n")

    file1.close()


"""
Esta funcion recoge el nombre del archivo .ply que se va a generar,
asi como los vertices y faces que este debe contener

PATH.ply + verts + faces --> create & save .PLY2

Incorpora colores en los vertices

"""


def verts_and_faces_and_colors_2PLY(output_file, verts, faces, colors, dc="."):
    # New ply file
    file1 = open(output_file, "w")
    # Escribir header
    phrases = header(len(verts), len(faces))
    [file1.write(phrase + "\n") for phrase in phrases]
    # Escribir vertices
    for vert, color in zip(verts, colors):
        [file1.write(str(coord).replace(".", dc) + " ") for coord in vert]
        [file1.write(str(c) + " ") for c in color]
        file1.write("\n")
    # Escribir faces
    for face in faces:
        file1.write("3 ")
        [file1.write(str(f).replace(".", dc) + " ") for f in face]
        file1.write("\n")
    file1.close()


"""
Esta funcion recoge el nombre del archivo .ply que se va a generar,
asi como los vertices y faces que este debe contener

PATH.ply + verts + faces --> create & save .PLY

NO INCORPORA COLORES

"""


def verts_and_faces2PLY(output_file, verts, faces, dc="."):
    # New ply file
    file1 = open(output_file, "w")
    # Escribir header
    phrases = header_no_color(len(verts), len(faces))
    [file1.write(phrase + "\n") for phrase in phrases]
    # Escribir vertices
    for vert in verts:
        [file1.write(str(coord).replace(".", dc) + " ") for coord in vert]
        file1.write("\n")
    # Escribir faces
    for face in faces:
        file1.write("3 ")
        [file1.write(str(f).replace(".", dc) + " ") for f in face]
        file1.write("\n")
    file1.close()


"""
Esta funcion recoge el nombre del archivo .ply que se va a generar,
asi como los vertices, UV y faces que este debe contener

PATH.ply + verts + UV + faces --> create & save .PLY

NO INCORPORA COLORES

"""


def verts_and_UV_faces2PLY(output_file, vertsUV, faces, dc="."):
    # New ply file
    file1 = open(output_file, "w")
    # Escribir header
    phrases = header_no_color_UV(len(vertsUV), len(faces))
    [file1.write(phrase + "\n") for phrase in phrases]
    # Escribir vertices
    for vertUV in vertsUV:
        [file1.write(str(coord).replace(".", dc) + " ") for coord in vertUV]
        file1.write("\n")
    # Escribir faces
    for face in faces:
        file1.write("3 ")
        [file1.write(str(f).replace(".", dc) + " ") for f in face]
        file1.write("\n")
    file1.close()


"""
Esta funcion recoge el nombre del archivo .ply que se va a generar,
asi como los vertices y los vectores normales que este debe contener

PATH.ply + verts + normals --> create & save .PLY

NO INCORPORA COLORES

"""


def verts_and_normals2PLY(output_file, verts, dc="."):
    # New ply file
    file1 = open(output_file, "w")
    # Escribir header
    phrases = header_vertex_and_normals(len(verts))
    [file1.write(phrase + "\n") for phrase in phrases]
    # Escribir vertices
    for vert in verts:
        [file1.write(str(coord).replace(".", dc) + " ") for coord in vert]
        file1.write("\n")
    file1.close()


"""
Esta funcion toma un path que apunta a un archivo .ply, 

"""


def plydata2ascii(ply_file):
    # Read using PlyData
    plydata = PlyData.read(ply_file)

    # Nada de tonterias, si el archivo ya es ASCII no hago nada...
    if not is_binary(ply_file):
        copyfile(ply_file, ply_file.replace(".ply", "_ASCII.ply"))
        return

    # Find the number of vertex and faces
    nverts = plydata.elements[0].count
    nfaces = plydata.elements[1].count

    # Create arrays and fill them with vertex and faces
    array4verts, array4faces = np.zeros(shape=(nverts, 3), dtype="float32"), np.zeros(
        shape=(nfaces, 3), dtype="int32"
    )
    for i in range(nverts):
        array4verts[i, :] = np.asarray(list(plydata.elements[0].data[i]))
    for i in range(nfaces):
        array4faces[i, :] = np.asarray(list(plydata.elements[1].data[i]))

    # Guardar <...>.ply ------> <...>_ASCII.ply
    verts_and_faces2PLY(
        ply_file.replace(".ply", "_ASCII.ply"), array4verts, array4faces
    )


"""
Esta funcion recoge el nombre del archivo .ply2 que se va a generar,
asi como los vertices y faces que este debe contener

PATH.ply2 + verts + faces --> create & save .PLY2

"""


def verts_and_faces2PLY2(output_file, verts, faces, dc="."):
    # New ply file
    file1 = open(output_file, "w")
    # Escribir header
    phrases = header_PLY2(len(verts), len(faces))
    [file1.write(phrase + "\n") for phrase in phrases]
    # Escribir vertices
    for vert in verts:
        [file1.write(str(coord).replace(".", dc) + " ") for coord in vert]
        file1.write("\n")
    # Escribir faces
    for face in faces:
        file1.write("3 ")
        [file1.write(str(f).replace(".", dc) + " ") for f in face]
        file1.write("\n")
    file1.close()


"""
Esta funcion coge un path que apunta a un archivo .ply, y recopila los arrays con
vertices y faces. Luego manda estos a una funcion para guardar el .ply2

.PLY --> .PLY2
"""


def ply2ply2(ply_file):
    # Read using PlyData
    plydata = PlyData.read(ply_file)

    # Find the number of vertex and faces
    nverts = plydata.elements[0].count
    nfaces = plydata.elements[1].count

    # Create arrays and fill them with vertex and faces
    array4verts, array4faces = np.zeros(shape=(nverts, 3), dtype="float32"), np.zeros(
        shape=(nfaces, 3), dtype="int32"
    )
    for i in range(nverts):
        array4verts[i, :] = np.asarray(list(plydata.elements[0].data[i]))[0:3]
    for i in range(nfaces):
        array4faces[i, :] = np.asarray(list(plydata.elements[1].data[i]))

    # Guardar <...>.ply ------> <...>_ASCII.ply
    verts_and_faces2PLY2(ply_file.replace(".ply", ".ply2"), array4verts, array4faces)


"""
Esta funcion coge un path que apunta a un archivo .ply2, lo lee, extrae vertices y faces
y genera un header para ply e incorpora estos vertices y faces

PLY2 --> PLY
"""


def ply22ply(ply2_file):
    # Open the ply2 fil
    ply2_array = open(ply2_file, "r").readlines()
    # Reading vertex and faces
    nverts = int(ply2_array[0])
    nfaces = int(ply2_array[1])
    verts = ply2_array[2 : nverts + 2]
    faces = ply2_array[nverts + 2 :]
    verts_array, faces_array = np.zeros(shape=(nverts, 3), dtype="float32"), np.zeros(
        shape=(nfaces, 3), dtype="int32"
    )
    for i, v in enumerate(verts):
        verts_array[i, :] = np.asarray(v.split("\n")[0].split(" "), dtype="float32")
    for i, f in enumerate(faces):
        faces_array[i, :] = np.asarray(f.split("\n")[0].split(" ")[1:], dtype="int32")

    # verts_and_faces2PLY(ply2_file.replace('.ply2', '.ply'), verts_array, faces_array)
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts_array, faces=faces_array, process=False)
    mesh.export(ply2_file.replace(".ply2", ".ply"))


"""
Esta funcion lee un archivo .STL 

"""


def read_vertex_STL(path_STL):
    # I can only do this in ascii
    assert not is_binary(path_STL), "(array2PLY.py) " + path_STL + "is binary, so..."

    # Read text file
    read_file = open(path_STL).readlines()

    # Calculate the number of faces
    # 1st resto al total de lineas 2, que corresponden a la primera y la ultima que abre y cierran el archivo
    # 2nd divido por 7 porque cada face ocupa 7 lineas en el texto
    num_of_faces = (len(read_file) - 2) / 7.0
    # 3rd me aseguro de que al dividir por 7 obtengo un numero entero
    assert num_of_faces == int(
        num_of_faces
    ), " dividiendo por 7 vas mal... (array2PLY.py)"
    num_of_faces = int(num_of_faces)

    # Array para guardar los 3 vertices que forman cada face (triangulo)
    array = np.zeros(shape=(num_of_faces, 3, 3), dtype="float32")

    # Recorro el archivo
    for i in range(num_of_faces):
        # Index to read the lines of the files
        position = 1 + i * 7

        # Lines, removing
        line1 = read_file[position + 2].split("\n")[0].split("vertex")[-1].split(" ")
        line2 = read_file[position + 3].split("\n")[0].split("vertex")[-1].split(" ")
        line3 = read_file[position + 4].split("\n")[0].split("vertex")[-1].split(" ")

        [line1.remove("") for empty_space in range(line1.count(""))]
        [line2.remove("") for empty_space in range(line2.count(""))]
        [line3.remove("") for empty_space in range(line3.count(""))]

        array[i, :, :] = np.asarray([line1, line2, line3])

    return array


# https://stackoverflow.com/questions/14088375/how-can-i-convert-rgb-to-cmyk-and-vice-versa-in-python
def rgb_to_cmyk(r, g, b):
    rgb_scale = 255
    cmyk_scale = 100
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / float(rgb_scale)
    m = 1 - g / float(rgb_scale)
    y = 1 - b / float(rgb_scale)

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = c - min_cmy
    m = m - min_cmy
    y = y - min_cmy
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c * cmyk_scale, m * cmyk_scale, y * cmyk_scale, k * cmyk_scale


def cmyk_to_rgb(c, m, y, k, cmyk_scale=100, rgb_scale=255):
    rgb_scale = 255
    cmyk_scale = 100
    r = rgb_scale * (1.0 - c / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    g = rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    b = rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    return r, g, b


# This function colors mesh_A faces with colors calculated from the vertex values of mesh_B
def paintOver(mesh_A, mesh_B, name_source, min_max=None):
    aux_dict = {0: "X", 1: "Y", 2: "Z"}
    if type(mesh_B).__name__ == "Trimesh":
        array_values = mesh_B.vertices
    elif type(mesh_B).__name__ == "ndarray":
        array_values = mesh_B

    # Para cada coordenada
    for i in range(3):
        mn = mx = None
        # En caso de haber introducido valores externos minimos y maximos, para relativizar los colores
        if min_max != None:
            mn, mx = min_max[0][i], min_max[1][i]

        # Pinto y guardo
        color = colorines(array_values[:, i], mn=mn, mx=mx)
        mesh_A.visual.vertex_colors = color
        mesh_A.export(name_source.replace(".ply", "_" + aux_dict[i] + ".ply"))


"""
#https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b
"""

# https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
def rgbA(value, only_one_color="rgb", minimum=0, maximum=255):
    # if minimum != -1 or maximum != -1:
    # minimum, maximum = float(min(value)), float(max(value))
    zeros = np.zeros(shape=value.shape, dtype=value.dtype)
    if only_one_color == "rgb":
        ratio = 2 * (value - minimum) / (maximum - minimum)

        b = np.maximum(zeros, 255 * (1 - ratio))
        r = np.maximum(zeros, 255 * (ratio - 1))

        b = np.asarray(b, dtype="uint8")
        r = np.asarray(r, dtype="uint8")

        g = 255 - b - r

        rgb0 = np.array([r, g, b, zeros + 255]).transpose()

        return rgb0.astype("uint8")

    elif only_one_color == "r" or only_one_color == "g" or only_one_color == "b":
        # Ratio para saltar de color en color
        ratio = (value - minimum) / (maximum - minimum)

        # Primero todo zeros
        rgb0 = np.array([zeros, zeros, zeros, zeros]).transpose()

        # Calculo rango de colores [0 255]
        color = np.maximum(zeros, 255 * (ratio))

        # coloco color en la columda correspondiente
        rgb0[:, ch2num(only_one_color)] = color

        return rgb0.astype("uint8")

    elif only_one_color == "grey":
        # Ratio para saltar de color en color
        ratio = (value - minimum) / (maximum - minimum)

        # Primero todo zeros
        rgb0 = np.array([zeros, zeros, zeros, zeros]).transpose()

        # Calculo rango de colores [0 255]
        color = np.maximum(zeros, 255 * (ratio))

        # El gris se caracteriza por tener todos los niveles de color igual
        rgb0[:, 0], rgb0[:, 1], rgb0[:, 2] = color, color, color

        return rgb0.astype("uint8")


# https://stackoverflow.com/questions/15140072/how-to-map-number-to-color-using-matplotlibs-colormap
def colorines(lista, colormap=cm.jet, mn=None, mx=None):
    if type(colormap).__name__ == "LinearSegmentedColormap":
        if mn == None:
            norm = Normalize(vmin=min(lista), vmax=max(lista))
        else:
            norm = Normalize(vmin=mn, vmax=mx)
        colores = colormap(norm(lista))
    else:
        colores = rgbA(lista, colormap)

    return colores
