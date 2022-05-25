#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:29:23 2020

@author: iesteban
"""

import trimesh
import scipy
import numpy as np
import matplotlib.cm as cm
import nibabel as nib
import gdist
import sys

import array2PLY
import io_AI

# import meshlabPY

# import nifti2colormesh

EMBRYO = "20190401_E2"
# CH     = 'DAPI'
CH = "mGFP"
tissue = "myo"

print("doing", EMBRYO)

PATH = "/media/iesteban/iesteban_HDD/Dropbox/ejemplo_Daniela_pintar_volumen"

mesh = trimesh.load_mesh(PATH + "/mesh_entrada2.ply", process=False)

final_number_faces = 18000
mesh = mesh.simplify_quadratic_decimation(int(final_number_faces))

d = 40


"""

READING SEGMENTATION FILE WITH CELLS-ID

"""
image_path = PATH + "/imagen_entrada_VOL.nii.gz"


img, dim_info = io_AI.load3D_basic(image_path)


# Sparse matrix
# row i, has all elements of the column(s) j at d<d
# matrix = geodesics.local_gdist_matrix(mesh, d)
print("      calculating geodesics...")
matrix = gdist.local_gdist_matrix(mesh.vertices, mesh.faces.astype(np.int32), d)
print("      END!")


# matrix_ij[1] --> the vertex from row i, that has N number of vertices j at d<d
# matrix_ij[0] --> the N vertices j, close to vertex i
#
#
#
#                           vertex 0                    vertex 1               vertex 2
#                   __________________________  ____________________________   ___________
#                  /                          \/                            \ /           \
#
# matrix_ij[1] --> 0  0  0  0 0   0   0   0  0  1   1   1   1   1   1   1  1  2 2  2  2  2        <--- vertices i
# matrix_ij[0] --> 12 24 13 2 811 645 865 40 65 777 104 353 505 676 841 77 44 6 12 54 68 654      <--- vertices j
#
#                  \__________________________/\____________________________/\___________/
#
#                         close to v1                close to v2              close to v3
#
#
#
matrix_ij = scipy.sparse.find(matrix)
matrix_i, matrix_j, distances = matrix_ij[1], matrix_ij[0], matrix_ij[2]
# I'm going to read the coordinates of every vertex j (aN convert them to pixels units)
j_coord = np.asarray(mesh.vertices[matrix_j])
j_pix = np.floor(
    j_coord / [dim_info["x_res"], dim_info["y_res"], dim_info["z_res"]]
).astype("uint16")
# nifti2colormesh.um2px(midS_mesh, dim_info)
# LEO LA INFORMACION EN LOS PIXELES DE LA IMAGEN
# Puedes ser cell_ID, cell_VOL... depende de img
j_VARIABLE = img[
    j_pix[:, 0], j_pix[:, 1], j_pix[:, 2]
]  # Y tambien me vale para leer el CSV, usando los cells id: VOLUMEN_CSV[j_cellID]


# matrix_ij[1] --> 0  0  0  0 0   0   0   0  0  1   1   1   1   1   1   1  1  2 2  2  2  2        <--- vertices i
# indices      --> 0  1  2  3 4   5   6   7  8  9   10  11  12  13  14  15 16 1718 19 20 21       <--- indices
#                  \__________________________/\_____________________________/\___________/
#                            de 0 a 9                    de 9 a 17              de 17 a 22
#
#                                                    |
#                                                   \|/
#
# tight_i      --> 0, 9, 17
#
# vertices_i solo tiene aquellos vertices, de todos los de mesh, que
# tienen la libreria que tienen 1 o mas vecines a d<d
tight_i = [np.where(matrix_i == i)[0][0] for i in np.unique(matrix_i)]


# Split j vertices according to the intervals dictated by tight_i
#
# tight_i      --> 0, 9, 17
#
# con np.split:
#                  [:0]      --> no dice nada (por eso lo de)
#                  [0:9]     --> vertice i=0 tiene cerca matrix_j[0:9]   ---> los guardo en la posicion 0 de -withme-
#                  [9:17]    --> vertice i=1 tiene cerca matrix_j[9:17]  ---> los guardo en la posicion 1 de -withme-
#                  [17:22]   --> vertice i=2 tiene cerca matrix_j[17:22] ---> los guardo en la posicion 2 de -withme-
#
# So, the vertex i, has at d<d the vertexes in withme_i[i], at distances included in thisfar[i]
withme_i_indices = np.split(matrix_j, tight_i)[1:]
thisfar_i = np.split(distances, tight_i)[1:]
# And... the coordinates of the vertexes close to i ar:
withme_i_coord = np.split(j_coord, tight_i)[1:]
withme_i_pix = np.split(j_pix, tight_i)[1:]
withme_i_VARIABLE = np.split(j_VARIABLE, tight_i)[1:]

# Area de todos los triangulos formados por los vertices d<d de i
withme_i_area_around = []
# So, the area of all the triangles that form the vertexes at d<d from i...
#     1.  All the faces to wich those vertices belong. Some faces will be made of
#         vertices that are not included in the original set
#
#            *     *           *
#                a__-- d\           the vertex i has very close vertex a, b, d, e
#         *     / \   /  \   *      they form a total of 3 faces
#              /   \ /    e         but, a, b, e and e also form faces with other vertices around(*)
#             b--___i__--'    *
#                            *
#        *       *
#
#     2. Cada vertice forma parte de muchas caras distintas, y por ello hay que
#        hacer np.unique() para tener las caras una vez en la lista
#
#     3. De todas estas caras, hay que quedarse con aquellas formadas por
#        3 vertices, contenidos en el grupo de vertices a d<d de i
#
#     4. Ahora que tengo las caras correctas, puedo obtener el area dentro del perimetro de i, a d<d
#
for i in range(mesh.vertices.shape[0]):
    # 1. Faces all the vertex close to i form
    all_faces = np.unique(mesh.vertex_faces[withme_i_indices[i]])
    # 2. Set of faces, only included once in the list, that form all the possible faces
    all_faces = (
        np.delete(all_faces, np.where(all_faces == -1)[0][0])
        if -1 in all_faces
        else all_faces
    )
    submesh_ppio = mesh.submesh([all_faces])[0]

    # 3. Determine which faces are formed completely by the vertex at d<d from i
    all_OK_faces = []
    for f in all_faces:  # para cada una de las caras...
        p1_as_set = set(list(mesh.faces[f]))  # ... estos son sus 3 vertices
        # de estos 3 vertices, ¿cuales estan presentes en la lista de vertices d<d de i, incluido el mismo vertice i?
        faces_in_common = list(p1_as_set.intersection(list(withme_i_indices[i]) + [i]))
        if len(faces_in_common) == 3:  # Si estan los 3, esta cara es valida
            all_OK_faces.append(f)
    # 4. Formo submesh con las faces correctas y calculo area
    submesh_OK = mesh.submesh([all_OK_faces])[0]
    withme_i_area_around.append(submesh_OK.area)


# Calculo el valor de cada indice
VARIABLE_per_i = np.asarray([np.mean(p) for p in withme_i_VARIABLE])  # VALOR MEDIO
# VARIABLE_per_i = np.asarray([len(np.unique(p))/a for a, p in zip(withme_i_area_around, withme_i_VARIABLE)]) # DENSIDAD: numero de celulas con distinta ID alrededor de p, a d<d


# En esta matrix voy a guardar los valores calculados para cada vertice
# Esta sera rellena en aquellas posiciones coincidentes con vertices_i, el resto tendra un color (0,0,0,0)... como un grillo!
all_colors = np.zeros(shape=(mesh.vertices.shape[0], 4), dtype="float64")

# Voy a convertir la variable calculada, sobre cada uno de los vertices indicados por vertice_i a RGBA
colores = array2PLY.colorines(VARIABLE_per_i, cm.jet)

# Voy a colocar los colores calculados para los vertices_i en el array total de color
all_colors[np.unique(matrix_i)] = colores

mesh.visual.vertex_colors = all_colors
mesh.export(PATH + "/outcome_d" + "_" + str(d) + ".ply")
