"""
    Sample Run:
    python plot/plot_depth_equivariance_point_cloud_image_feature_gif.py
"""
import os, sys
sys.path.append(os.getcwd())


import glob
from matplotlib import pyplot as plt

import numpy as np
np.set_printoptions   (precision= 4, suppress= True)
from lib.helpers.file_io import write_lines, imread
from plot.common_operations import open_gif_writer, convert_fig_to_ubyte_image, add_ubyte_image_to_gif_writer, close_gif_writer

# =================================================================================================
# Main starts here
# =================================================================================================
curr_folder      = os.getcwd()
folder           = "/home/abhinav/Desktop/deviant_latex/"
output_folder    = "depth_equivariance"
image_file       = os.path.join(folder, "images/depth_equivariance_with_projective.tex")
num_pts          = 101

gif_image_folder = os.path.join(folder, output_folder)
gif_path         = os.path.join(folder, "depth_equivariance.gif")
images_list      = sorted(glob.glob(gif_image_folder + "/*.png"))

pc_linspace      = np.linspace(1.9, 1.3, num_pts)
img_linspace     = np.linspace(2.0, 1.3, num_pts)

for cnt in range(1, num_pts):
    lines = "\\begin{tikzpicture}[scale=0.318, every node/.style={scale=0.53}, every edge/.style={scale=0.53}]\n"
    lines+= "\\tikzset{vertex/.style = {shape=circle, draw=black!70, line width=0.06em, minimum size=1.4em}}\n"
    lines+= "\\tikzset{edge/.style = {-{Triangle[angle=60:.06cm 1]},> = latex'}}\n\n"


    lines+= "% Top left point cloud\n"
    lines+= "\\node[inner sep=0pt, thick] (input) at (-12.3, 0.0) {\\includegraphics[trim={1.0cm 1.2cm 4.3cm 0.2cm}, clip, height=1.9cm]{images/000000.png}};\n"
    lines+= "\\draw [draw=projectionBorderShade, line width=0.05em]    (-16.6, -1.65) rectangle (-8, 1.65) node[]{};\n\n"

    lines+= "\\draw [draw=projectionBorderShade, line width=0.05em, fill=white]    (-16.6, -7.65) rectangle (-8, -4.35) node[]{};\n"
    lines+= "\\node[inner sep=0pt, thick] (input) at (-12.3,-6) {\includegraphics[trim={1.0cm 1.2cm 4.3cm 0.2cm}, clip, height=" + str(pc_linspace[cnt]) + "cm]{images/000000.png}};\n\n"

    lines+= "\\node [inner sep=1pt, scale= 1.2] at (-12.3, 2.1)  {Point Cloud};\n\n"

    lines+= "\\draw [-{Triangle[angle=60:.1cm 1]}, draw=proposedShade, line width=0.1em, shorten <=0.5pt, shorten >=0.5pt, >=stealth](-12.3, -1.65) node[]{}-- (-12.3,-4.35) node[pos=0.5, scale= 1.1, align= center]{~~~~~~~~~~~~~~~~~~~~~~~~~Depth Translation\\\\~~~~~~~~~};\n\n"


    lines+= "%================ Arrows connecting Point cloud to image =========================\n"
    lines+= "\\draw [-{Triangle[angle=60:.1cm 1]},draw=black, line width=0.05em, shorten <=0.5pt, shorten >=0.5pt, >=stealth](-8,0) node[]{}-- (-4.25,0) node[]{};\n"
    lines+= "\\node [inner sep=1pt, scale= 1.2] at (-6.3, 0.5)  {Projective};\n\n"

    lines+= "\\draw [-{Triangle[angle=60:.1cm 1]},draw=black, line width=0.05em, shorten <=0.5pt, shorten >=0.5pt, >=stealth](-8,-6) node[]{}-- (-4.25,-6) node[]{};\n"
    lines+= "\\node [inner sep=1pt, scale= 1.2] at (-6.3, -5.5)  {Projective};\n\n"


    lines+= "% Top left image\n"
    lines+= "% left bottom right top\n"
    lines+= "\\node[inner sep=0pt, thick] (input) at (0,0) {\includegraphics[trim={5cm 0 5cm 0}, clip, height=2.0cm]{images/000008.png}};\n"
    lines+= "\\draw [draw=projectionBorderShade, line width=0.05em]    (-4.25, -1.65) rectangle (4.25, 1.65) node[]{};\n\n"

    lines+= "% Bottom left image\n"
    lines+= "\\draw [draw=projectionBorderShade, line width=0.05em, fill=black!80]    (-4.25, -7.65) rectangle (4.25, -4.35) node[]{};\n"
    lines+= "\\node[inner sep=0pt] (input) at (0,-6.0) {\includegraphics[trim={5cm 0 5cm 0}, clip, height=" + str(img_linspace[cnt]) + "cm]{images/000008.png}};\n\n"

    lines+= "\\node [inner sep=1pt, scale= 1.2] at (-0, 2.1)  {Image};\n\n"

    lines+= "% Top right image\n"
    lines+= "\\draw [draw=projectionBorderShade, line width=0.05em]    (10.7, -1.7) rectangle (19.3, 1.7) node[]{};\n"
    lines+= "\\node[inner sep=0pt, thick] (input) at (15,0) {\includegraphics[trim={2.3cm 0 2.3cm 0}, clip, height=2.0cm]{images/filter_1.png}};\n\n"

    lines+= "% Bottom right image\n"
    lines+= "\\draw [draw=projectionBorderShade, line width=0.05em, fill=white]    (10.75, -7.65) rectangle (19.25, -4.35) node[]{};\n"
    lines+= "\\node[inner sep=0pt, thick] (input) at (15,-6) {\includegraphics[trim={2.2cm 0 2.2cm 0}, clip, height=" + str(img_linspace[cnt]) + "cm]{images/filter_1.png}};\n\n"

    lines+= "\\node [inner sep=1pt, scale= 1.2] at (15, 2.1)  {Features};\n\n"

    lines+= "% blue vertical arrows\n"
    lines+= "\\draw [-{Triangle[angle=60:.1cm 1]}, draw=proposedShade, line width=0.1em, shorten <=0.5pt, shorten >=0.5pt, >=stealth](0, -1.65) node[]{} -- (0,-4.35) node[pos=0.5, scale= 1.1, align= center]{~~~~~~~~~~~~~~~~~~~~~~~~~Depth Translation\\\\~~~~~~~~~$=\\transformationMath_\scaleNotation$};\n\n"

    lines+= "\\draw [-{Triangle[angle=60:.1cm 1]}, draw=proposedShade, line width=0.1em, shorten <=0.5pt, shorten >=0.5pt, >=stealth](15, -1.65) node[]{}-- (15,-4.35) node[pos=0.5, scale= 1.1, align= center]{~~~~~~~~~~~~~~~~~~~~~~~~~~Depth Translation\\\\~~~~~~~~~$=\\transformationMath_\scaleNotation$};\n\n"



    lines+= "% black arrows\n"
    lines+= "\\draw [-{Triangle[angle=60:.1cm 1]}, draw=black, line width=0.05em, shorten <=0.5pt, shorten >=0.5pt, >=stealth](4.25, -6) node[]{}-- (10.75, -6) node[pos=0.5, scale= 2.4, align= center]{};\n"
    lines+= "\\node [inner sep=1pt, scale= 2, align= center] at (7.5, -5.75)  {*~~~~~~~~~~~~~~};\n\n"

    lines+= "\\draw [-{Triangle[angle=60:.1cm 1]}, draw=black, line width=0.05em, shorten <=0.5pt, shorten >=0.5pt, >=stealth](4.25, 0) node[]{}-- (10.75, 0) node[pos=0.5, scale= 2.4, align= center]{};\n"
    lines+= "\\node [inner sep=1pt, scale= 2, align= center] at (7.5, 0.25)  {*~~~~~~~~~~~~~~};\n\n"


    lines+= "% Box around Conv filters\n"
    lines+= "\\draw [draw=projectionBorderShade, line width=0.05em, fill=white]    (5.8, 1.2) rectangle (8.2, -7.2) node[]{};\n"
    lines+= "%\node [inner sep=1pt, scale= 1.2, align= center] at (7.5, 1.8)  {\ses{} Convolution};\n\n"

    lines+= "% Conv filters\n"
    lines+= "\\node[inner sep=0pt, thick] (input) at (7,0) {\includegraphics[trim={0cm 0cm 0cm 22cm}, clip, height=1.25cm]{images/sesn_basis_sample.png}};\n"
    lines+= "\\node[inner sep=0pt, thick] (input) at (7,-6) {\includegraphics[trim={0cm 22cm 0cm 0cm}, clip, height=1.25cm]{images/sesn_basis_sample.png}};\n"

    lines+= "\\draw [-{Triangle[angle=60:.1cm 1]}, draw=black, densely dashed, line width=0.04em, shorten <=0.5pt, shorten >=0.5pt, >=stealth] (7, -5.1) node[]{}  -- (7, -0.9) node[pos=0.5, scale= 0.9, align= center]{~~~~~~~~~$\\transformationMath_{\scaleNotation^{-1}}$};\n"


    lines+= "\\end{tikzpicture}\n"


    write_lines(image_file, lines_with_return_character= lines)

    # Compile the latex
    os.chdir(folder)
    command = "pdflatex -quiet depth_equivariance.tex"
    os.system(command)

    # convert pdf to png
    command = "convert -density 600 -trim -alpha remove depth_equivariance.pdf " + output_folder + "/" + str(cnt).zfill(6) + ".png"
    os.system(command)

    os.chdir(curr_folder)


# Convert to gif
gif_writer = open_gif_writer(gif_path, duration= 50, loop= 0)
for i, img_path in enumerate(images_list):
    if i % 10 == 0:
        print(i)
    img = imread(img_path, rgb= True)
    fig = plt.figure(dpi= 300, figsize= (10,6))
    plt.imshow(img)
    plt.axis('off')
    add_ubyte_image_to_gif_writer(gif_writer, convert_fig_to_ubyte_image(fig))
    plt.close()
close_gif_writer(gif_writer)