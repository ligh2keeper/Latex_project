import subprocess
from PIL import Image
import numpy as np
import random
import os


def _create_tex(formula, packages):
    """
    Create tex from the formula and any optional packages.
    Return latex string
    """
    tex_elements = [r'\documentclass[preview]{standalone}', r'\usepackage{amsmath}']
    if packages:
        tex_elements += [r'\usepackage{' + package + '}' for package in packages]
    tex_elements += [r'\begin{document}', r'\begin{equation*}']
    tex_elements += [formula]
    tex_elements += [r'\end{equation*}', r'\end{document}']

    return "\n".join(tex_elements)

def _crop(inname, outname):

    image=Image.open('{}1.png'.format(inname)).convert('L')
    image.load()
    image_data = np.asarray(image)
    image_data = 255 - image_data

    non_empty_columns = np.where(image_data.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data.max(axis=1)>0)[0]

    if len(non_empty_rows) and len(non_empty_columns):
        cropbox = (min(non_empty_columns), min(non_empty_rows), max(non_empty_columns), max(non_empty_rows))
        image = image.crop(cropbox)

    filename = '{}.png'.format(outname)
    image.save(filename)
    return filename

def _remove_ignore_errors(filename):
    """
    Remove a file but ignore errors. We shouldn;t fail just because a temp file didn't get deleted.
    :param filename:
    :return: None
    """
    try:
        os.remove(filename)
    except Exception:
        pass


def rasterise_formula(name, formula, dpi=300, packages=None):

    unique_name = "{}-{}".format(name, random.randint(100000, 999999))
    tex = _create_tex(formula, packages)
    tex_fn = '{}.tex'.format(unique_name)
    with open(tex_fn, 'w') as tex_file:
        tex_file.write(tex)
    process = subprocess.Popen('latex -interaction=batchmode {}.tex'.format(unique_name), shell=True,
                               stdout=subprocess.PIPE)
    process.communicate()
    process.wait()

    process = subprocess.Popen('dvipng -T tight -D {} {}.dvi'.format(dpi, unique_name), shell=True,
                               stdout=subprocess.PIPE)
    process.communicate()
    process.wait()

    filename = _crop(unique_name, name)

    _remove_ignore_errors("{}.aux".format(unique_name))
    _remove_ignore_errors("{}.log".format(unique_name))
    _remove_ignore_errors("{}.tex".format(unique_name))
    _remove_ignore_errors("{}.dvi".format(unique_name))
    _remove_ignore_errors("{}1.png".format(unique_name))

    return filename
