import os
import matplotlib.pyplot as plt
import numpy as np
import math


def save(fig_id,fig_extension="png",resolution=300,directorio_root=".",tight_layout=False):
	#directorio_root
	imagenes_path = os.path.join(directorio_root, "imagenes")
	path = os.path.join(imagenes_path, fig_id + "." + fig_extension)
	if not os.path.isdir(imagenes_path):
		os.makedirs(imagenes_path)
	print("Saving figure", fig_id)
	if tight_layout:
		plt.tight_layout()
	n=0
	while(os.path.isfile(path)):
		n=n+1
		path=os.path.join(imagenes_path, fig_id +str(n)+ "." + fig_extension)

	plt.savefig(path, format=fig_extension, dpi=resolution)
