# Stage_CHPS_2025
Mise en place d'un système Out-Of-Core pour le logiciel [SimLOD](https://github.com/m-schuetz/SimLOD)

Instruction de compilation/installation pour SIMLOD sur Juliet :

 	visu (pour lire tous les fichiers, prendre 6 cpus, 32go ram, 16go vram)
 	scl enable gcc-toolset-14 bash
	git clone https://github.com/m-schuetz/SimLOD.git
	ajout de #include <stdint.h> dans fichier modules/progressive_octree/SimlodLoader.h
	ajout de std:: devant appelle à format dans le fichier modules/progressive_octree/main_progressive_octree.cpp 
	mkdir out && cd out
	cmake .. && make
	export CUDA_PATH=/usr/local/cuda-12.1
	./SimLOD
yé, ça marche !

Instructions pour relancer SimLOD sur Juliet depuis une précédente compilation/installation :

	scl enable gcc-toolset-14 bash
	export CUDA_PATH=/usr/local/cuda-12.1
	./SimLOD
 ou

  	./start.sh
   	cd out && ./SimlLOD
yé, ça remarche !
