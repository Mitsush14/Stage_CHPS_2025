// Some code in this file, particularly frustum, ray and intersection tests, 
// is adapted from three.js. Three.js is licensed under the MIT license
// This file this follows the three.js licensing
// License: MIT https://github.com/mrdoob/three.js/blob/dev/LICENSE

#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "math.cuh"
#include "structures.cuh"

#include "../CudaPrint/CudaPrint.cuh"

// Namespace appartenant à CUDA et facilitant la gestion des groupes de threads
namespace cg = cooperative_groups; 

// Constantes
constexpr uint64_t VOXEL_BACKLOG_CAPACITY = 10'000'000; // Capacité max pour les voxels en attente
constexpr float MAX_PROCESSING_TIME = 10.0f; // Temps max de traitement

// Variables globales

Allocator* allocator; // Allocation de mémoire (TODO : voir comment ça fonctionne)
uint32_t* errorValue; // Simple valeur d'erreur
AllocatorGlobal* allocator_persistent; // Allocation de mémoire persistante (TODO : voir comment ça fonctionne)
Stats* stats = nullptr; // Struct pour stocker des stats

Point* backlog_voxels      = nullptr; // Tableau de voxels en attente
Node** backlog_targets     = nullptr; // Tableau de noeuds cibles pour les voxels en attente
uint32_t* numBacklogVoxels = nullptr; // Nombre de voxels en attente
Chunk** chunkQueue         = nullptr; // File d'attente de chunks
CudaPrint* cudaprint       = nullptr; // Objet pour afficher des messages CUDA

// https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=8 (byte order inverted)
uint32_t SPECTRAL[8] = {
	0x4f3ed5,
	0x436df4,
	0x61aefd,
	0x8be0fe,
	0x98f5e6,
	0xa4ddab,
	0xa5c266,
	0xbd8832,
};

// Création d'un voxel pour un point donné du nuage de points
// Si voxel existe déjà, ignorer le point
// pX_full, pY_full, pZ_full : coordonnées du voxel à la profondeur maximale de l'octree
// - Create a voxel for <point> in <node>.
// - If a voxel at that cell exists, ignore point
// - <pX_full> are a point's voxel coordinates at maximum octree depth
void sampleVoxel(Node* node, 
	uint32_t pX_full, uint32_t pY_full, uint32_t pZ_full, 
	Point point, 
	float3 octreeMin, float3 octreeMax, float octreeSize
){

	if(node->grid == nullptr) return;
	// if(level != 0) return;

	// node coordinate in current level's voxel precision
	// level 0: [0, 128), 
	// level 1: [0, 256), ...
	// uint32_t nX = node->X * 128 * (1 << node->level);
	// uint32_t nY = node->Y * 128 * (1 << node->level);
	// uint32_t nZ = node->Z * 128 * (1 << node->level);

	// point coordinate in current level's voxel precision
	// from max precision of 2^24 = [0, 16'777'216)
	// for level 0, we need to go from 2^24 to 2^7
	// for level 1, we need to go from 2^24 to 2^8
	// dividing by 2^n is equal to 2^(24 - n)
	// so we need to divide by 2^(17-level)
	// uint32_t pX_leveled = pX_full / (1 << (17 - node->level));
	// uint32_t pY_leveled = pY_full / (1 << (17 - node->level));
	// uint32_t pZ_leveled = pZ_full / (1 << (17 - node->level));

	// reduce voxel coordinates from octree's max_depth to current node's depth,
	// by halving for each level in between
	uint32_t pX_leveled = pX_full / (1 << ((MAX_DEPTH + 1) - node->level));
	uint32_t pY_leveled = pY_full / (1 << ((MAX_DEPTH + 1) - node->level));
	uint32_t pZ_leveled = pZ_full / (1 << ((MAX_DEPTH + 1) - node->level));

	// now make voxel coordinates relative to current node instead of octree
	uint32_t pX = pX_leveled % GRID_SIZE;
	uint32_t pY = pY_leveled % GRID_SIZE;
	uint32_t pZ = pZ_leveled % GRID_SIZE;

	// check if point's bit in node's 1-bit voxel sampling grid is set
	uint32_t voxelIndex = pX + pY * GRID_SIZE + pZ * GRID_SIZE * GRID_SIZE;
	uint32_t voxelGridElementIndex = voxelIndex / 32;
	uint32_t voxelGridElementBitIndex = voxelIndex % 32;

	uint32_t bitmask = 1 << voxelGridElementBitIndex;
	uint32_t old_nonatomic = node->grid->values[voxelGridElementIndex];
	if((old_nonatomic & bitmask) != 0) return;

	uint32_t old = atomicOr(&node->grid->values[voxelGridElementIndex], bitmask);

	// if it is not set, we create a voxel from the point!
	if((old & bitmask) == 0){
		// first point in cell!
		atomicAdd(&node->numVoxels, 1);

		float nodeSize = octreeSize / pow(2.0f, float(node->level));

		// node-min
		float nodeMin_x = (float(node->X) + 0.0f) * nodeSize + octreeMin.x;
		float nodeMin_y = (float(node->Y) + 0.0f) * nodeSize + octreeMin.y;
		float nodeMin_z = (float(node->Z) + 0.0f) * nodeSize + octreeMin.z;

		// TODO: QUANTIZE
		Point voxel;
		voxel.x = nodeMin_x + nodeSize * (float(pX) + 0.5f) / float(GRID_SIZE);
		voxel.y = nodeMin_y + nodeSize * (float(pY) + 0.5f) / float(GRID_SIZE);
		voxel.z = nodeMin_z + nodeSize * (float(pZ) + 0.5f) / float(GRID_SIZE);
		voxel.color = point.color;

		uint32_t backlogIndex = atomicAdd(numBacklogVoxels, 1);
		backlog_voxels[backlogIndex] = voxel;
		backlog_targets[backlogIndex] = node;
	}
}

// TODO : vérifier comment ça fonctionne
// Permet de compter les points dans un octree
// Si nb points supérieurs à MAX_POINTS_PER_NODE, il est ajouté à une liste de noeuds à diviser
// Points en trop stockés dans spilledPoints pour être réinsérés après division des noeuds

bool doCounting(
	Node* root, Point* points, int numPoints, 
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Node* nodes, 
	Node** spillingNodes, uint32_t* numSpillingNodes,
	Point* spilledPoints, uint32_t* numSpilledPoints,
	uint32_t countIteration
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	auto t_00 = nanotime();

	// quantization grid for coordinates
	// we want octree node coordinates for a max depth of, e.g., 16
	float fGridSize = pow(2.0f, float(MAX_DEPTH));

	*numSpillingNodes = 0;

	grid.sync();

	auto countPoint = [&](Point point){

		// node coordinate at MAX_DEPTH
		uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize;
		uint32_t Y = fGridSize * (point.y - octreeMin.y) / octreeSize;
		uint32_t Z = fGridSize * (point.z - octreeMin.z) / octreeSize;

		// integer point coordinate relative to root node
		uint32_t pX = MAX_DEPTH_GRIDSIZE * (point.x - octreeMin.x) / octreeSize;
		uint32_t pY = MAX_DEPTH_GRIDSIZE * (point.y - octreeMin.y) / octreeSize;
		uint32_t pZ = MAX_DEPTH_GRIDSIZE * (point.z - octreeMin.z) / octreeSize;

		Node* current = root;

		// traverse to leaf node, compute some data about it
		int level = 0;
		uint32_t level_X;
		uint32_t level_Y;
		uint32_t level_Z;
		uint32_t child_X;
		uint32_t child_Y;
		uint32_t child_Z;
		uint32_t childIndex;

		for(; level < MAX_DEPTH; level++){

			level_X = X >> (MAX_DEPTH - level - 1);
			level_Y = Y >> (MAX_DEPTH - level - 1);
			level_Z = Z >> (MAX_DEPTH - level - 1);

			child_X = level_X & 1;
			child_Y = level_Y & 1;
			child_Z = level_Z & 1;

			childIndex = (child_X << 2) | (child_Y << 1) | child_Z;

			if(current->children[childIndex] == nullptr){
				// current == leaf!
				break;
			}else{
				current = current->children[childIndex];
			}
		}

		Node* leaf = current;

		// count points in leaf nodes
		if(leaf->countIteration < countIteration){

			// one atomicAdd per point
			// uint32_t old = atomicAdd(&leaf->numPoints, 1);
			// if(old == MAX_POINTS_PER_NODE){
			// 	// needs splitting
			// 	uint32_t spillIndex = atomicAdd(numSpillingNodes, 1);
			// 	spillingNodes[spillIndex] = leaf;
			// }

			// merge atomicAdds within warps to reduce contention
			uint64_t leafptr = uint64_t(leaf);
			auto warp = cg::coalesced_threads();
			auto group = cg::labeled_partition(warp, leafptr);

			uint32_t old = 0;
			if(group.thread_rank() == 0){
				old = atomicAdd(&leaf->counter, group.num_threads());

				if(old <= MAX_POINTS_PER_NODE)
				if(old + group.num_threads() > MAX_POINTS_PER_NODE)
				{
					// needs splitting
					uint32_t spillIndex = atomicAdd(numSpillingNodes, 1);
					spillingNodes[spillIndex] = leaf;
				}
			}
		}
	};

	auto t_10 = nanotime();

	// Count points of current batch
	processRange(numPoints, [&](int pointID){
		Point point = points[pointID];
		
		countPoint(point);
	});

	grid.sync();
	
	auto t_20 = nanotime();

	// count spilled points of previous iterations of current batch
	processRange(*numSpilledPoints, [&](int pointID){
		Point point = spilledPoints[pointID];
		
		countPoint(point);
	});

	grid.sync();
	auto t_30 = nanotime();

	// iterate through "numSpillingNodes" nodes.
	// borrow "numSpillingNodes" as a counter and temparily store its value in realNumSpillingNodes
	uint32_t realNumSpillingNodes = *numSpillingNodes;

	grid.sync();

	__shared__ int sh_nodeIndex;

	while(true){
		block.sync();

		if(block.thread_rank() == 0){
			sh_nodeIndex = atomicAdd(numSpillingNodes, 1) - realNumSpillingNodes;
		}

		block.sync();

		if(sh_nodeIndex >= realNumSpillingNodes) break;

		Node* node = spillingNodes[sh_nodeIndex];

		int numChunks = (node->numPoints + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;
		Chunk* chunk = node->points;
		int chunkIndex = 0;

		// iterate through all points in all chunks
		for(
			int pointIndex = block.thread_rank();
			pointIndex < node->numPoints;
			pointIndex += block.num_threads()
		){

			int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

			if(chunkIndex < targetChunkIndex){
				chunk = chunk->next;
				chunkIndex++;
			}

			int pIndex = pointIndex % POINTS_PER_CHUNK;
			Point point = chunk->points[pIndex];
			uint32_t spillID = atomicAdd(numSpilledPoints, 1);
			spilledPoints[spillID] = point;
		}
	}

	grid.sync();
	// now revert to original value
	*numSpillingNodes = realNumSpillingNodes;

	grid.sync();
	auto t_40 = nanotime();

	processRange(stats->numNodes, [&](int nodeIndex){
		nodes[nodeIndex].countIteration = countIteration;
	});

	grid.sync();
	auto t_50 = nanotime();

	return *numSpillingNodes == 0;
}


//TODO : vérifier comment ça fonctionne plus précisément
// Permet de diviser les noeuds qui ont trop de points
/* 
	-Création de 8 noeuds enfants pour chaque noeud à diviser
		-Chaque enfant hérite des propriétés du noeud parent avec coordonnées et niveau modifiés en conséquence
		-Enfants ajoutés au tableau de noeuds (nodes) grâce à atomicAdd(&stats->numNodes, 8) -> voir comment ça fonctionne
	-Ancien noeud mis dans file attente (chunkQueue)
		-Compteur de points remis à 0
	-Si noeud n'a pas de grille d'occupation, allocation d'une grille + remise à 0 des nouvelles grilles
*/	
bool doSplitting(Node* nodes, Node** spillingNodes, uint32_t* numSpillingNodes){
	auto grid = cg::this_grid();

	// split the spilling nodes
	// PRINT("split %i spilling nodes \n", *numSpillingNodes);
	processRange(*numSpillingNodes, [&](int spillNodeIndex){
		Node* spillingNode = spillingNodes[spillNodeIndex];

		// create child nodes
		uint32_t childOffset = atomicAdd(&stats->numNodes, 8);
		for(int i = 0; i < 8; i++){

			int cx = (i >> 2) & 1;
			int cy = (i >> 1) & 1;
			int cz = (i >> 0) & 1;

			Node child;
			child.counter = 0;
			child.numPoints = 0;
			child.points = nullptr;
			memset(&child.children, 0, 8 * sizeof(Node*));
			child.level = spillingNode->level + 1;
			child.X = 2 * spillingNode->X + cx;
			child.Y = 2 * spillingNode->Y + cy;
			child.Z = 2 * spillingNode->Z + cz;
			child.countIteration = 0;
			memcpy(&child.name[0], &spillingNode->name[0], 20);
			child.name[child.level] = i + '0';
			// child.voxels = Array<Point>();
			child.numVoxels = 0;
			child.numVoxelsStored = 0;
			
			nodes[childOffset + i] = child;

			spillingNode->children[i] = &nodes[childOffset + i];
		}

		// return chunks to chunkQueue
		Chunk* chunk = spillingNode->points;
		while(chunk != nullptr){
			
			Chunk* next = chunk->next;
			
			chunk->next = nullptr;
			int32_t oldIndex = atomicAdd(&stats->numAllocatedChunks, -1);
			int32_t newIndex = oldIndex - 1;
			chunkQueue[newIndex] = chunk;

			chunk = next;
		}

		spillingNode->numPoints = 0;
		spillingNode->points = nullptr;

		// allocate occupancy grid for the spilled node
		if(spillingNode->grid == nullptr){
			spillingNode->grid = (OccupancyGrid*)allocator_persistent->alloc(sizeof(OccupancyGrid));
		}
	});

	grid.sync();

	// clear the newly allocated occupancy grids
	uint32_t numElements = *numSpillingNodes * (GRID_NUM_CELLS / 32u);
	// numElements = 1;
	// PRINT("clear new occupancy grids, %u elements \n", numElements);
	grid.sync();
	processRange(numElements, [&](int cellIndex){
		int gridIndex = cellIndex / (GRID_NUM_CELLS / 32u);
		int localCellIndex = cellIndex % (GRID_NUM_CELLS / 32u);

		Node* spillingNode = spillingNodes[gridIndex];

		spillingNode->grid->values[localCellIndex] = 0;
	});
}

/*
	-Expander l'octree
		-Récupération de la grille sur laquelle on effectue les opérations
		-20 itérations (pourquoi ?)
			-Comptage des points dans les noeuds (doCounting)
				-Si nb points > MAX_POINTS_PER_NODE, on ajoute le noeud à la liste des noeuds à diviser
				-Points en trop stockés dans spilledPoints
			-Si fini, on sort de la boucle
			-Sinon, on split les noeuds stockés dans splilledPoints (doSplitting)
				-Création de 8 noeuds enfants pour chaque noeud à diviser
				-Ancien noeud mis dans file attente (chunkQueue)
				-Si noeud n'a pas de grille d'occupation, allocation d'une grille + remise à 0 des nouvelles grilles
	-Échantillonnage des voxels
		-Parcours de l'octree pour chaque point
		-Création d'un voxel pour	
*/
void expand(
	Node* root, Point* points, int numPoints, 
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Node* nodes, 
	Node** spillingNodes, uint32_t* numSpillingNodes,
	Point* spilledPoints, uint32_t* numSpilledPoints,
	uint32_t batchIndex
){
	auto grid = cg::this_grid();
	for(int i = 0; i < 20; i++){

		grid.sync();

		bool isFinished = doCounting(root, points, numPoints, 
			octreeMin, octreeMax, octreeSize,
			nodes, 
			spillingNodes, numSpillingNodes, 
			spilledPoints, numSpilledPoints,
			batchIndex + 1);

		grid.sync();

		if(isFinished) break;

		doSplitting(nodes, spillingNodes, numSpillingNodes);

		grid.sync();
	}

	grid.sync();
}

/*
	-Récupération de la grille sur laquelle on effectue les opérations
	-Calcul de la taille de grille (2^MAX_DEPTH)
	-Définition de la lambda traverse 
		-Pour un point passé en paramètre :
			-Calcul des coordonnées du noeud à la profondeur maximale de l'octree (savoir à quel noeud le point appartient sur la profondeur max)
			-Calcul des coordonnées entières du point par rapport au noeud racine
			-Initialisation du noeud courant à la racine
			-Parcours de l'octree jusqu'à un noeud feuille
				-Calcul des coordonnées du noeud courant à la profondeur maximale
				-Calcul des coordonnées entières du point par rapport au noeud courant
				-Calcul de l'index du noeud enfant
				-Appel de la fonction sampleVoxel
					-Si le noeud courant n'a pas d'enfant, on sort de la boucle -> on a trouvé la feuille où se trouve le point
					-Sinon, on passe au noeud enfant
	-Utilisation de processRange pour traiter ts les points de la grille (utilisation de traverse)
	-Utilisation de processRange pour traiter les points en trop stockés dans spilledPoints (utilisation de traverse)
*/
void voxelSampling(
	Node* root, Point* points, int numPoints, 
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Point* spilledPoints, uint32_t* numSpilledPoints
){
	auto grid = cg::this_grid();

	float fGridSize = pow(2.0f, float(MAX_DEPTH));

	auto traverse = [&](Point point){

		// node coordinate at MAX_DEPTH
		uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize;
		uint32_t Y = fGridSize * (point.y - octreeMin.y) / octreeSize;
		uint32_t Z = fGridSize * (point.z - octreeMin.z) / octreeSize;

		// integer point coordinate relative to root node
		uint32_t pX = MAX_DEPTH_GRIDSIZE * (point.x - octreeMin.x) / octreeSize;
		uint32_t pY = MAX_DEPTH_GRIDSIZE * (point.y - octreeMin.y) / octreeSize;
		uint32_t pZ = MAX_DEPTH_GRIDSIZE * (point.z - octreeMin.z) / octreeSize;

		Node* current = root;

		int level = 0;
		uint32_t level_X;
		uint32_t level_Y;
		uint32_t level_Z;
		uint32_t child_X;
		uint32_t child_Y;
		uint32_t child_Z;
		uint32_t childIndex;

		for(; level < MAX_DEPTH; level++){

			level_X = X >> (MAX_DEPTH - level - 1);
			level_Y = Y >> (MAX_DEPTH - level - 1);
			level_Z = Z >> (MAX_DEPTH - level - 1);

			child_X = level_X & 1;
			child_Y = level_Y & 1;
			child_Z = level_Z & 1;

			childIndex = (child_X << 2) | (child_Y << 1) | child_Z;

			sampleVoxel(current, pX, pY, pZ, point, octreeMin, octreeMax, octreeSize);

			if(current->children[childIndex] == nullptr){
				// current == leaf!
				break;
			}else{
				current = current->children[childIndex];
			}
		}
	};

	processRange(numPoints, [&](int index){
		Point point = points[index];

		traverse(point);
	});

	processRange(*numSpilledPoints, [&](int index){
		Point point = spilledPoints[index];

		traverse(point);
	});
}

/*
	Permet d'allouer les chunks de points pour chaque noeud
	-Récupération de la grille sur laquelle on effectue les opérations
	-Utilisation de processRange pour traiter ts les noeuds
		-Si le noeud est une feuille et que le nombre de points est inférieur au compteur, on alloue des chunks supplémentaires
			-Calcul du nombre de chunks requis ((numéro de points du noeud(?) + nombre de points max par chunk - 1) / nombre de points max par chunk) 
			-Calcul du nombre de chunks existants ((nb points du noeud + nb points max par chunk -1)/nb points max par chunk)
			-Calcul du nombre de chunks supplémentaires requis
			-Si des chunks supplémentaires sont requis (numAdditionallyRequiredChunks > 0)
				-Récupération du dernier chunk existant (on prend l'actuel et tant qu'il y a des chunks (numExistingChunks), on prend le suivant)
				-Pour chaque chunk supplémentaire requis
					-Incrémentation du compteur de chunks alloués (atomicAdd)
					-Initialisation d'un chunk à null
					-Si le compteur de chunks alloués est supérieur à la taille de la file d'attente, on alloue un nouveau chunk
					-Sinon, on prend un chunk de la file d'attente
					-Initialisation du chunk suivant à null
					-Si le chunk précédent est nul, on l'ajoute au noeud
					-Sinon, on ajoute le chunk au chunk précédent
	-Synchronisation de la grille
	-Si le thread est le premier, il met à jour la taille de la file d'attente
*/
void allocatePointChunks(Node* root, Point* points, int numPoints, Node* nodes){
	auto grid = cg::this_grid();

	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		if(node->isLeafFn())
		if(node->numPoints < node->counter){

			int numRequiredChunks = (node->counter + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;
			int numExistingChunks = (node->numPoints + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;
			int numAdditionallyRequiredChunks = numRequiredChunks - numExistingChunks;
			
			if(numAdditionallyRequiredChunks > 0){

				Chunk* prevChunk = node->points;
				for(int i = 1; i < numExistingChunks; i++){
					prevChunk = prevChunk->next;
				}

				for(int i_chunk = 0; i_chunk < numAdditionallyRequiredChunks; i_chunk++){
					uint32_t chunkIndex = atomicAdd(&stats->numAllocatedChunks, 1);
					Chunk* chunk = nullptr;

					if(chunkIndex >= stats->chunkPoolSize){
						// allocate a new chunk if chunk pool is too small
						chunk = (Chunk*)allocator_persistent->alloc(sizeof(Chunk));
					}else{
						// otherwise take from chunk pool
						chunk = chunkQueue[chunkIndex];
					}

					chunk->next = nullptr;

					if(prevChunk == nullptr){
						node->points = chunk;
						prevChunk = chunk;
					}else{
						prevChunk->next = chunk;
						prevChunk = chunk;
					}
				}
				
			}
		}
	});

	grid.sync();

	// raise chunk pool size counter if we allocated new chunks
	if(grid.thread_rank() == 0){
		stats->chunkPoolSize = max(stats->chunkPoolSize, stats->numAllocatedChunks);
	}
}


/*
	Insertion des points dans l'octree
	-Récupération de la grille sur laquelle on effectue les opérations
	-Calcule de la taille de la grille (2^MAX_DEPTH)
	-Définition de la lambda insertPoint
		-Pour un point passé en paramètre
			-Calcul des coordonnées du noeud à la profondeur maximale de l'octree
			-Initialisation du noeud courant à la racine
			-Parcours de l'octree jusqu'au noeud feuille où doit être ajouté le point
			-Incrémentation du nombre de points du noeud (atomicAdd)
			-Calcul de l'index du chunk correspondant au noeud (pointInNodeIndex / POINTS_PER_CHUNK)
			-Calcul de l'index du point dans le chunk (pointInNodeIndex % POINTS_PER_CHUNK)
			-Récupération du chunk correspondant au noeud
			-Vérif si chunk null -> si oui, return
			-Sinon, va au dernier chunk du noeud
	-Pour chaque point du noeud -> insertPoint
	-Synchronisation de la grille
	-Pour chaque point stocké dans spilledPoints -> insertPoint
		-Si trop de points stockés (> 3M), erreur + return
	-Synchronisation de la grille
*/
void insertPoints(
	Node* root, Point* points, uint32_t numPoints,
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Point* spilledPoints, uint32_t* numSpilledPoints,
	int batchIndex
){
	auto grid = cg::this_grid();

	// INSERT POINTS INTO NODES 
	// PRINT("insert points into nodes \n");
	float fGridSize = pow(2.0f, float(MAX_DEPTH));
	// float3 octreeSize = octreeMax - octreeMin;
	
	auto insertPoint = [&](Point point){
		// node coordinate at MAX_DEPTH
		uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize;
		uint32_t Y = fGridSize * (point.y - octreeMin.y) / octreeSize;
		uint32_t Z = fGridSize * (point.z - octreeMin.z) / octreeSize;

		Node* current = root;

		// traverse to leaf node, compute some data about it
		int level = 0;
		uint32_t level_X;
		uint32_t level_Y;
		uint32_t level_Z;
		uint32_t child_X;
		uint32_t child_Y;
		uint32_t child_Z;
		uint32_t childIndex;

		for(; level < MAX_DEPTH; level++){

			level_X = X >> (MAX_DEPTH - level - 1);
			level_Y = Y >> (MAX_DEPTH - level - 1);
			level_Z = Z >> (MAX_DEPTH - level - 1);

			child_X = level_X & 1;
			child_Y = level_Y & 1;
			child_Z = level_Z & 1;

			childIndex = (child_X << 2) | (child_Y << 1) | child_Z;

			if(current->children[childIndex] == nullptr){
				// current == leaf!
				break;
			}else{
				current = current->children[childIndex];
			}
		}

		Node* leaf = current;

		uint32_t pointInNodeIndex = atomicAdd(&leaf->numPoints, 1);
		uint32_t chunkIndex = pointInNodeIndex / POINTS_PER_CHUNK;
		uint32_t pointInChunkIndex = pointInNodeIndex % POINTS_PER_CHUNK;

		Chunk* chunk = leaf->points;

		if(chunk == nullptr){
			// printf("chunk is NULL: %s \n", leaf->name);
			cudaprint->print("chunk is NULL: {} \n", (const char*)leaf->name);

			return;
		}

		for(int i = 0; i < chunkIndex; i++){
			if(chunk == 0) break;

			chunk = chunk->next;
		}

		chunk->points[pointInChunkIndex] = point;
	};

	// INSERT POINTS FROM CURRENT BATCH
	processRange(numPoints, [&](int pointID){
		Point point = points[pointID];
		
		insertPoint(point);
	});

	grid.sync();

	// INSERT POINTS FROM SPILLED NODES
	// (essentially redistributing from spilled to new leaves)
	processRange(*numSpilledPoints, [&](int pointID){

		if(pointID > 3'000'000){
			printf("too many spilled points \n");
			return;
		}

		Point point = spilledPoints[pointID];

		insertPoint(point);
	});

	grid.sync();
}

/*
	Allocation de chunks de voxels pour chaque noeud
	-Récupération de la grille sur laquelle on effectue les opérations
	-Définition de la lambda processRange pour un noeud donné
		-Calcul si le noeud a besoin d'un nouveau chunk de voxels ((nb points du noeud + nb points max par chunk - 1) / nb points max par chunk)
		-Si non (requiredChunk == 0), return
		-Si oui, check de si le node a déjà un chunk alloué
			-Si non, on alloue un nouveau chunk grâce à allocator_persistent->alloc + initialisation du chunk suivant à null
		-Récupération de la liste de chunks de voxel du noeud
		-Itération sur le nombre de chunk demandés :
			-Si le suivant est null, on alloue un nouveau + initialisation du suivant du courant au chunk nouvellement alloué
			-Passage au chunk suivant
*/
void allocateVoxelChunks(Node* nodes){
	auto grid = cg::this_grid();

	processRange(stats->numNodes, [&](int nodeIndex){

		Node* node = &nodes[nodeIndex];

		int requiredChunks = (node->numVoxels + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;

		if(requiredChunks == 0) return;

		if(node->voxelChunks == nullptr){
			node->voxelChunks = (Chunk*)allocator_persistent->alloc(sizeof(Chunk));
			node->voxelChunks->next = nullptr;
		}

		Chunk* chunk = node->voxelChunks;

		for(int chunkIndex = 1; chunkIndex < requiredChunks; chunkIndex++){

			if(chunk->next == nullptr){
				Chunk* newChunk = (Chunk*)allocator_persistent->alloc(sizeof(Chunk));
				newChunk->next = nullptr;
				chunk->next = newChunk;
			}

			chunk = chunk->next;

		}

	});
}

/*
	Insertion des voxels dans l'octree
	-Récupération de la grille sur laquelle on effectue les opérations
	-Définition de la lambda processRange pour chaque voxel stocké dans backlog_voxels
		-Récupération du voxel et du noeud cible
		-Incrémentation du nombre de voxels stockés dans le noeud (atomicAdd)
		-Calcul de l'index du chunk correspondant au noeud (numVoxelsStored / POINTS_PER_CHUNK)
		-Itération sur les chunks pour atteindre le chunk correspondant
		-Calcul de l'index du voxel dans le chunk (numVoxelsStored % POINTS_PER_CHUNK)
		-Insertion du voxel dans le chunk
*/
void insertVoxels(int batchIndex){
	auto grid = cg::this_grid();

	// if(grid.thread_rank() == 0 && (*numBacklogVoxels) > 1'000'000){
	// 	printf("numBacklogVoxels: %.1f M\n", float(*numBacklogVoxels) / 1'000'000.0f);
	// }

	processRange(*numBacklogVoxels, [&](int index){
		Point voxel = backlog_voxels[index];
		Node* target = backlog_targets[index];

		uint32_t voxelIndex = atomicAdd(&target->numVoxelsStored, 1);
		uint32_t chunkIndex = voxelIndex / POINTS_PER_CHUNK;

		Chunk* chunk = target->voxelChunks;

		for(int i = 0; i < chunkIndex; i++){
			chunk = chunk->next;
		}

		uint32_t chunkLocalVoxelIndex = voxelIndex % POINTS_PER_CHUNK;

		chunk->points[chunkLocalVoxelIndex] = voxel;
	});
}


/*
	?
	-Récupération de la grille sur laquelle on effectue les opérations
	-Récupération du bloc où seront fait les opérations
	-Si rank == 0, initialisation des compteurs
	-Synchronisation de la grille
	-Utilisation d'expand pour augmenter l'octree
	-Synchronisation de la grille
	-Création des voxels à partir des points (voxelSampling)
	-Synchronisation de la grille
	-Allocation de mémoire pour les points dans les noeuds feuilles (allocatePointChunks)
	-Synchronisation de la grille
	-Allocation de mémoire pour les voxels de chaque noeud (allocateVoxelChunks)
	-Synchronisation de la grille
	-Insertion des points dans les noeuds feuilles (insertPoints)
	-Synchronisation de la grille
	-Insertion des voxels dans l'octree (insertVoxels)
	-Synchronisation de la grille
	-Si rank == 0, calcul du temps d'exécution de chaque étape enregistré au préalable
*/

void addBatch(
	Point* points, uint32_t batchSize,
	int batchIndex,
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Node* nodes, 
	Node** spillingNodes, uint32_t* numSpillingNodes,
	Point* spilledPoints, uint32_t* numSpilledPoints
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	auto t_00 = nanotime();

	if(grid.thread_rank() == 0){
		*numSpilledPoints = 0;
		*numBacklogVoxels = 0;
	}

	grid.sync();

	Node* root = &nodes[0];

	auto t_10 = nanotime();

	// EXPAND OCTREE
	expand(root, points, batchSize, 
		octreeMin, octreeMax, octreeSize,
		nodes, 
		spillingNodes, numSpillingNodes, 
		spilledPoints, numSpilledPoints,
		batchIndex
	);

	grid.sync();

	auto t_20 = nanotime();

	// CREATE VOXEL SAMPLES
	voxelSampling(root, points, batchSize, 
		octreeMin, octreeMax, octreeSize,
		spilledPoints, numSpilledPoints
	);

	grid.sync();

	auto t_30 = nanotime();

	// ALLOCATE MEMORY FOR POINTS IN LEAF NODES
	allocatePointChunks(root, points, batchSize, nodes);

	grid.sync();

	auto t_40 = nanotime();

	// ALLOCATE VOXEL MEMORY FOR EACH NODE
	allocateVoxelChunks(nodes);

	grid.sync();

	auto t_50 = nanotime();

	// INSERT POINTS INTO LEAF NODES
	insertPoints(root, points, batchSize,
		octreeMin, octreeMax, octreeSize,
		spilledPoints, numSpilledPoints, batchIndex
	);

	grid.sync();

	auto t_60 = nanotime();

	// PRINT("insertVoxels()\n");
	insertVoxels(batchIndex);

	grid.sync();

	auto t_70 = nanotime();

	grid.sync();

	if(grid.thread_rank() == 0){
		float t_00_70 = double(t_70 - t_00) / 1'000'000.0;
		float t_00_10 = double(t_10 - t_00) / 1'000'000.0;
		float t_10_20 = double(t_20 - t_10) / 1'000'000.0;
		float t_20_30 = double(t_30 - t_20) / 1'000'000.0;
		float t_30_40 = double(t_40 - t_30) / 1'000'000.0;
		float t_40_50 = double(t_50 - t_40) / 1'000'000.0;
		float t_50_60 = double(t_60 - t_50) / 1'000'000.0;
		float t_60_70 = double(t_70 - t_60) / 1'000'000.0;

		cudaprint->print("t_00_70: {:.3f}, t_00_10: {:.3f}, expand: {:.3f}, createVoxels: {:.3f}, t_30_40: {:.3f}, t_40_50: {:.3f}, insertPoints: {:.3f}, t_60_70: {:.3f} \n", 
			t_00_70,
			t_00_10,
			t_10_20,
			t_20_30,
			t_30_40,
			t_40_50,
			t_50_60,
			t_60_70
		);
	}
}


/*
	CEST CA QUI EST APPELE DANS [main_progressive_octree.cpp](le main), updateOctree
	Kernel Cuda pour la construction de l'octree (__global__ = c'est sur le GPU, mais on peut l'appeler depuis le CPU)
	-Initialisation des variables (grille, bloc, temps initial, allocator, allocator_persistent, numBatchesUploaded_global) TODO : voir comment fonctionne leur fonction d'allocation
	-Allocation de la mémoire pour les variables de stats
		-ellapsed_nanos : temps écoulé
		-backlog_voxels : mémoire pour les voxels en attente
		-backlog_targets : mémoire pour les noeuds en attente
		-numBacklogVoxels : nombre de voxels en attente
		-spillingNodes : mémoire pour les noeuds à diviser
		-numSpillingNodes : nombre de noeuds à diviser
		-spilledPoints : mémoire pour les points à redistribuer
		-numSpilledPoints : nombre de points à redistribuer
		-chunkQueue : file d'attente pour les chunks
	-Synchronisation de la grille
	-Création de la boîte englobante de l'octree
		-boxSize : taille de la boîte
		-octreeSize : taille de l'octree (max sur les 3 dimensions de la box)
		-octreeMin : coin inférieur de la boîte (?)
		-octreeMax : coin supérieur de la boîte (octreeMin + octreeSize)
		-cubePosition : centre de la boîte vis à vis de l'octree
	-Synchronisation de la grille
	-Création d'une variable globale numBatchesUploaded qui récupère la valeur de _numBatchesUploaded_volatile	(pour être sûr que ts les threads aient la même valeur)
	-Synchronisation de la grille
	-Incrémentation de numBatchesUploaded (atomicAdd) pour récupérer le nombre de batchs à traiter
	-Synchronisation de la grille
	-Création de variables pour les tailles max des sous-patchs et des batchs
	-Création de variables pour le nombre de batch à créer, le numéro du 1er batch, le numéro du dernier batch
	-Boucle sur les batchs :
		-Récupération de la taille du batch
		-Récupération des points du batch
		-Récupération d'infos sur la mémoire (taille totale, utilisation actuelle) + calcul de la taille que prendra le batch ds la mémoire
		-Si mémoire presque pleine + rank == 0 :
			-Message d'erreur + passage d'un booléen sur la mémoire à true + sortie de la boucle
		-Appel à addBatch pour ajouter le batch à l'octree
		-Synchronisation de la grille
		-Si rank == 0 :
			-Incrémentation du compteur de batchs traités et du nombre de points traités
		-Synchronisation de la grille
		-Si rank == 0 :
			-Calcul du temps écoulé depuis le début de la frame
		-Synchronisation de la grille
		-Si temps écoulé depuis le début > 16ms :
			-On sort de la boucle
	-Calcul de plusieurs stats
*/
extern "C" __global__
void kernel_construct(
	const Uniforms uniforms,
	Point* points,
	uint32_t* buffer,
	uint8_t* buffer_persistent,
	Node* nodes,
	Stats* _stats,
	uint64_t* frameStartTimestamp,
	CudaPrint* _cudaprint,
	uint32_t* _numBatchesUploaded_volatile, // could change at any moment by the parallel upload stream
	uint32_t* batchSizes
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	auto tStart = nanotime();

	cudaprint = _cudaprint;

	if(grid.thread_rank() == 0){
		*frameStartTimestamp = tStart;
	}
	grid.sync();

	stats = _stats;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;
	allocator_persistent = (AllocatorGlobal*)buffer_persistent;

	uint32_t& numBatchesUploaded_global = *allocator->alloc<uint32_t*>(4);

	grid.sync();

	// ALLOCATE STUFF
	uint64_t& ellapsed_nanos     = *allocator->alloc<uint64_t*>(8);

	// memory for "backlog"
	backlog_voxels     = allocator->alloc<Point*>(VOXEL_BACKLOG_CAPACITY * sizeof(Point));
	backlog_targets    = allocator->alloc<Node**>(VOXEL_BACKLOG_CAPACITY * sizeof(Node*));
	numBacklogVoxels   = allocator->alloc<uint32_t*>(4);

	// List of nodes that received too many points and ned to be split
	Node** spillingNodes         =  allocator->alloc<Node**>(100'000 * sizeof(Node*));
	uint32_t* numSpillingNodes   =  allocator->alloc<uint32_t*>(4);

	// Points from spilled&split nodes that need to be redistributed to the new leaves
	// It's quite large because in the worst case, a single point can trigger 
	// <MAX_POINTS_PER_NODE> points to be spilled
	Point* spilledPoints = allocator->alloc<Point*>(10'000'000 * sizeof(Point));
	uint32_t* numSpilledPoints = allocator->alloc<uint32_t*>(4);

	chunkQueue = allocator->alloc<Chunk**>(sizeof(Chunk*) * 1'000'000);

	grid.sync();

	float3 boxSize      = uniforms.boxMax - uniforms.boxMin;
	float octreeSize    = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 octreeMin    = uniforms.boxMin;
	float3 octreeMax    = octreeMin + octreeSize;
	float3 cubePosition = uniforms.boxMin + octreeSize * 0.5f;

	auto tStartExpand = nanotime();

	grid.sync();

	// We need to make sure that all threads get the same "numBatchesUploaded" value.
	// _numBatchesUploaded_volatile is likely not safe and may be modified at any time by the parallel async upload stream.
	if(grid.thread_rank() == 0){
		numBatchesUploaded_global = *_numBatchesUploaded_volatile;
	}
	grid.sync();
	uint32_t numBatchesUploaded = atomicAdd(&numBatchesUploaded_global, 0);

	grid.sync();

	constexpr int MAX_SUBPATCH_SIZE      = 1'000'000;
	constexpr uint64_t MAX_BATCH_SIZE    = 1'000'000;

	uint32_t numBatches = min(numBatchesUploaded - stats->batchletIndex, 20);
	uint32_t firstBatch = stats->batchletIndex;
	uint32_t lastBatch = firstBatch + numBatches;

	auto tStart_addBatches = nanotime();
	
	// ADD BATCHES OF POINTS TO OCTREE
	for(int batchIndex = firstBatch; batchIndex < lastBatch; batchIndex++){
		
		uint32_t ringSlotIndex = batchIndex % BATCH_STREAM_SIZE;
		uint32_t batchSize = batchSizes[ringSlotIndex];
		Point* sub_points = points + ringSlotIndex * MAX_BATCH_SIZE;

		uint64_t memCapacity = uniforms.persistentBufferCapacity;
		uint64_t memUsed = allocator_persistent->offset;
		uint64_t safetyMargin = 200'000'000;
		bool memCapacityReached = memUsed + safetyMargin >= memCapacity;

		if(memCapacityReached && grid.thread_rank() == 0){
			printf("persistent memory capacity almost reached, ignoring further points. capacity: %i MB, used: %i MB \n",
				int32_t(memCapacity / (1024llu * 1024llu)),
				int32_t(memUsed / (1024llu * 1024llu))
			);

			stats->memCapacityReached = true;
		}else if(!memCapacityReached){
			stats->memCapacityReached = false;
		}

		if(memCapacityReached) break;

		addBatch(
			sub_points, batchSize,
			stats->batchletIndex,
			octreeMin, octreeMax, octreeSize,
			nodes, 
			spillingNodes, numSpillingNodes,
			spilledPoints, numSpilledPoints
		);
		
		grid.sync();

		if(grid.thread_rank() == 0){
			atomicAdd(&stats->batchletIndex, 1);
			atomicAdd(&stats->numPointsProcessed, batchSize);
		}

		grid.sync();

		if(grid.thread_rank() == 0){
			ellapsed_nanos = nanotime() - (*frameStartTimestamp);
		}

		grid.sync();

		// skip remaining batches if time budget is exceeded
		float ellapsed_ms = float(ellapsed_nanos) / 1'000'000.0f;
		if(ellapsed_ms > MAX_PROCESSING_TIME){
			// if(grid.thread_rank() == 0){
			// 	uint32_t numBatchesAdded = batchIndex - firstBatch;
			// 	printf("%2u / %2u batches processed this frame. stopping early after %.1f ms. \n",
			// 		numBatchesAdded, numBatches, ellapsed_ms
			// 	);
			// }

			break;
		}
	}

	auto tEndExpand = nanotime();
	float durationExpandMS = double(tEndExpand - tStartExpand) / 1'000'000.0;

	grid.sync();

	// compute stats about the octree
	uint32_t* counter_inner           = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_leaves          = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_nonempty_leaves = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_points          = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_voxels          = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_chunks_points   = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_chunks_voxels   = allocator->alloc<uint32_t*>(4);

	if(grid.thread_rank() == 0){
		*counter_inner           = 0;
		*counter_leaves          = 0;
		*counter_nonempty_leaves = 0;
		*counter_points          = 0;
		*counter_voxels          = 0;
		*counter_chunks_points   = 0;
		*counter_chunks_voxels   = 0;
	}
	grid.sync();

	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		if(node->isLeafFn()){
			atomicAdd(counter_leaves, 1);
			atomicAdd(counter_points, node->numPoints);
			atomicAdd(counter_chunks_points, (node->numPoints + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK);

			if(node->numPoints > 0){
				atomicAdd(counter_nonempty_leaves, 1);
			}
		}else{
			atomicAdd(counter_inner, 1);
			atomicAdd(counter_voxels, node->numVoxels);
			atomicAdd(counter_chunks_voxels, (node->numVoxels + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK);
		}

	});

	grid.sync();

	if(grid.thread_rank() == 0){
		stats->numInner                  = *counter_inner;
		stats->numLeaves                 = *counter_leaves;
		stats->numNonemptyLeaves         = *counter_nonempty_leaves;
		stats->numPoints                 = *counter_points;
		stats->numVoxels                 = *counter_voxels;
		stats->numChunksPoints           = *counter_chunks_points;
		stats->numChunksVoxels           = *counter_chunks_voxels;
		stats->allocatedBytes_momentary  = allocator->offset;
		stats->allocatedBytes_persistent = allocator_persistent->offset;
		stats->frameID                   = uniforms.frameCounter;
	}
}

