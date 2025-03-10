#pragma once


constexpr float PI = 3.1415; // Simplification de PI
// constexpr int MAX_POINTS_PER_NODE = 100;
// constexpr int MAX_POINTS_PER_NODE = 5'000;
// constexpr uint32_t POINTS_PER_CHUNK = 1000;
constexpr bool RIGHTSIDE_BOXES = false;
constexpr bool RIGHTSIDE_NODECOLORS = false;

constexpr bool ENABLE_TRACE = false;
// constexpr int MAX_DEPTH = 20;
// constexpr float MAX_DEPTH_GRIDSIZE = 268'435'456.0f;

// constexpr int MAX_POINTS_PER_NODE       = 5'000;
// constexpr uint32_t POINTS_PER_CHUNK     = 256;
// constexpr uint32_t GRID_SIZE            = 64;
// constexpr uint32_t GRID_NUM_CELLS       = GRID_SIZE * GRID_SIZE * GRID_SIZE;
// constexpr int MAX_DEPTH                 = 17;
// constexpr float MAX_DEPTH_GRIDSIZE      = 16'777'216.0f;

// Constantes
constexpr int MAX_POINTS_PER_NODE    = 50'000; // Nombre de points maximum par noeud
constexpr uint32_t POINTS_PER_CHUNK  = 1000; // Nombre de points maximum par chunk
constexpr uint32_t GRID_SIZE         = 128; // Taille de la grille
constexpr uint32_t GRID_NUM_CELLS    = GRID_SIZE * GRID_SIZE * GRID_SIZE; // Nombre de cellules de la grille
constexpr int MAX_DEPTH              = 20; // Profondeur maximale de l'octree
constexpr float MAX_DEPTH_GRIDSIZE   = 268'435'456.0f; // Taille maximale de la grille

constexpr uint64_t BATCH_STREAM_SIZE = 50; // Taille du batch de stream

// Structure représentant un point
struct Point{
	float x; // Coordonnée x
	float y; // Coordonnée y
	float z; // Coordonnée z
	uint32_t color; // Couleur du point (probablement ARGB)
};

//Structure représentant un voxel
struct Voxel{
	uint8_t X; //Coordonnée x
	uint8_t Y; //Coordonnée y
	uint8_t Z; //Coordonnée z
	uint8_t filler; //Remplissage -> alignement mémoire (?)
	uint32_t color; // Couleur du voxel (probablement ARGB)
};

struct Lines{
	unsigned int count = 0;
	unsigned int padding0;
	unsigned int padding1;
	unsigned int padding2;
	Point* vertices;
};
float dot_cuda(const float4& a, const float4& b){
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}
// Surcharge de l'opérateur de multiplication pour les float4 -> reourne un vecteur de 4 floats
float4 operator*(const mat4& a, const float4& b){
	return make_float4(
		dot_cuda(a.rows[0], b),
		dot_cuda(a.rows[1], b),
		dot_cuda(a.rows[2], b),
		dot_cuda(a.rows[3], b)
	);
}

// Structure représentant un chunk de points
struct Chunk{
	Point points[POINTS_PER_CHUNK]; // Tableau de points
	int size; // Nombre de points actuellement dans le chunk
	int padding_0; // Remplissage -> alignement mémoire (?)
	Chunk* next; // Pointeur vers le prochain chunk
};

// Structure représentant une grille d'occupation
struct OccupancyGrid{
	// gridsize^3 occupancy grid; 1 bit per voxel
	uint32_t values[GRID_NUM_CELLS / 32u]; // Tableau de 32 bits -> 1 bit par voxel
};

// Structure représentant un noeud de l'octree
struct Node{
	Node* children[8]; // Tableau de 8 enfants
	uint32_t counter = 0; // Compteur de points pour le noeud complet, enfants inclus
	// uint32_t counters[8] = {0, 0, 0, 0, 0, 0, 0, 0};

	uint32_t numPoints = 0;
	uint32_t level = 0; // Niveau de profondeur dans l'octree
	uint32_t X = 0; // Coordonnée x dans l'octree
	uint32_t Y = 0; // Coordonnée y dans l'octree
	uint32_t Z = 0; // Coordonnée z dans l'octree
	uint32_t countIteration = 0;
	uint32_t countFlag = 0;
	uint8_t name[20] = {'r', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Nom du noeud (utilisé dans les fichiers)
	bool visible = false; // Visibilité du noeud (à afficher ou non)
	bool isFiltered = false; // Filtre appliqué ou non (?) -> pas de réel intéret pour le moment, juste les feuilles sont mis en true, fin 
	bool isLeaf = true; // Est une feuille ?
	bool isLarge = false; // Est large ? (comment ça, mon reuf ?) -> permet de différencier si le noeud est trop gros pour être affiché et donc afficher les enfants

	OccupancyGrid* grid = nullptr; // Grille d'occupation pour le noeud
	
	Chunk* points = nullptr; // Points du noeud
	Chunk* voxelChunks = nullptr; // Voxel chunks du noeud

	uint32_t numVoxels = 0; // Nombre de voxels
	uint32_t numVoxelsStored = 0; // Nombre de voxels stockés
 
	// bool spilled(){
	// 	return counter > MAX_POINTS_PER_NODE;
	// }

	// Vérifie si le noeud est une feuille = ts les enfants sont null
	bool isLeafFn(){

		if(children[0] != nullptr) return false;
		if(children[1] != nullptr) return false;
		if(children[2] != nullptr) return false;
		if(children[3] != nullptr) return false;
		if(children[4] != nullptr) return false;
		if(children[5] != nullptr) return false;
		if(children[6] != nullptr) return false;
		if(children[7] != nullptr) return false;

		return true;
	}

	// récupère l'ID du noeud
	uint64_t getID(){
		uint64_t id = 0;

		id = id | ((name[ 0] == 'r' ? 1 : 0));
		id = id | ((name[ 1] - '0') <<  3);
		id = id | ((name[ 2] - '0') <<  6);
		id = id | ((name[ 3] - '0') <<  9);
		id = id | ((name[ 4] - '0') << 12);
		id = id | ((name[ 5] - '0') << 15);
		id = id | ((name[ 6] - '0') << 18);
		id = id | ((name[ 7] - '0') << 21);
		id = id | ((name[ 8] - '0') << 24);
		id = id | ((name[ 9] - '0') << 27);
		id = id | (uint64_t((name[10] - '0')) << 30);
		id = id | (uint64_t((name[11] - '0')) << 33);
		id = id | (uint64_t((name[12] - '0')) << 36);
		id = id | (uint64_t((name[13] - '0')) << 39);
		id = id | (uint64_t((name[14] - '0')) << 42);
		id = id | (uint64_t((name[15] - '0')) << 45);
		id = id | (uint64_t((name[16] - '0')) << 48);
		id = id | (uint64_t((name[17] - '0')) << 51);
		id = id | (uint64_t((name[18] - '0')) << 53);

		return id;
	}

};