#include <iostream>
#include <filesystem>
#include <cstdint>
#include <fstream>
#include <vector>
using namespace std;

constexpr uint32_t GRID_SIZE         = 128; // Taille de la grille
constexpr uint32_t GRID_NUM_CELLS    = GRID_SIZE * GRID_SIZE * GRID_SIZE; // Nombre de cellules de la grille
constexpr uint32_t POINTS_PER_CHUNK  = 1000; // Nombre de points maximum par chunk


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
	uint32_t numVoxels = 0; // Nombre de voxels
	uint32_t numVoxelsStored = 0; // Nombre de voxels stockés

    OccupancyGrid* grid = nullptr; // Grille d'occupation pour le noeud
	
	Chunk* points = nullptr; // Points du noeud
	Chunk* voxelChunks = nullptr; // Voxel chunks du noeud
 
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

namespace fs = std::filesystem;

void clearFolder(const fs::path& folderPath) {
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        fs::remove_all(entry);
    }
}

void createSaveFolder(const fs::path& folderPath) {
    if (fs::exists(folderPath)) {
        std::cout << "Folder exists. Clearing contents..." << std::endl;
        clearFolder(folderPath);
    } else {
        std::cout << "Folder does not exist. Creating it..." << std::endl;
        fs::create_directories(folderPath);
    }
}

string getNameAsString(Node node) {
    return string(reinterpret_cast<const char*>(node.name));
}

void saveNodes(const fs::path& folderPath, std::vector<Node>& nodes){
    createSaveFolder(folderPath);
    for (const auto& node : nodes) {
        string filename = folderPath.string() + "/" + getNameAsString(node) + ".dat";

        ofstream outFile(filename, ios::binary);
        if (!outFile) {
            cerr << "Error: Cannot open file " << filename << " for writing." << endl;
            return;
        }

        outFile.write(reinterpret_cast<const char*>(&node), sizeof(Node));
        outFile.close();
        cout << "Saved node to: " << filename << endl;
    }
}

//TODO : va falloir save le parent aussi
void saveChunks(){

}

void savePoints(){

}


void readNodes(const fs::path& folderPath){
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        string filename = folderPath.string() + "/"+entry.path().filename().string();;

        Node node;
        ifstream inFile(filename, ios::binary);
        if (!inFile) {
            cerr << "Error: Cannot open file " << filename << " for reading." << endl;
            return;
        }

        inFile.read(reinterpret_cast<char*>(&node), sizeof(Node));
        inFile.close();
        cout << "Read from: " << filename
         << " - Counter: " << node.counter
         << " - Name: " << getNameAsString(node)
         << " - Visible: " << (node.visible ? "true" : "false")
         << endl;
    }
}

void readChunks(){

}

void readPoints(){

}

int main(){
    fs::path saveFolder = "octree";
    createSaveFolder(saveFolder);

    std::vector<Node> nodes(3);
    
    // Initialize some sample data
    nodes[0].counter = 100;
    nodes[0].numPoints = 10;
    nodes[0].level = 1;
    nodes[0].X = 0; nodes[0].Y = 1; nodes[0].Z = 2;
    nodes[0].countIteration = 5;
    nodes[0].countFlag = 3;
    nodes[0].name[0] = 'r'; nodes[0].name[1] = '1';
    nodes[0].visible = true;
    nodes[0].isLeaf = true;
    nodes[0].isLarge = false;
    nodes[0].numVoxels = 50;
    nodes[0].numVoxelsStored = 45;

    nodes[1].counter = 200;
    nodes[1].numPoints = 20;
    nodes[1].level = 2;
    nodes[1].X = 1; nodes[1].Y = 2; nodes[1].Z = 3;
    nodes[1].countIteration = 6;
    nodes[1].countFlag = 4;
    nodes[1].name[0] = 'r'; nodes[1].name[1] = '2';
    nodes[1].visible = true;
    nodes[1].isLeaf = false;
    nodes[1].isLarge = true;
    nodes[1].numVoxels = 100;
    nodes[1].numVoxelsStored = 95;

    nodes[2].counter = 300;
    nodes[2].numPoints = 30;
    nodes[2].level = 3;
    nodes[2].X = 2; nodes[2].Y = 3; nodes[2].Z = 4;
    nodes[2].countIteration = 7;
    nodes[2].countFlag = 5;
    nodes[2].name[0] = 'r'; nodes[2].name[1] = '3';
    nodes[2].visible = false;
    nodes[2].isLeaf = true;
    nodes[2].isLarge = false;
    nodes[2].numVoxels = 150;
    nodes[2].numVoxelsStored = 145;

    saveNodes(saveFolder, nodes);
    readNodes(saveFolder);
    
    // Read the array of nodes back
    
}

