#include "structures.cuh"
#include <iostream>
#include <filesystem>
#include <cstdint>
#include <fstream>
#include <vector>
#include <cuda.h>
using namespace std;
namespace fs = std::filesystem;

struct PointerData{
    CUdeviceptr node_ptr;
    int children[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    bool saved = false;
};

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

string getNameAsString(Node* node) {
    return string(reinterpret_cast<const char*>(node->name));
}

void saveNode(Node* node, const fs::path& folderPath){
    string filename = folderPath.string() + "/" + getNameAsString(node);
    ofstream outFile(filename, ios::binary);
    if (!outFile) {
        cerr << "Error: Cannot open file " << filename << " for writing." << endl;
        return;
    }
    outFile.write(reinterpret_cast<const char*>(&node), sizeof(Node));
    outFile.close();
    cout << "Saved node to: " << filename << endl;
}

void saveNodes(const fs::path& folderPath, vector<PointerData> pointerDataVector){
    //Création du dossier où seront stocker les fichiers de l'octree
    createSaveFolder(folderPath);
    int nodeIndex = -1;
    //On peut faire for auto car on sait pas si on doit faire les noeuds dans l'ordre
    while(true){
        nodeIndex +=1;
        int numNodes = pointerDataVector.size();
        if(nodeIndex >= numNodes) break;
        if(!pointerDataVector[nodeIndex].saved){
            //Faut récupérer les données selon le pointeur
            Node* node = (Node*)malloc(sizeof(Node));
            cuMemcpyDtoH(node, pointerDataVector[nodeIndex].node_ptr, sizeof(Node));
            saveNode(node, folderPath);
            pointerDataVector[nodeIndex].saved = true;
            free(node);
        }
    }
}