# auto-fix-uv-seams
Tries to fix the seams using Linear Least Squares
Initial source code : https://gist.github.com/ssylvan/18fb6875824c14aa2b8c

## Getting Started

Grab this repo, tinygltf: https://github.com/syoyo/tinygltf and stb : https://github.com/nothings/stb libraries 

### Prerequisites

tinygltf and stb libraries are required
The code is compiled with g++ :  g++ -fopenmp -g src/TextureBorderGenerator.cpp -o MyApp.exe -I libs/stb -I libs/tinigltf -L libs/stb -L libs/tinigltf
