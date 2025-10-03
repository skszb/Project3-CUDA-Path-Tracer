#pragma once
#include <vector_types.h>

class GuiDataContainer;
class Scene;
void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
