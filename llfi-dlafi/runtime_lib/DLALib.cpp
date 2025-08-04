
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <set>

#include "DLALib.h"
#define SETSIZE 710

static int globalMLLayerNo=0,fmul_count=0,myfault=0;
static long long shape[10000]= {0};
static int layers[10000]={0};

static int64_t DLALayerNum=0;
static std::set <int> input_indices[SETSIZE], weight_indices[SETSIZE] , weight_indices_prof[18][16][SETSIZE];
// static std::unordered_set <std::pair<int, int> > input_weight_indices[SETSIZE];
static int  SAdim=0, sampler=1;
static int total_layers = 0, fi_layer = -1;
static bool do_fi=false;
static FILE *IndexFile, *MLlayerFile, *SAConfigFile, *Layer_SAConfigFile;
static bool isDeviceSA=false, isDataflowWS=false;


extern "C" {
#include "Utils.h"

    //read SAConfig
    void read_SA_config(){
        fprintf(stderr,"Read SA config!!\n");
        char SAConfigFileName[80];
        strncpy(SAConfigFileName, "llfi.SA.config.txt", 80);
        SAConfigFile = fopen(SAConfigFileName, "r");
        if (SAConfigFile == NULL) {
            fprintf(stderr, "ERROR: Unable to open (read) Index stat file ! %s\n",
                    SAConfigFile);
            exit(1);
        }

        char line[100];
        while (fgets(line, sizeof(line), SAConfigFile)) {
            if (strncmp(line, "deviceType=", 11) == 0) {
                    if (strncmp(line, "CPU", 3) == 0) {
                        isDeviceSA = false;
                    }
                    else{
                        isDeviceSA = true;                    
                    }
                    fprintf(stderr,"|||||**||||||isDeviceSA=%d \n",(int)isDeviceSA);
            }
            if (strncmp(line, "SystolicArrayDataflow=", 22) == 0) {
                    line[strcspn(line, "\n")] = 0;
                    char* arg = strtok(line, "=");
                    char* value = strtok(NULL, "");
                    if (strncmp(value, "OS", 2) == 0) {
                        isDataflowWS = false;
                    }
                    else if (strncmp(value, "WS", 2) == 0){
                        isDataflowWS = true;                    
                    }
                    fprintf(stderr,"|||||**||||||isDataflowWS=%d \n",(int)isDataflowWS);
                    fprintf(stderr,line);
                    fprintf(stderr,"\n\n");
            }
            if (strncmp(line, "SystolicArrayDimension=", 23) == 0) {
                    line[strcspn(line, "\n")] = 0;
                    char* arg = strtok(line, "=");
                    char* value = strtok(NULL, "");
                    SAdim =  atoll(value);
                    fprintf(stderr,"|||||**||||||SAdim=%d \n",(int)SAdim);
            }
            if (strncmp(line, "SystolicArraySample=", 20) == 0) {
                    line[strcspn(line, "\n")] = 0;
                    char* arg = strtok(line, "=");
                    char* value = strtok(NULL, "");
                    sampler =  atoll(value);
                    fprintf(stderr,"|||||**||||||sampler=%d \n",(int)sampler);
            }
        }
        fclose(SAConfigFile);
        
        fprintf(stderr,"done2\n");
    }
    //initilization of FI
    void read_all_indices(){
        char line[10000];
        while (fgets(line, sizeof(line), IndexFile)) {
            if (strncmp(line, "ml_layer_W", 10) == 0) {
                int layerNum;

                char *token = strtok(line, ",");
                int tokenCount = 0;
                while (token != NULL) {
                    tokenCount++;
                    if (tokenCount == 1) {
                        sscanf(token, "ml_layer_W=%d", &layerNum);
                    } else {
                        int x = atol(token);
                        weight_indices[layerNum].insert(x);
                        // fprintf(stderr,"found x=%d\n",x);
                    }
                    token = strtok(NULL, ",");
                }
                
            }
        }
    }

    void read_indices(int mac_x,int mac_y){

        fprintf(stderr,"Read Faulty Indices\n");
        DLALayerNum=0;
        std::string indexfilename = "./FIlocations/SAFI.indices."+std::to_string(mac_x)+"."+std::to_string(mac_y)+".txt";
        IndexFile = fopen(indexfilename.c_str(), "r");
        if (IndexFile == NULL) {
            fprintf(stderr, "ERROR: Unable to open (read) Index stat file - %s\n",
                    IndexFile);
            exit(1);
        }
        read_all_indices();


        fprintf(stderr,"read layer_SAConfig\n");
        char Layer_SAConfigFileName[80];
        strncpy(Layer_SAConfigFileName, "my.layer_SA.config.txt", 80);
        Layer_SAConfigFile = fopen(Layer_SAConfigFileName, "r");
        if (Layer_SAConfigFile == NULL) {
            fprintf(stderr, "ERROR: Unable to open (read) Index stat file ! %s\n",
                    Layer_SAConfigFile);
            exit(1);
        }
        char line[100];
        while (fgets(line, sizeof(line), Layer_SAConfigFile)) {
            if (strncmp(line, "total_layers", 12) == 0) {
                    line[strcspn(line, "\n")] = 0;
                    char* arg = strtok(line, "=");
                    char* value = strtok(NULL, "");
                    total_layers =  atoll(value);
                    fprintf(stderr,"|||||**||||||total_layers=%d \n",(int)total_layers);
            }
            if (strncmp(line, "fi_layer", 8) == 0) {
                    line[strcspn(line, "\n")] = 0;
                    char* arg = strtok(line, "=");
                    char* value = strtok(NULL, "");
                    fi_layer =  atoll(value);
                    fprintf(stderr,"|||||**||||||fi_layer=%d \n",(int)fi_layer);
            }
        }
        fclose(Layer_SAConfigFile);
        fprintf(stderr,"done1\n");
    }


    //FI functions:
    void DLAWeightIndex(int64_t x){
        do_fi = (fi_layer == -1 | (DLALayerNum%total_layers == fi_layer)) & (!(weight_indices[DLALayerNum].find(x) == weight_indices[DLALayerNum].end()));

        myfault += do_fi;
    }

    bool do_fiMAC(){return do_fi;}

    void LLTFIInjectFault(char* operatorConfig, int8_t* outputPtr,
                      int64_t outputRank, int8_t* outputShape,
                      int8_t* outputStrides, int64_t inputShapeRank,
                      int8_t* inputShapePtr) {
        // fprintf(stderr,"num faults in layer %d: %d \n",DLALayerNum,myfault);
        myfault = 0;
        DLALayerNum++;   
    }

    void LLTFIInjectFaultMatMul(char* operatorConfig, int8_t* outputPtr,
                                int64_t outputRank, int8_t* outputShape,
                                int8_t* outputStrides, int64_t input1ShapeRank,
                                int8_t* input1ShapePtr, int64_t input2ShapeRank,
                                int8_t* input2ShapePtr) {
        // fprintf(stderr,"num faults in layer %d: %d \n",DLALayerNum,myfault);
        myfault = 0;
        DLALayerNum++;
    }









    //profiling pass functions:

     void locate_weight_indices(int layerNum, int MAC_x, int MAC_y, int Dim, int C,int H,int W, int M){

        // fprintf(stderr,"locate_weight indices layerNum= %d MAC_x= %d MAC_y= %d Dim= %d C= %d H= %d W= %d M= %d\n",layerNum,MAC_x,MAC_y,Dim,C,H,W,M);
        // fprintf(stderr,"W\tH\tC\tM\tindex\n");
        // fprintf(stderr,"-----------------------------\n");

        for(int i = 0 ; i < M; i++){
            if(i%Dim != MAC_x)
                continue;
            for(int j=0;j<C;j++){
                for(int h1=0;h1<H;h1++){
                    if((j*H+h1)%(Dim-Dim%H) != MAC_y)
                        continue;
                    for(int w1=0;w1<W;w1++){
                        int ind = w1+ h1*W+ j*W*H+ i*W*H*C;
                        weight_indices_prof[MAC_x/sampler][MAC_y/sampler][layerNum].insert(ind);
                        // fprintf(stderr,"%d\t%d\t%d\t%d\t%d(%ld) \t %d\n",w1,h1,j,i,ind,weight_indices_prof[MAC_x][MAC_y][layerNum].size(),(h1*C+j));
                    }   
                }
            }
        }

    }
    void locate_Matmulweight_indices(int layerNum, int MAC_x, int MAC_y, int Dim, int N, int M){
        
        // fprintf(stderr,"locate_Matmulweight indices layerNum= %d MAC_x= %d MAC_y= %d Dim= %d N= %d M= %d\n",layerNum,MAC_x,MAC_y,Dim,N,M);
        // fprintf(stderr,"N\tM\tindex\n");
        // fprintf(stderr,"-----------------------------\n");
        
        for(int i = 0 ; i < N; i++){
            if(i%Dim != MAC_y)
                continue;
            for(int j=0;j<M;j++){
                if(j%Dim != MAC_x)
                    continue;
                int ind = j+i*M; 
                weight_indices_prof[MAC_x/sampler][MAC_y/sampler][layerNum].insert(ind);
                // fprintf(stderr,"%d\t%d\t%d(%ld)\n",i,j,ind,weight_indices_prof[MAC_x][MAC_y][layerNum].size());

            
            }
        }

    }

    void LLTFIInjectFaultProf(char* operatorConfig, int8_t* outputPtr,
                        int64_t outputRank, int8_t* outputShape,
                        int8_t* outputStrides, int64_t inputShapeRank,
                        int8_t* inputShapePtr) {

        ConvolutionOp *op = new ConvolutionOp();

        // Deserialize the operator configuration.
        char *temp = (char*)malloc(100 * sizeof(char));
        // fprintf(stderr,"file is %s -\n",operatorConfig);
        sscanf(operatorConfig,
                "%s %s %ld %ld %ld %ld %s %ld %ld %s %ld %ld %s %ld %ld %ld %ld",
                op->operationName, temp, &(op->kernelSize[0]), &(op->kernelSize[1]),
                &(op->kernelSize[2]), &(op->kernelSize[3]), temp, &(op->strides[0]),
                &(op->strides[1]), temp, &(op->dilations[0]), &(op->dilations[1]),
                temp, &(op->paddings[0]), &(op->paddings[1]), &(op->paddings[2]),
                &(op->paddings[3]));
        free(temp);

        // Set output of the operator.
        op->outputRank = outputRank;
        op->outputPtr = (float*)outputPtr;
        op->outputShape = (int64_t*)outputShape;
        op->outputStride = (int64_t*)outputStrides;

        // Set input of the operator.
        op->inputShapeRank = inputShapeRank;
        op->inputShapePtr = (int64_t*)inputShapePtr;

        fprintf(stderr,"1conv ops:  %s kernel(%ld, %ld, %ld, %ld) stride(%ld, %ld)  dilations(%ld, %ld) padding(%ld %ld %ld %ld)\n", op->operationName, 
                (op->kernelSize[0]), (op->kernelSize[1]), (op->kernelSize[2]), (op->kernelSize[3]),
                (op->strides[0]), (op->strides[1]),  (op->dilations[0]), (op->dilations[1]),
                (op->paddings[0]), (op->paddings[1]), (op->paddings[2]),(op->paddings[3]));   
        fprintf(stderr," 2input (%ld %ld %ld %ld)\n",op->inputShapePtr[0],op->inputShapePtr[1],op->inputShapePtr[2],op->inputShapePtr[3]);  
        fprintf(stderr," 3output (%ld %ld %ld %ld)\n",op->outputShape[0],op->outputShape[1],op->outputShape[2],op->outputShape[3]);  


        assert((op->dilations[0] == 1 && op->dilations[1] == 1) && "Dilations are not supported");
        // assert(op->kernelSize[1] == op->inputShapePtr[1] && "The input channels are not equal to Filter channels");
        assert( (op->inputShapePtr[1] % op->kernelSize[1] == 0) && "The input channels are not equal to Filter channels (or can not be grouped)");
        assert(op->kernelSize[0] == op->outputShape[1] && "The output channels are not equal to Number of Filters");
        
        read_SA_config();
        fprintf(stderr,"here?\n");
        if(isDataflowWS){
            for(int i = 0; i < SAdim; i++){
                for(int j = 0; j < SAdim; j++){
                    if(i % sampler == 0  &&  j % sampler == 0)
                    locate_weight_indices(DLALayerNum,i,j,SAdim,op->kernelSize[1],op->kernelSize[2],op->kernelSize[3],op->kernelSize[0]);
                }
            }
        }
        
        fprintf(stderr,"here now?\n");
        DLALayerNum++;
    }

    void LLTFIInjectFaultMatMulProf(char* operatorConfig, int8_t* outputPtr,
                                int64_t outputRank, int8_t* outputShape,
                                int8_t* outputStrides, int64_t input1ShapeRank,
                                int8_t* input1ShapePtr, int64_t input2ShapeRank,
                                int8_t* input2ShapePtr) {
        // fprintf(stderr,"YO2\n");
        MatMulOp *op = new MatMulOp();

        // Deserialize the operator configuration.
        sscanf(operatorConfig, "%s", op->operationName);

        // Set output of the operator.
        op->outputRank = outputRank;
        op->outputPtr = (float*)outputPtr;
        op->outputShape = (int64_t*)outputShape;
        op->outputStride = (int64_t*)outputStrides;

        // Set input of the operator.
        op->input1ShapeRank = input1ShapeRank;
        op->input1ShapePtr = (int64_t*)input1ShapePtr;
        op->input2ShapeRank = input2ShapeRank;
        op->input2ShapePtr = (int64_t*)input2ShapePtr;
        fprintf(stderr,"MatMul ops:  %s \n", op->operationName);
        fprintf(stderr," input1 (");   
        for (int i = 0 ; i < op->input1ShapeRank;i++){
            fprintf(stderr,"%ld ",op->input1ShapePtr[i]);   
        }
        fprintf(stderr,")\n");   
        fprintf(stderr," input2 (");   
        for (int i = 0 ; i < op->input2ShapeRank;i++){
            fprintf(stderr,"%ld ",op->input2ShapePtr[i]);   
        }
        fprintf(stderr,")\n");
        fprintf(stderr," output (");   
        for (int i = 0 ; i < op->outputRank;i++){
            fprintf(stderr,"%ld ",op->outputShape[i]);   
        }
        fprintf(stderr,")\n");   
        
        read_SA_config();
        if(isDataflowWS){
            for(int i = 0; i < SAdim; i++){
                for(int j = 0; j < SAdim; j++){
                    if(i % sampler == 0  &&  j % sampler == 0)
                        locate_Matmulweight_indices(DLALayerNum,i,j,SAdim,op->input2ShapePtr[0],op->input2ShapePtr[1]);
                }
            }
        }
        DLALayerNum++;
    }

}

void DLAPrintIndices(){
    fprintf(stderr,"yo whats up! %d \n",int(isDataflowWS));
    if(isDataflowWS){
        for(int MAC_x = 0; MAC_x < SAdim/sampler; MAC_x++){
            for(int MAC_y = 0; MAC_y < SAdim/sampler; MAC_y++){
                FILE *indexFile;
                std::string indexfilename = "./FIlocations/SAFI.indices."+std::to_string(MAC_x)+"."+std::to_string(MAC_y)+".txt";
                // fprintf(stderr,"writing into file %s \n",indexfilename.c_str());
                indexFile = fopen(indexfilename.c_str(), "w");
                if (indexFile == NULL) {
                    fprintf(stderr, "ERROR: Unable to open index file ** %s\n",
                            indexfilename);
                    exit(1);
                }
            // weight_indices_prof[MAC_x][MAC_y]
                fprintf(indexFile, "# do not edit %d\n",SETSIZE);
                for( int i = 0; i < SETSIZE; i++){
                    if(weight_indices_prof[MAC_x][MAC_y][i].size()){
                        // fprintf(stderr,"weight_indices_prof[MAC_x][MAC_y][i].size() %d %ld\n",i,weight_indices_prof[MAC_x][MAC_y][i].size());
                        fprintf(indexFile, "ml_layer_W=%d,",i);
                        int cnt = 0;
                        for(auto x : weight_indices_prof[MAC_x][MAC_y][i]){
                            fprintf(indexFile, "%d,",x);
                            cnt++;
                            if(cnt==200){ //set a threshold so that each line doesn't get too large
                                cnt = 0;
                                fprintf(indexFile, "-1\n");
                                fprintf(indexFile, "ml_layer_W=%d,",i);
                            }
                        }
                        fprintf(indexFile, "-1\n");
                    }
                    
                }

                fclose(indexFile);
            }
        }
    }
}