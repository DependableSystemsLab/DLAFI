#define OPTION_LENGTH 512
#define MULTIPLE_CYCLE_LENGTH 100
#include <iostream>
#include <map>
#include <cstdlib>
#include <time.h>
#include <bitset>

class TensorOperator {
public:
	// Operator configurations.
  char operationName[10];

	// Operator output.
  float* outputPtr;
  int64_t outputRank;
  int64_t* outputShape;
  int64_t* outputStride;

	virtual void print() = 0;
};

// Structure of the convolution operator.
// Dimensions are in [N x C x H x W] format.
class ConvolutionOp : public TensorOperator {
public:
	// Operator configurations.
	long kernelSize[4];
  long strides[2];
  long paddings[4];
  long dilations[2];

  // Operator input.
	int64_t inputShapeRank;
  int64_t* inputShapePtr;

	void print() {
		printf("\n---------------------------\n");
		printf("Operator Name: %s\n", this->operationName);
		printf("Input Shape: [");
		for (int i = 0; i < this->inputShapeRank; i++)
			printf("%ld ,", this->inputShapePtr[i]);
		printf("]\n");
		printf("Kernel Shape: [%ld, %ld, %ld, %ld]\n",
				this->kernelSize[0], this->kernelSize[1],
				this->kernelSize[2], this->kernelSize[3]);
		printf("Strides: [%ld, %ld]\n", this->strides[0],
				this->strides[1]);
		printf("Padding: [%ld, %ld, %ld, %ld]\n",
				this->paddings[0], this->paddings[1],
				this->paddings[2], this->paddings[3]);
		printf("Dilations: [%ld, %ld]\n", this->dilations[0],
				this->dilations[1]);
		printf("Output Shape: [");
		for (int i = 0; i < this->outputRank; i++)
			printf("%ld ,", this->outputShape[i]);
		printf("]\n");
		printf("Output Strides: [");
		for (int i = 0; i < this->outputRank; i++)
			printf("%ld ,", this->outputStride[i]);
		printf("]\n");
		printf("---------------------------\n");
	}
};

// Structure for the matrix multiplication or Gemm operator.
// Dimensions are in [N x C x H x W] format.
class MatMulOp : public TensorOperator {
public:
  // Operator inputs.
  int64_t input1ShapeRank;
  int64_t* input1ShapePtr;
  int64_t input2ShapeRank;
  int64_t* input2ShapePtr;

	void print() {
		printf("\n---------------------------\n");
		printf("Operator Name: %s\n", this->operationName);
		printf("Input1 Shape: [");
		for (int i = 0; i < this->input1ShapeRank; i++)
			printf("%ld ,", this->input1ShapePtr[i]);
		printf("]\n");
		printf("Input2 Shape: [");
		for (int i = 0; i < this->input2ShapeRank; i++)
			printf("%ld ,", this->input2ShapePtr[i]);
		printf("]\n");
		printf("Output Shape: [");
		for (int i = 0; i < this->outputRank; i++)
			printf("%ld ,", this->outputShape[i]);
		printf("]\n");
		printf("Output Strides: [");
		for (int i = 0; i < this->outputRank; i++)
			printf("%ld ,", this->outputStride[i]);
		printf("]\n");
		printf("---------------------------\n");
	}
};

// Structure to hold content of the config file.
struct Config {
  //=========================Original Fault Config================
  char fi_type[OPTION_LENGTH];
  bool fi_accordingto_cycle;
  long long fi_cycle;
  long fi_index;
  int fi_reg_index;
  int fi_bit;
  int fi_num_bits;
  long long fi_second_cycle;
  int fi_max_multiple; //JUNE 3rd
  long long fi_next_cycles[MULTIPLE_CYCLE_LENGTH];
  //==========================Fault Pattern Config================
  char deviceType[10];
  int systolicArraySize;
  char dataMappingScheme[3];
  char faultType[10];
  char faultPrecision[15];
  char fp_fi_type[10];
  int fp_fi_max_multiple;
  int fp_fi_bit;
  int fp_fi_cycle;
};
static Config runtimeConfig = {"bitflip", false, -1, -1, -1, -1, 1, -1, -1, {-1},
            "TPU", 16, "WS", "Transient", "Approximate", "bitflip",
            1, -1, 1};

static bool hasParsedConfig = false;
static int fi_next_cycles_count = 0;

// Inject fault pattern in the output of the tensor.
void injectFaultPattern(TensorOperator*);

// Function to parse LLFI's config file.
void ParseLLFIConfigFile();




struct Strategy {
    std::vector<std::array<int, 3>> conditions;
    std::vector<int> X;
    std::vector<int> Y;
    std::vector<int> DivisibleTiles;
};

struct Mapping {
    std::vector<Strategy> strategies;
};

static std::vector<Mapping> SA_mapping_conv, SA_mapping_matmul;