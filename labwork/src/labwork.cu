#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTHa aaaa ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);

    std::string inputFilename;
    std::string inputFilename2;
    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if ((lwNum != 2) ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
      if (lwNum == 6){
                inputFilename2 = std::string(argv[3]);
                labwork.loadInputImage2(inputFilename2);
                int width =  labwork.getWidth();
                int height = labwork.getHeight();
                char * cmd = (char *)malloc(sizeof(char)*100); ;
                sprintf(cmd, "convert %s -resize %dx%d %s",argv[3], width, height, argv[3]);
                const char * command = cmd;
                system(command);

                labwork.loadInputImage2(inputFilename2);

	}
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
 	    printf("labwork 1 OpenMP ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());

            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
   	    printf("labwork 3 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 4:
	printf("cava");
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
 	printf("labwork 4 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 5:
	//    labwork.labwork5_CPU();
        //    labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork52_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
         case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}
void Labwork::loadInputImage2(std::string inputFileName) {
    inputImage2 = jpegLoader.load(inputFileName);
}
int Labwork::getWidth() {
	return inputImage->width;
}

int Labwork::getHeight() {
	return inputImage->height;
}
void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));


    // do something here
#pragma omp parralel for
 for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }


    }

}


int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
	printf("%s\n", prop.name);
	printf("Clock Rate: %d\n", prop.clockRate);
	printf("Total global memory: %u", prop.totalGlobalMem);
	printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
	printf("Number of cores:     %d\n",getSPcores(prop) );
	printf("Warp size: %d\n", prop.warpSize);
	// something more here
    }

}

__global__ void grayscale2(uchar3 *input, uchar3 *output, int width, int height)
{
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int w=  width ;
	int tid = r*w + c;
	if (c<width)
	{
		if(r < height)
		{
			output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
			output[tid].z = output[tid].y = output[tid].x;
		}
	}
}

__global__ void grayscale(uchar3 *input, uchar3 *output) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
	output[tid].z = output[tid].y = output[tid].x;
}
void Labwork::labwork3_GPU() {


    // Calculate number of pixels
        int pixelCount = inputImage->width * inputImage->height;
        outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
	uchar3 *devInput;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput,pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
	cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing
	int blockSize = 64;
	int numBlock = pixelCount / blockSize;
        printf("numblock %d\n", numBlock);
	grayscale<<<numBlock, blockSize>>>(devInput, devOutput);

    // Copy CUDA Memory from GPU to CPU
	cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
	cudaFree(devInput);
	cudaFree(devOutput);
}
__global__ void blur(uchar3 *input, uchar3 *output, int width, int height)
{
	int convolution [7][7]={{0,0,1,2,1,0,0},{0,3,13,22,13,3,0},{1,13,59,97,59,13,1},{2,22,97,159,97,22,2},{1,13,59,97,59,13,1},{0,3,13,22,13,3,0},{0,0,1,2,1,0,0}};
        int tidx = blockIdx.x*blockDim.x + threadIdx.x;
        int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = tidy*width+tidx;
	if (tidx<width)
        {
                if(tidy < height)
                {
			int left=0;
			int right=7;
			int up=0;
			int bottom=7;
			int somme=0;
			if (tidx<3)
			{
				left=3-tidx;
			}
			 if (tidy<3)
                        {
                                up=3-tidy;
                        }
			  if (width-tidx<3)
                        {
                                right=4+width-tidx;
                        }
			  if (height-tidy<3)
                        {
                                bottom=4+height-tidy;
                        }

			for (int haut= up ; haut < bottom ; haut++)
			{
				 for (int gauche=left ; gauche < right ; gauche++)
	                        {
					somme +=(input[(tidy-3+haut)*width + (tidx-3+gauche)].x)*convolution[haut][gauche];
                	        }

			}
			int coeff=0;
			for (int j =0; j < 7 ; j++)
                        {
                                 for (int i=0 ; i < 7 ; i++)
                                {
                                        coeff +=convolution[j][i];
                                }

                        }
			output[tid].x= somme/coeff;
			output[tid].z = output[tid].y = output[tid].x;


                }
        }
}
__global__ void blur2(uchar3 *input, uchar3 *output, int width, int height )
{
	__shared__ char tile[32][32];
        int convolution [7][7]={{0,0,1,2,1,0,0},{0,3,13,22,13,3,0},{1,13,59,97,59,13,1},{2,22,97,159,97,22,2},{1,13,59,97,59,13,1},{0,3,13,22,13,3,0},{0,0,1,2,1,0,0}};
        int tidx = blockIdx.x*blockDim.x + threadIdx.x;
        int tidy = blockIdx.y*blockDim.y + threadIdx.y;
        int tid = tidy*width+tidx;

	tile[threadIdx.y][threadIdx.x] = input[tid].x;

	__syncthreads();

        if (tidx<width )
        {
                if(tidy < height )
                {
                        int left=0;
                        int right=7;
                        int up=0;
                        int bottom=7;
                        int somme=0;
                        if (threadIdx.x<3)
                        {
                                left=3-threadIdx.x;
                        }
                        if (threadIdx.y <3)
                        {
                                up=3-threadIdx.y;
                        }
                        if (blockDim.x-threadIdx.x<3)
                        {
                                right=4+blockDim.x-threadIdx.x;
                        }
                        if (blockDim.y-threadIdx.y <3)
                        {
                                bottom=4+blockDim.y-threadIdx.y;
                        }

                        for (int haut= up ; haut < bottom ; haut++)
                        {
                                 for (int gauche=left ; gauche < right ; gauche++)
                                {
                                        somme +=(tile[threadIdx.y -3+haut][threadIdx.x -3+gauche])*convolution[haut][gauche];
                                }

                        }
                        int coeff=0;
                        for (int j =0; j < 7 ; j++)
                        {
                                 for (int i=0 ; i < 7 ; i++)
                                {
                                        coeff +=convolution[j][i];
                                }

			  }

                        output[tid].x= somme/coeff;
                        output[tid].z = output[tid].y = output[tid].x;

                }
        }
	 __syncthreads();
}

void Labwork::labwork4_GPU() {
    // Calculate number of pixels
        int pixelCount = inputImage->width * inputImage->height;
        outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
        uchar3 *devInput;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput,pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing
	dim3 blockSize = dim3(32, 32);
	int width =inputImage->width / blockSize.x;
	int height=inputImage->height / blockSize.y;

	if ((inputImage->width % blockSize.x)>0)
	{
		width++;
	}
	if ((inputImage->height % blockSize.y)>0)
        {
		height++;
	}
	dim3 gridSize = dim3(width, height);
	grayscale2<<<gridSize, blockSize>>>(devInput, devOutput , inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU

        cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
        cudaFree(devInput);
        cudaFree(devOutput);
}

void Labwork::labwork5_CPU() {

}

void Labwork::labwork5_GPU() {
  // Calculate number of pixels
        int pixelCount = inputImage->width * inputImage->height;
        outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
        uchar3 *devInput;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput,pixelCount * sizeof(uchar3));
	uchar3 *devGray;
        cudaMalloc(&devGray, pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing
        dim3 blockSize = dim3(32, 32);
        int width =inputImage->width / blockSize.x;
        int height=inputImage->height / blockSize.y;

        if ((inputImage->width % blockSize.x)>0)
        {
                width++;
        }
        if ((inputImage->height % blockSize.y)>0)
        {
                height++;
        }
        dim3 gridSize = dim3(width, height);
        grayscale2<<<gridSize, blockSize>>>(devInput, devGray , inputImage->width, inputImage->height);
        blur<<<gridSize, blockSize>>>(devGray, devOutput , inputImage->width, inputImage->height);
    // Copy CUDA Memory from GPU to CPU


        cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
        cudaFree(devInput);
        cudaFree(devOutput);
	cudaFree(devGray);
}
void Labwork::labwork52_GPU() {
        int pixelCount = inputImage->width * inputImage->height;
        outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
        uchar3 *devInput;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput,pixelCount * sizeof(uchar3));
        uchar3 *devGray;
        cudaMalloc(&devGray, pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing
        dim3 blockSize = dim3(32, 32);
        int width =inputImage->width / blockSize.x;
        int height=inputImage->height / blockSize.y;

        if ((inputImage->width % blockSize.x)>0)
        {
                width++;
        }
        if ((inputImage->height % blockSize.y)>0)
        {
                height++;
        }
        dim3 gridSize = dim3(width, height);
        grayscale2<<<gridSize, blockSize>>>(devInput, devGray , inputImage->width, inputImage->height);
        blur2<<<gridSize, blockSize>>>(devGray, devOutput , inputImage->width, inputImage->height);
    // Copy CUDA Memory from GPU to CPU


        cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
        cudaFree(devInput);
        cudaFree(devOutput);
        cudaFree(devGray);
}
__global__ void brightness(uchar3 *input, uchar3 *output, int width, int height, int bright)
{
        int c = blockIdx.x*blockDim.x + threadIdx.x;
        int r = blockIdx.y*blockDim.y + threadIdx.y;
        int w=  width ;
        int tid = r*w + c;
        if (c<width)
        {
                if(r < height)
                {
                        output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3+ bright;
                        if (output[tid].x<0)
                        {
                                output[tid].x=0;
                        }
                        if (output[tid].x>255)
                        {
                                output[tid].x=255;
                        }
			output[tid].z = output[tid].y = output[tid].x;
                }
        }
}
__global__ void binerization(uchar3 *input, uchar3 *output, int width, int height)
{
        int c = blockIdx.x*blockDim.x + threadIdx.x;
        int r = blockIdx.y*blockDim.y + threadIdx.y;
        int w=  width ;
        int tid = r*w + c;
        if (c<width)
        {
                if(r < height)
                {
                        output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
                        if (output[tid].x<128)
                        {
                                output[tid].z = output[tid].y = output[tid].x=0;
                        }
                        else
                        {
                                output[tid].z = output[tid].y = output[tid].x=255;
                        }
                }
        }
}
__global__ void blending(uchar3 *input, uchar3 *input2, uchar3 *output, int width, int height,double cons)
{
        int c = blockIdx.x*blockDim.x + threadIdx.x;
        int r = blockIdx.y*blockDim.y + threadIdx.y;
        int w=  width ;
        int tid = r*w + c;
        if (c<width)
        {
                if(r < height)
                {
                        output[tid].x = input[tid].x*c +input2[tid].x*(1-c);
			output[tid].y = input[tid].y*c +input2[tid].y*(1-c);
			output[tid].z = input[tid].z*c +input2[tid].z*(1-c);

                }
        }
}


void Labwork::labwork6_GPU() {
  // Calculate number of pixels
        int pixelCount = inputImage->width * inputImage->height;
        outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
        uchar3 *devInput;
	uchar3 *devInput2;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devInput2, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput,pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
	cudaMemcpy(devInput2, inputImage2->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing
        dim3 blockSize = dim3(16, 16);
        int width =inputImage->width / blockSize.x;
        int height=inputImage->height / blockSize.y;

        if ((inputImage->width % blockSize.x)>0)
        {
                width++;
        }
        if ((inputImage->height % blockSize.y)>0)
        {
                height++;
        }
        dim3 gridSize = dim3(width, height);
        //binerization<<<gridSize, blockSize>>>(devInput, devOutput , inputImage->width, inputImage->height);
	//brightness<<<gridSize, blockSize>>>(devInput, devOutput , inputImage->width, inputImage->height,20);
	blending<<<gridSize, blockSize>>>(devInput,devInput2, devOutput , inputImage->width, inputImage->height,0.5);

    // Copy CUDA Memory from GPU to CPU

        cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
        cudaFree(devInput);
        cudaFree(devOutput);

}

__global__ void reduceFinal(uchar3 *input, uchar3 *output, int width) {
  extern __shared__ int cache[];
	int min =0;
	unsigned int c = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int r = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int tid = r*width + c;
 	unsigned int localtid = threadIdx.x;

	cache[localtid] = input[tid].x;
	__syncthreads();

	// reduction in cache
	for (int s = 1; s < blockDim.x; s *= 2) {
		int index = s * 2 * localtid;
		if (index < blockDim.x) {
			//min
			if (cache[localtid] > cache[localtid + s]){
				cache[localtid] = cache[localtid + s];
			}
		}
	}
        	 __syncthreads();

        // only first thread writes back
        if (localtid == 0){
                min = cache[0];

        }
int max=255;
   for (int s = 1; s < blockDim.x; s *= 2) {
                int index = s * 2 * localtid;
                if (index < blockDim.x) {
                        //min
                        if (cache[localtid] < cache[localtid + s]){
                                cache[localtid] = cache[localtid + s];
                        }
                }
        }
                 __syncthreads();
        // only first thread writes back
        if (localtid == 0){
                max = cache[0];

        }


	output[tid].x = (255*(input[tid].x - min))/(max-min) ;
	output[tid].z = output[tid].y = output[tid].x ;
}

void Labwork::labwork7_GPU() {

  // Calculate number of pixels
        int pixelCount = inputImage->width * inputImage->height;
        outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
        uchar3 *devInput;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput,pixelCount * sizeof(uchar3));
        uchar3 *devGray;
        cudaMalloc(&devGray, pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing
        dim3 blockSize = dim3(32, 32);
        int width =inputImage->width / blockSize.x;
        int height=inputImage->height / blockSize.y;

        if ((inputImage->width % blockSize.x)>0)
        {
                width++;
        }
        if ((inputImage->height % blockSize.y)>0)
        {
                height++;
        }
int  sharedMemSize=256;
        dim3 gridSize = dim3(width, height);
        grayscale2<<<gridSize, blockSize>>>(devInput, devGray , inputImage->width, inputImage->height);
        reduceFinal<<<gridSize, blockSize, sharedMemSize>>>(devGray, devOutput, inputImage->width );
    // Copy CUDA Memory from GPU to CPU


        cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
        cudaFree(devInput);
        cudaFree(devOutput);
        cudaFree(devGray);

}
__global__ void RGBtoHSV(uchar3 *input, double *outh, double *outs, double *outv, int width, int height)
{
        int c = blockIdx.x*blockDim.x + threadIdx.x;
        int r = blockIdx.y*blockDim.y + threadIdx.y;
        int w=  width ;
        int tid = r*w + c;
	double delta=0;
	outh[tid] = 4.0;
        if (c<width)
        {
                if(r < height)
                {
			double red =(double) input[tid].x/255;
			double green =(double) input[tid].y/255;
			double blue = (double) input[tid].z/255;
			double color[3] = {red,green,blue};
                        double max =color[0];
			double min = color[0];
			for (int i =1; i<3 ; i++)
			{
				if(color[i]>max)
				{
					max =color[i];
				}
				if(color[i]<min)
                                {
                                        min =color[i];
                                }

			}
			delta =max-min;
			if (delta != 0)
			{
				if(max = color[0])
				{
					outh[tid] = 60*(((int)((green - blue)/delta)) % 6) ;
				}
				 if(max =color[1])
                                {
					outh[tid] = 60*((blue - red)/delta+2) ;
                                }
				 if(max =color[2])
                                {
					outh[tid] = 60*((red - green)/delta+4) ;
                                }

			}
			outv[tid] = max ;
			if (max == 0){
				outs[tid] = 0 ;
			}
			else
			{
				outs[tid] = delta / max ;
			}
                }
        }
}
__global__ void HSVtoRGB(double *outh, double *outs, double *outv, uchar3 *output, int width,int height)
{
	int c = blockIdx.x*blockDim.x + threadIdx.x;
        int r = blockIdx.y*blockDim.y + threadIdx.y;
        int w=  width ;
        int tid = r*w + c;

	double H = outh[tid] ;
	double S = outs[tid] ;
	double V = outv[tid] ;
	double d = H / 60 ;
	double hi =(int) d % 6 ;
	double f = d - hi ;
	double l = V *(1-S) ;
	double m = V*(1-(f*S)) ;
	double n = V*(1-((1-f)*S)) ;
	if ((H >= 0) and (H<60))
	{
		output[tid].x = V*255 ; output[tid].y = n*255 ; output[tid].z = l*255 ;
	}
	if ((H>= 60) and (H <120))
	{
		output[tid].x = m*255 ; output[tid].y = V*255 ; output[tid].z = l*255 ;
	}
	if ((H>= 120) and (H <180))
	{
		output[tid].x = l*255 ; output[tid].y = V*255 ; output[tid].z = n*255 ;
	}
	if ((H>= 180) and (H < 240))
	{
		output[tid].x = l*255 ; output[tid].y = m*255 ; output[tid].z = V*255 ;
	}
	if ((H>= 240) and (H < 300))
	{
		output[tid].x = n*255 ; output[tid].y = l*255 ; output[tid].z = V*255 ;
	}
	if ((H>= 300) and (H <= 360))
	{
		output[tid].x = V*255 ; output[tid].y = l*255 ; output[tid].z = m*255 ;
	}
}
void Labwork::labwork8_GPU() {
  // Calculate number of pixels
        int pixelCount = inputImage->width * inputImage->height;
        outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
        uchar3 *devInput;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput,pixelCount * sizeof(uchar3));
        uchar3 *devGray;
        cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
	double *outh;
        double *outs;
        double *outv;

	cudaMalloc(&outh, pixelCount * sizeof(double));
        cudaMalloc(&outs, pixelCount * sizeof(double));
        cudaMalloc(&outv, pixelCount * sizeof(double));
    // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing
        dim3 blockSize = dim3(32, 32);
        int width =inputImage->width / blockSize.x;
        int height=inputImage->height / blockSize.y;

        if ((inputImage->width % blockSize.x)>0)
        {
                width++;
        }
        if ((inputImage->height % blockSize.y)>0)
        {
                height++;
        }
        dim3 gridSize = dim3(width, height);
	RGBtoHSV<<<gridSize, blockSize>>>(devInput,outh, outs, outv,inputImage->width, inputImage->height);
	HSVtoRGB<<<gridSize, blockSize>>>(outh, outs, outv, devOutput ,inputImage->width, inputImage->height);
    // Copy CUDA Memory from GPU to CPU


        cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
        cudaFree(devInput);
        cudaFree(devOutput);
        cudaFree(devGray);

}
__global__ void histogramme(uchar3 *input, uchar3 *output, int width, int height, int bright)
{
        int c = blockIdx.x*blockDim.x + threadIdx.x;
        int r = blockIdx.y*blockDim.y + threadIdx.y;
        int w=  width ;
        int tid = r*w + c;
        if (c<width)
        {
                if(r < height)
                {
			}	}
}
void Labwork::labwork9_GPU() {
	// Calculate number of pixels
        int pixelCount = inputImage->width * inputImage->height;

	outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
        uchar3 *devInput;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput,pixelCount * sizeof(uchar3));
        uchar3 *devGray;
        cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
        double *outh;
        double *outs;
        double *outv;

	cudaMalloc(&outh, pixelCount * sizeof(double));
        cudaMalloc(&outs, pixelCount * sizeof(double));
        cudaMalloc(&outv, pixelCount * sizeof(double));
    // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing
        dim3 blockSize = dim3(32, 32);
        int width =inputImage->width / blockSize.x;
        int height=inputImage->height / blockSize.y;

        if ((inputImage->width % blockSize.x)>0)
        {
                width++;
        }
        if ((inputImage->height % blockSize.y)>0)
        {
                height++;
        }
        dim3 gridSize = dim3(width, height);
	grayscale2<<<gridSize, blockSize>>>(devInput, devGray , inputImage->width, inputImage->height);
        RGBtoHSV<<<gridSize, blockSize>>>(devInput,outh, outs, outv,inputImage->width, inputImage->height);
        HSVtoRGB<<<gridSize, blockSize>>>(outh, outs, outv, devOutput ,inputImage->width, inputImage->height);
    // Copy CUDA Memory from GPU to CPU


        cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
        cudaFree(devInput);
        cudaFree(devOutput);
        cudaFree(devGray);

}
__global__ void kumahara(int w , double *outh, double *outs, double *outv,uchar3 *input, uchar3 *output, int width,int height)
{
        int co = blockIdx.x*blockDim.x + threadIdx.x;
        int r = blockIdx.y*blockDim.y + threadIdx.y;
        int tid = r*width + co;
	double a = 0.0 ;
	double b = 0.0 ;
	double c = 0.0 ;
	double d = 0.0 ;
	int red =0;
	int green=0;
	int blue =0;
	double wSquare=(w+1)*(w+1);
	if ((co > w) and (co < width-(w+1)) and (r > w) and (r< height-(w+1)))
	{
   		for(int i=0; i<w+1; ++i)
		{
			for (int j =0; j<w+1;j++)
			{
				a += outv[(tid-i)-(j*width) ] ;
				b += outv[(tid+i)-(j*width) ] ;
				c += outv[(tid-i)+(j*width) ] ;
				d += outv[(tid+i)+(j*width) ] ;
	  		}
		}
		a = a/wSquare;
	        b = b/wSquare;
        	c = c/wSquare;
        	d = d/wSquare;

		for(int i=0; i<w+1 ; ++i)
		{
			for (int j=0;j<w+1 ; j++)
			{
        			a += outv[(tid-i)-(j*width) ] - a;
        			b += outv[(tid+i)-(j*width) ] - b;
        			c += outv[(tid-i)+(j*width) ] - c;
        			d += outv[(tid+i)+(j*width) ] - d;
	  		}
		}
		double comparaison[4]={a,b,c,d};
		double min = comparaison[0];
		if(min<0)
                {
	                min*= -1;
                }

		for (int i=0;i<4;i++)
		{
			if(comparaison[i]<0)
			{
				comparaison[i]*= -1;
			}
			if (comparaison[i]<min)
			{
				min = comparaison[i];
			}
		}
		int factI =1;
		int factJ=1;
		if(min = a)
		{
			factI = -1;
                	factJ= -1;
		}
		if(min = d)
                {
                        factJ= -1;
                }
		if(min = c)
                {
                        factI = -1;
                }
		for(int i=0; i<w+1 ; ++i)
                {
                        for (int j=0;j<w+1 ; j++)
                        {
				red +=input[(tid+ factI*i)+(factJ*j*width) ].x ;
		        	green += input[(tid+ factI*i)+(factJ*j*width) ].y ;
        			blue += input[(tid+ factI*i)+(factJ*j*width) ].z ;
			}
		}
		red = red/wSquare;
                blue = blue/wSquare ;
                green = green/wSquare;
		for(int i=0; i<w+1 ; ++i)
                {
                        for (int j=0;j<w+1 ; j++)
                        {
                                output[(tid+ factI*i)+(factJ*j*width) ].x=red ;
                                output[(tid+ factI*i)+(factJ*j*width) ].y=green ;
                                output[(tid+ factI*i)+(factJ*j*width) ].z=blue ;
                        }
                }


	}
	else
	{
		output[tid].x=input[tid].x;
		output[tid].y=input[tid].y;
		output[tid].z=input[tid].z;
	}
}

void Labwork::labwork10_GPU(){
   // Calculate number of pixels
        int pixelCount = inputImage->width * inputImage->height;

        outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory
        uchar3 *devInput;
        uchar3 *devOutput;
        cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devOutput,pixelCount * sizeof(uchar3));
        uchar3 *devGray;
        cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
        double *outh;
        double *outs;
        double *outv;

        cudaMalloc(&outh, pixelCount * sizeof(double));
        cudaMalloc(&outs, pixelCount * sizeof(double));
        cudaMalloc(&outv, pixelCount * sizeof(double));
    // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing
        dim3 blockSize = dim3(32, 32);
        int width =inputImage->width / blockSize.x;
        int height=inputImage->height / blockSize.y;

        if ((inputImage->width % blockSize.x)>0)
        {
                width++;
        }
        if ((inputImage->height % blockSize.y)>0)
        {
                height++;
        }
        dim3 gridSize = dim3(width, height);
        RGBtoHSV<<<gridSize, blockSize>>>(devInput,outh, outs, outv,inputImage->width, inputImage->height);
        kumahara<<<gridSize, blockSize>>>(20,outh, outs, outv,devInput, devOutput ,inputImage->width, inputImage->height);
    // Copy CUDA Memory from GPU to CPU


        cudaMemcpy(outputImage, devOutput,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);

    // Cleaning
        cudaFree(devInput);
        cudaFree(devOutput);
        cudaFree(devGray);

}
































