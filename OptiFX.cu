#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <sys/time.h>
#include <string.h>

typedef struct
{
    int height;
    int width;
    int pixel_size;
    png_infop info_ptr;
    png_byte *buf;
} PNG_RAW;

typedef struct {
    char input_file[256];
    char algorithm[256];
    char output_file[256];
} Job;

long long timeInMilliseconds(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}

PNG_RAW *read_png(char *file_name)
{
    PNG_RAW *png_raw = (PNG_RAW *)malloc(sizeof(PNG_RAW));

    FILE *fp = fopen(file_name, "rb");
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    int pixel_size = png_get_rowbytes(png_ptr, info_ptr) / width;
    png_raw->width = width;
    png_raw->height = height;
    png_raw->pixel_size = pixel_size;
    png_raw->buf = (png_byte *)malloc(width * height * pixel_size * sizeof(png_byte));
    png_raw->info_ptr = info_ptr;
    int k = 0;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width * pixel_size; j++)
        {
            png_raw->buf[k++] = row_pointers[i][j];
        }
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    fclose(fp);
    return png_raw;
}

void write_png(char *file_name, PNG_RAW *png_raw)
{
    FILE *fp = fopen(file_name, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_init_io(png_ptr, fp);
    png_infop info_ptr = png_raw->info_ptr;
    int width = png_raw->width;
    int height = png_raw->height;
    int pixel_size = png_raw->pixel_size;
    png_bytepp row_pointers;
    row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
    for (int i = 0; i < height; i++)
        row_pointers[i] = (png_bytep)malloc(width * pixel_size);
    int k = 0;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width * pixel_size; j++)
        {
            row_pointers[i][j] = png_raw->buf[k++];
        }

    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    for (int i = 0; i < height; i++)
        free(row_pointers[i]);
    free(row_pointers);
    fclose(fp);
}

__global__ void GreyscaleKernel(png_byte *d_P, int height, int width)
{
    // Calculate the row # of the d_P element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_P element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread computes one element of d_P if in range
    if ((Row < height) && (Col < width))
    {
        // calculate the index of the pixel in the buffer
        int idx = (Row * width + Col) * 3;

        // calculate the grayscale value for the pixel
        png_byte gray = (png_byte)((d_P[idx] + d_P[idx+1] + d_P[idx+2]) / 3);

        // set the red, green, and blue values to the grayscale value
        d_P[idx] = gray;
        d_P[idx+1] = gray;
        d_P[idx+2] = gray;
    }
}

__global__ void SharpenKernel(png_byte* d_P, int height, int width)
{
    // Calculate the row # of the d_P element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_P element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of d_P if in range
    if ((Row < height) && (Col < width))
    {
      int index = (Row * width + Col) * 3;
      int weights[3][3] = { { 0, -1, 0 }, { -1, 5, -1 }, { 0, -1, 0 } };

      int sum_red = 0;
      int sum_green = 0;
      int sum_blue = 0;

      int sharpenSize = 30;

      //convolve the matrix with the image
      for (int i = -sharpenSize; i <= sharpenSize; i++)
      {
          for (int j = -sharpenSize; j <= sharpenSize; j++)
          {
              int currRow = Row + i;
              int currCol = Col + j;

              int weight = weights[i + 1][j + 1];
	      if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width)
              {
              	int sharpenIndex = (currRow * width + currCol) * 3;
              	sum_red += weight * d_P[sharpenIndex];
              	sum_green += weight * d_P[sharpenIndex + 1];
              	sum_blue += weight * d_P[sharpenIndex + 2];
	      }
          }
      }

      int max_val = 255;
      int min_val = 0;

      //normalize the output to [0, 255]
      int new_red = max(min_val, min(max_val, sum_red));
      int new_green = max(min_val, min(max_val, sum_green));
      int new_blue = max(min_val, min(max_val, sum_blue));

      d_P[index] = (png_byte)new_red;
      d_P[index + 1] = (png_byte)new_green;
      d_P[index + 2] = (png_byte)new_blue;
    }
}

__global__ void GrayscaleKernel(png_byte *d_P, int height, int width)
{
    // Calculate the row # of the d_P element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_P element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread computes one element of d_P if in range
    if ((Row < height) && (Col < width))
    {
        // calculate the index of the pixel in the buffer
        int idx = (Row * width + Col) * 3;

        // calculate the grayscale value for the pixel
        png_byte gray = (png_byte)((d_P[idx] + d_P[idx+1] + d_P[idx+2]) / 3);

        // set the red, green, and blue values to the grayscale value
        d_P[idx] = gray;
        d_P[idx+1] = gray;
        d_P[idx+2] = gray;
    }
}

__global__ void BlurKernel(png_byte* d_P, int height, int width)
{
    // Calculate the row # of the d_P element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_P element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread computes one element of d_P if in range
    if ((Row < height) && (Col < width))
    {
        int index = (Row * width + Col) * 3;

        int blurSize = 50; 

        int avg_red = 0;
        int avg_green = 0;
        int avg_blue = 0;

        //blur by averaging pixel values within the blurSize neighborhood
        for (int i = -blurSize; i <= blurSize; i++)
        {
            for (int j = -blurSize; j <= blurSize; j++)
            {
                int currRow = Row + i;
                int currCol = Col + j;

                //makee sure  pixel is within image boundaries
                if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width)
                {
                    int blurIndex = (currRow * width + currCol) * 3;
                    avg_red += d_P[blurIndex];
                    avg_green += d_P[blurIndex + 1];
                    avg_blue += d_P[blurIndex + 2];
                }
            }
        }

        int numPixels = (2 * blurSize + 1) * (2 * blurSize + 1);
        int avgRed = avg_red / numPixels;
        int avgGreen = avg_green / numPixels;
        int avgBlue = avg_blue / numPixels;

        d_P[index] = (png_byte)avgRed;
        d_P[index + 1] = (png_byte)avgGreen;
        d_P[index + 2] = (png_byte)avgBlue;
    }
}

__global__ void EdgeDetectionKernel(png_byte* d_P, int height, int width)
{
    // Calculate the row # of the d_P element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_P element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread computes one element of d_P if in range
    if ((Row < height) && (Col < width))
    {
        // calculate the index of the pixel in the buffer
        int idx = (Row * width + Col) * 3;

        // calculate the grayscale value for the pixel
        png_byte gray = (png_byte)((d_P[idx] + d_P[idx+1] + d_P[idx+2]) / 3);

        // set the red, green, and blue values to the grayscale value
        d_P[idx] = gray;
        d_P[idx+1] = gray;
        d_P[idx+2] = gray;

        // perform edge detection using a simple kernel
        int kernel[3][3] = {
            {-1, -1, -1},
            {-1, 8, -1},
            {-1, -1, -1}
        };

        int edge = 0;
        int edgeSize = 5;
        for (int i = -edgeSize; i <= edgeSize; i++) {
            for (int j = -edgeSize; j <= edgeSize; j++) {
                int neighborRow = Row + i;
                int neighborCol = Col + j;
                if ((neighborRow >= 0) && (neighborRow < height) && (neighborCol >= 0) && (neighborCol < width)) {
                    int neighborIdx = (neighborRow * width + neighborCol) * 3;
                    int neighborGray = (d_P[neighborIdx] + d_P[neighborIdx+1] + d_P[neighborIdx+2]) / 3;
                    edge += neighborGray * kernel[i + 1][j + 1];
                }
            }
        }

        // set the grayscale value to the calculated edge value
        d_P[idx] = edge;
        d_P[idx+1] = edge;
        d_P[idx+2] = edge;
    }
}

__global__ void RGBCorrectionKernel(png_byte *d_P, int height, int width)
{
    // Calculate the row # of the d_P element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_P element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread computes one element of d_P if in range
    if ((Row < height) && (Col < width))
    {
        int index = (Row * width + Col) * 3;

        // Retrieve the original RGB values
        int red = d_P[index];
        int green = d_P[index + 1];
        int blue = d_P[index + 2];

        // Apply RGB channel correction
        int correctedRed = red + 10;   
        int correctedGreen = green - 5; 
        int correctedBlue = blue;       

        // Clamp the corrected values to the valid range (0-255)
        correctedRed = min(max(correctedRed, 0), 255);
        correctedGreen = min(max(correctedGreen, 0), 255);
        correctedBlue = min(max(correctedBlue, 0), 255);

        // Update the RGB values in the image
        d_P[index] = (png_byte)correctedRed;
        d_P[index + 1] = (png_byte)correctedGreen;
        d_P[index + 2] = (png_byte)correctedBlue;
    }
}

__global__ void BrightenKernel(png_byte *d_P, int height, int width)
{
    // Calculate the row # of the d_P element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_P element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread computes one element of d_P if in range
    if ((Row < height) && (Col < width))
    {
        d_P[(Row * width + Col) * 3 + 1] = (png_byte)min((d_P[(Row * width + Col) * 3 + 1]*2),255);
        d_P[(Row * width + Col) * 3 + 2] = (png_byte)min((d_P[(Row * width + Col) * 3 + 2]*2),255);
        d_P[(Row * width + Col) * 3 + 3] = (png_byte)min((d_P[(Row * width + Col) * 3 + 3]*2),255);
    }
}

void process_on_device(PNG_RAW *png_raw, char *Algorithm)
{
    int m = png_raw->height;
    int n = png_raw->width;
    int pixel_size = png_raw->pixel_size;

    dim3 DimGrid((n - 1) / 16 + 1, (m - 1) / 16 + 1, 1);
    dim3 DimBlock(16, 16, 1);

    png_byte *d_P;
    cudaError_t err;

    long long start = timeInMilliseconds();

    err = cudaMalloc((void **)&d_P, m * n * pixel_size * sizeof(png_byte));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_P, png_raw->buf, m * n * pixel_size, cudaMemcpyHostToDevice);

    if (strcmp(Algorithm, "Grayscale") == 0)
    {
        GrayscaleKernel<<<DimGrid, DimBlock>>>(d_P, m, n);
    }
    else if (strcmp(Algorithm, "Brightness") == 0)
    {
        BrightenKernel<<<DimGrid, DimBlock>>>(d_P, m, n);
    }
    else if (strcmp(Algorithm, "RGBChannel") == 0)
    {
        RGBCorrectionKernel<<<DimGrid, DimBlock>>>(d_P, m, n);
    }
    else if (strcmp(Algorithm, "Blur") == 0)
    {
        BlurKernel<<<DimGrid, DimBlock>>>(d_P, m, n);
    }
    else if (strcmp(Algorithm, "Sharpen") == 0)
    {
        SharpenKernel<<<DimGrid, DimBlock>>>(d_P, m, n);
    }
    else if (strcmp(Algorithm, "EdgeDetection") == 0)
    {
        EdgeDetectionKernel<<<DimGrid, DimBlock>>>(d_P, m, n);
    }
    else
    {
        printf("Unsupported Algorithm\n");
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(png_raw->buf, d_P, m * n * pixel_size, cudaMemcpyDeviceToHost);

    long long end = timeInMilliseconds();

    printf("Timing on Device is %lld millis\n", end - start);

    cudaFree(d_P);
}


void parse_jobs(const char *filename, Job *jobs, int *num_jobs) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error with file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[1024];
    *num_jobs = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        char algorithm[64];
        sscanf(line, "%s %s %s", jobs[*num_jobs].input_file, algorithm, jobs[*num_jobs].output_file);

        if (strcmp(algorithm, "Brightness") == 0) {
            strcpy(jobs[*num_jobs].algorithm, "Brightness");
        } else if (strcmp(algorithm, "Grayscale") == 0) {
            strcpy(jobs[*num_jobs].algorithm, "Grayscale");
        } else if (strcmp(algorithm, "RGBChannel") == 0) {
            strcpy(jobs[*num_jobs].algorithm, "RGBChannel");
        } else if (strcmp(algorithm, "Blur") == 0) {
            strcpy(jobs[*num_jobs].algorithm, "Blur");
        } else if (strcmp(algorithm, "Sharpen") == 0) {
            strcpy(jobs[*num_jobs].algorithm, "Sharpen");
        } else if (strcmp(algorithm, "EdgeDetection") == 0) {
            strcpy(jobs[*num_jobs].algorithm, "EdgeDetection");
        }
        else {
            fprintf(stderr, "Error: Unknown algorithm: %s", algorithm);
            exit(EXIT_FAILURE);
        }

        (*num_jobs)++;
        if (*num_jobs >= 128) {
            fprintf(stderr, "Error: Maximum number of jobs exceeded (128)\n");
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}

int main(int argc, char **argv)
{

    if (argc != 2) {
        printf("Error: Usage: %s <file.txt>\n",argv[0]);
        exit(EXIT_FAILURE);
    }

    Job jobs[128];
    int num_jobs;

    parse_jobs(argv[1], jobs, &num_jobs);

    for (int i = 0; i < num_jobs; i++) {
        PNG_RAW *png_raw = read_png(jobs[i].input_file);

        if (png_raw->pixel_size != 3)
        {
            printf("Error, png file must be on 3 Bytes per pixel\n");
            exit(0);
        }
        else
            printf("Processing Image of %d x %d pixels\n", png_raw->width, png_raw->height);

        printf("%s", jobs[i].algorithm);

        process_on_device(png_raw, jobs[i].algorithm);

        write_png(jobs[i].output_file, png_raw);

        printf("Processing finished for job %d\n", i + 1);
    }
}

