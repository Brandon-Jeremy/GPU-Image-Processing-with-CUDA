# OptiFX - Image Processing with CUDA C

OptiFX is a C CUDA code designed to provide various image processing services, including Image Blur, Brighten, Darken, RGB Channel Correction, Color Enhancement, Sharpening, Edge Detection, and Grayscale conversion.

## Usage

OptiFX uses a text file that follows a specific syntax:

`input.png algorithmname output.png`

- `inputimg.png`: The input image file.
- `algorithmname`: The desired image processing algorithm to apply.
- `outputimg.png`: The output image file.

To prepare the text file for execution, a preprocessor script needs to be run first:

`python preprocessor.py`


The preprocessor ensures that the text file is sorted in a specific order. This step optimizes the GPU performance by avoiding unnecessary memory deallocation and reallocation for images every time the kernel operations are executed.

After the preprocessor is run, the text file is ready for use by the C CUDA code.

Inside the C CUDA code, a separate job is created for each image and its corresponding algorithm. This allows efficient parallel execution of image processing tasks on the GPU.

## Getting Started

To get started with OptiFX, follow these steps:

1. Install the necessary dependencies, including CUDA and the required CUDA libraries.

2. Clone the OptiFX repository to your local machine.

3. Place your input images in the appropriate directory.

4. Edit the input text file to specify the desired image processing tasks.

5. Run the preprocessor script using the provided command:

`python preprocessor.py`


6. Compile and run the C CUDA code using your preferred compiler.

7. Check the output directory to find the processed images.
